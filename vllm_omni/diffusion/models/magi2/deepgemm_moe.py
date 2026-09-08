# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in MATE BF16 grouped W13; retain the existing sorted W2 path.

Sorted IDs address flattened top-k *route slots*, not input rows. Output rows
retain their original padded sorted coordinates, including unused capacity.
GEMM output rounds to BF16 before SwiGLU7, matching the fused MoE kernel.
"""

from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton


def prepare_sorted_input(
    hidden: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_padded: torch.Tensor,
    top_k: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for the padding/gather contract (also used by CPU tests)."""
    capacity = sorted_ids.numel()
    padded = (capacity + block_m - 1) // block_m * block_m
    slots = sorted_ids.new_full((padded,), hidden.shape[0] * top_k)
    slots[:capacity].copy_(sorted_ids)
    rows = torch.arange(padded, device=hidden.device)
    row_experts = expert_ids[rows // block_m].to(torch.int32)
    active = (rows < num_padded.reshape(())) & (row_experts >= 0)
    valid = active & (slots >= 0) & (slots < hidden.shape[0] * top_k)
    token_rows = (slots // top_k).clamp(0, hidden.shape[0] - 1).long()
    gathered = hidden.index_select(0, token_rows)
    gathered.masked_fill_(~valid[:, None], 0)
    # Keep an expert ID for intra-expert padding; whole unused blocks use -1.
    row_experts = torch.where(active, row_experts, -1)
    return gathered, row_experts, valid


def interleaved_swiglu7(packed: torch.Tensor) -> torch.Tensor:
    """BF16 GEMM output -> FP32 activation -> BF16 intermediate."""
    if packed.ndim != 2 or packed.shape[1] % 2:
        raise ValueError("Expected [rows, 2 * intermediate] interleaved gate/up")
    gate = packed[:, 0::2].float().clamp(max=7.0)
    up = packed[:, 1::2].float().clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(packed.dtype)


@triton.jit
def gather_kernel(
    hidden,
    slots,
    experts,
    count,
    gathered,
    row_experts,
    capacity: tl.constexpr,
    input_rows: tl.constexpr,
    k: tl.constexpr,
    top_k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    padded: tl.constexpr,
    block_r: tl.constexpr,
):
    row = tl.program_id(0) * block_r + tl.arange(0, block_r)
    cols = tl.arange(0, block_k)
    slot = tl.load(slots + row, row < capacity, input_rows * top_k)
    expert = tl.load(experts + row // block_m, row < padded, -1)
    active = (row < tl.load(count)) & (expert >= 0)
    valid = active & (slot >= 0) & (slot < input_rows * top_k)
    values = tl.load(
        hidden + (slot // top_k)[:, None] * k + cols[None, :],
        valid[:, None] & (cols[None, :] < k),
        0,
    )
    tl.store(
        gathered + row[:, None] * k + cols[None, :],
        values,
        (row[:, None] < padded) & (cols[None, :] < k),
    )
    tl.store(row_experts + row, tl.where(active, expert, -1), row < padded)


@triton.jit
def activation_kernel(
    packed,
    slots,
    experts,
    count,
    output,
    capacity: tl.constexpr,
    input_rows: tl.constexpr,
    n: tl.constexpr,
    top_k: tl.constexpr,
    block_m: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    row, col = offsets // n, offsets % n
    slot = tl.load(slots + row, row < capacity, input_rows * top_k)
    expert = tl.load(experts + row // block_m, row < capacity, -1)
    valid = (row < capacity) & (row < tl.load(count)) & (expert >= 0)
    valid = valid & (slot >= 0) & (slot < input_rows * top_k)
    gate = tl.load(packed + row * (2 * n) + 2 * col, valid, 0).to(tl.float32)
    up = tl.load(packed + row * (2 * n) + 2 * col + 1, valid, 0).to(tl.float32)
    gate = tl.minimum(gate, 7.0)
    up = tl.maximum(tl.minimum(up, 7.0), -7.0)
    value = gate * tl.sigmoid(1.702 * gate) * (up + 1.0)
    tl.store(output + offsets, value, offsets < capacity * n)


def _validate(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_padded: torch.Tensor,
    top_k: int,
    block_m: int,
) -> int:
    if top_k < 1 or block_m not in (128, 256):
        raise ValueError("DeepGEMM requires positive top-k and 128/256-row alignment")
    if hidden.dtype != torch.bfloat16 or w13.dtype != torch.bfloat16:
        raise ValueError("Only BF16 W13 is supported")
    if hidden.ndim != 2 or min(hidden.shape) == 0 or not hidden.is_contiguous():
        raise ValueError("Expected nonempty contiguous [head-token rows, hidden] input")
    if (
        w13.ndim != 3
        or not w13.is_contiguous()
        or w13.shape[0] == 0
        or w13.shape[1] == 0
        or w13.shape[1] % 2
        or w13.shape[2] != hidden.shape[1]
    ):
        raise ValueError("W13 must be contiguous [experts, 2 * intermediate, hidden]")
    if any(t.device != hidden.device for t in (w13, sorted_ids, expert_ids, num_padded)):
        raise ValueError("All W13 tensors must be on the same device")
    if (
        sorted_ids.ndim != 1
        or expert_ids.ndim != 1
        or num_padded.numel() != 1
        or any(t.dtype != torch.int32 or not t.is_contiguous() for t in (sorted_ids, expert_ids, num_padded))
    ):
        raise ValueError("Route metadata must be contiguous int32 vectors and a scalar count")
    padded = (sorted_ids.numel() + block_m - 1) // block_m * block_m
    if not padded or expert_ids.numel() < padded // block_m:
        raise ValueError("Expert block metadata does not cover sorted capacity")
    return padded


def sorted_w13(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_padded: torch.Tensor,
    top_k: int,
    block_m: int = 128,
    *,
    backend: str = "mubin",
    gemm: Callable | None = None,
) -> torch.Tensor:
    """Produce the exact padded sorted intermediate consumed by existing W2.

    No sorting, weight packing, expert remapping, route-weight multiplication,
    or sum reduction is changed here. ``gemm`` is injectable for CPU tests.
    """
    padded = _validate(hidden, w13, sorted_ids, expert_ids, num_padded, top_k, block_m)
    if backend not in {"mubin", "mutlass"}:
        raise ValueError("DeepGEMM W13 backend must be explicit: mubin or mutlass")
    if gemm is None:
        try:
            from mate.deep_gemm import m_grouped_bf16_gemm_nt_contiguous
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("MAGI2_USE_DEEPGEMM_W13 requires MATE BF16 grouped GEMM") from exc
        if hidden.device.type not in {"musa", "privateuseone"}:
            raise ValueError("MATE DeepGEMM W13 requires a MUSA device")
        gemm = m_grouped_bf16_gemm_nt_contiguous
    if hidden.device.type == "cpu":
        gathered, row_experts, valid = prepare_sorted_input(hidden, sorted_ids, expert_ids, num_padded, top_k, block_m)
        packed_output = hidden.new_zeros((padded, w13.shape[1]))
        gemm(gathered, w13, packed_output, row_experts, alignment_m=block_m, backend=backend)
        activated = interleaved_swiglu7(packed_output)
        activated.masked_fill_(~valid[:, None], 0)
        return activated[: sorted_ids.numel()]
    gathered = hidden.new_empty((padded, hidden.shape[1]))
    row_experts = expert_ids.new_empty((padded,))
    gather_kernel[((padded + 15) // 16,)](
        hidden,
        sorted_ids,
        expert_ids,
        num_padded,
        gathered,
        row_experts,
        sorted_ids.numel(),
        hidden.shape[0],
        hidden.shape[1],
        top_k,
        block_m,
        1 << (hidden.shape[1] - 1).bit_length(),
        padded,
        16,
    )
    # Unused GEMM rows may remain unwritten; activation masks them before load.
    packed_output = hidden.new_empty((padded, w13.shape[1]))
    gemm(gathered, w13, packed_output, row_experts, alignment_m=block_m, backend=backend)
    intermediate = hidden.new_empty((sorted_ids.numel(), w13.shape[1] // 2))
    activation_kernel[((intermediate.numel() + 1023) // 1024,)](
        packed_output,
        sorted_ids,
        expert_ids,
        num_padded,
        intermediate,
        sorted_ids.numel(),
        hidden.shape[0],
        intermediate.shape[1],
        top_k,
        block_m,
        1024,
    )
    return intermediate
