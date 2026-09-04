# SPDX-License-Identifier: Apache-2.0

"""Fixed-size Triton kernels for MAGI-2 mHC post processing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _mhc_mix_output_kernel(
    streams_ptr,
    block_out_ptr,
    post_ptr,
    res_ptr,
    out_ptr,
    hidden,
    NUM_STREAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < hidden
    offs_n = tl.arange(0, NUM_STREAM)

    acc = tl.zeros([NUM_STREAM, BLOCK_C], dtype=tl.float32)
    for j in tl.static_range(NUM_STREAM):
        stream = tl.load(
            streams_ptr + (token * NUM_STREAM + j) * hidden + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        res = tl.load(
            res_ptr + token * NUM_STREAM * NUM_STREAM + offs_n * NUM_STREAM + j
        )
        acc += res[:, None] * stream[None, :]

    written = tl.load(block_out_ptr + token * hidden + offs_c, mask=mask_c, other=0.0)
    post = tl.load(post_ptr + token * NUM_STREAM + offs_n)
    acc += post[:, None] * written.to(tl.float32)[None, :]

    tl.store(
        out_ptr + (token * NUM_STREAM + offs_n[:, None]) * hidden + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_c[None, :],
    )


@triton.jit
def _mhc_sinkhorn_kernel(
    h_ptr,
    out_ptr,
    eps,
    NUM_STREAM: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """Run the fixed four-stream Sinkhorn normalization per token."""

    token = tl.program_id(0)
    offs = tl.arange(0, 16)
    rows = offs // 4
    cols = offs - rows * 4
    matrix = tl.load(h_ptr + token * NUM_STREAM * NUM_STREAM + offs)
    matrix = tl.exp(matrix - tl.max(matrix, axis=0))
    for _ in tl.static_range(NUM_ITERS):
        col0 = tl.sum(tl.where(cols == 0, matrix, 0.0), axis=0)
        col1 = tl.sum(tl.where(cols == 1, matrix, 0.0), axis=0)
        col2 = tl.sum(tl.where(cols == 2, matrix, 0.0), axis=0)
        col3 = tl.sum(tl.where(cols == 3, matrix, 0.0), axis=0)
        col_sum = tl.where(
            cols == 0,
            col0,
            tl.where(cols == 1, col1, tl.where(cols == 2, col2, col3)),
        )
        matrix = matrix / (col_sum + eps)
        row0 = tl.sum(tl.where(rows == 0, matrix, 0.0), axis=0)
        row1 = tl.sum(tl.where(rows == 1, matrix, 0.0), axis=0)
        row2 = tl.sum(tl.where(rows == 2, matrix, 0.0), axis=0)
        row3 = tl.sum(tl.where(rows == 3, matrix, 0.0), axis=0)
        row_sum = tl.where(
            rows == 0,
            row0,
            tl.where(rows == 1, row1, tl.where(rows == 2, row2, row3)),
        )
        matrix = matrix / (row_sum + eps)
    tl.store(out_ptr + token * NUM_STREAM * NUM_STREAM + offs, matrix)


def _mhc_mix_output(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    if streams.ndim != 3:
        raise ValueError("streams must have shape [tokens, streams, hidden]")
    tokens, num_stream, hidden = streams.shape
    if num_stream != 4:
        raise ValueError("MAGI-2 fused mHC kernel requires exactly 4 streams")
    if block_out.shape != (tokens, hidden):
        raise ValueError("invalid mHC branch output shape")
    if post.shape != (tokens, num_stream):
        raise ValueError("invalid mHC post coefficient shape")
    if residual.shape != (tokens, num_stream, num_stream):
        raise ValueError("invalid mHC residual matrix shape")
    out = torch.empty_like(streams)
    block_c = min(1024, triton.next_power_of_2(hidden))
    _mhc_mix_output_kernel[(tokens, triton.cdiv(hidden, block_c))](
        streams,
        block_out,
        post.contiguous(),
        residual.contiguous(),
        out,
        hidden,
        NUM_STREAM=num_stream,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _mhc_sinkhorn(
    logits: torch.Tensor,
    *,
    num_iters: int,
    eps: float,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("expected [tokens, streams, streams] logits")
    tokens, num_stream, _ = logits.shape
    if num_stream != 4 or num_iters > 64:
        raise ValueError("MAGI-2 fused Sinkhorn requires 4 streams and iters<=64")
    out = torch.empty_like(logits, dtype=torch.float32)
    _mhc_sinkhorn_kernel[(tokens,)](
        logits.contiguous(),
        out,
        eps,
        NUM_STREAM=num_stream,
        NUM_ITERS=num_iters,
        num_warps=1,
    )
    return out



def _mhc_mix_output_fake(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(streams)


def _mhc_sinkhorn_fake(logits: torch.Tensor, *, num_iters: int, eps: float) -> torch.Tensor:
    return torch.empty_like(logits, dtype=torch.float32)


# Register at import time so compiled diffusion regions see an opaque custom op
# instead of a first-call Triton graph break/JIT.
direct_register_custom_op(
    "magi2_mhc_mix_output",
    _mhc_mix_output,
    mutates_args=[],
    fake_impl=_mhc_mix_output_fake,
)
mhc_mix_output = torch.ops.vllm.magi2_mhc_mix_output

direct_register_custom_op(
    "magi2_mhc_sinkhorn",
    _mhc_sinkhorn,
    mutates_args=[],
    fake_impl=_mhc_sinkhorn_fake,
)
mhc_sinkhorn = torch.ops.vllm.magi2_mhc_sinkhorn


__all__ = ["mhc_mix_output", "mhc_sinkhorn"]
