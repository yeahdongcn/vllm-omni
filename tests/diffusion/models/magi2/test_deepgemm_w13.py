# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W13 route-slot, head-bank, padding, and BF16 activation contracts."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/magi2/deepgemm_moe.py"
_spec = importlib.util.spec_from_file_location("deepgemm_w13_contract", _path)
assert _spec is not None and _spec.loader is not None
impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(impl)

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


def metadata() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two input rows per head, top-k=2; flattened banks have two experts/head.
    # Expert 0 owns slots 0/2; expert 3 owns slots 5/7. Padding is not compacted.
    slots = torch.full((263,), 8, dtype=torch.int32)
    slots[:2] = torch.tensor([0, 2])
    slots[128:130] = torch.tensor([5, 7])
    return slots, torch.tensor([0, 3, -1], dtype=torch.int32), torch.tensor([256], dtype=torch.int32)


def cpu_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    ids: torch.Tensor,
    *,
    alignment_m: int,
    backend: str,
) -> None:
    assert alignment_m == 128 and backend == "mubin"
    assert out.dtype == torch.bfloat16
    for expert in range(b.shape[0]):
        selected = ids == expert
        out[selected] = (a[selected].float() @ b[expert].float().T).bfloat16()


@pytest.mark.cpu
def test_route_slots_are_not_input_rows() -> None:
    hidden = torch.arange(64).reshape(4, 16).bfloat16()
    slots, experts, count = metadata()
    gathered, row_experts, valid = impl.prepare_sorted_input(hidden, slots, experts, count, 2, 128)
    assert gathered.shape == (384, 16)
    torch.testing.assert_close(gathered[:2], hidden[:2], rtol=0, atol=0)
    torch.testing.assert_close(gathered[128:130], hidden[2:], rtol=0, atol=0)
    assert valid.sum() == 4
    assert torch.count_nonzero(gathered[~valid]) == 0
    assert torch.all(row_experts[128:256] == 3)
    assert torch.all(row_experts[256:] == -1)


@pytest.mark.cpu
def test_interleaved_swiglu7_clamps_and_rounding() -> None:
    gate = torch.tensor([-12.0, -1.0, 0.0, 6.96875, 7.0, 8.0], dtype=torch.bfloat16)
    up = torch.tensor([-8.0, -7.0, -1.0, 0.0, 7.0, 8.0], dtype=torch.bfloat16)
    packed = torch.stack((gate, up), dim=-1).reshape(1, -1)
    g, u = gate.float().clamp(max=7), up.float().clamp(-7, 7)
    expected = (g * torch.sigmoid(1.702 * g) * (u + 1)).bfloat16().reshape(1, -1)
    torch.testing.assert_close(impl.interleaved_swiglu7(packed), expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_sorted_gemm_keeps_head_bank_and_padded_coordinates() -> None:
    torch.manual_seed(42)
    hidden = torch.randn(4, 16).bfloat16()
    weights = torch.randn(4, 32, 16).bfloat16()
    slots, experts, count = metadata()
    got = impl.sorted_w13(hidden, weights, slots, experts, count, 2, gemm=cpu_gemm)
    assert got.shape == (263, 16)
    for expert, start, input_rows in ((0, 0, hidden[:2]), (3, 128, hidden[2:])):
        rounded = (input_rows.float() @ weights[expert].float().T).bfloat16()
        torch.testing.assert_close(got[start : start + 2], impl.interleaved_swiglu7(rounded), rtol=0, atol=0)
    assert torch.count_nonzero(got[2:128]) == 0
    assert torch.count_nonzero(got[130:]) == 0
    assert torch.isfinite(got).all() and got.std() > 0


@pytest.mark.cpu
@pytest.mark.parametrize("fault", ["dtype", "layout", "block", "topk", "metadata", "backend"])
def test_invalid_contract_rejected(fault: str) -> None:
    hidden = torch.zeros(4, 16).bfloat16()
    weights = torch.zeros(4, 32, 16).bfloat16()
    slots, experts, count = metadata()
    if fault == "dtype":
        hidden = hidden.float()
    if fault == "layout":
        weights = weights.transpose(1, 2)
    if fault == "metadata":
        experts = experts[:1]
    with pytest.raises(ValueError):
        impl.sorted_w13(
            hidden,
            weights,
            slots,
            experts,
            count,
            0 if fault == "topk" else 2,
            64 if fault == "block" else 128,
            backend="auto" if fault == "backend" else "mubin",
            gemm=cpu_gemm,
        )


@pytest.mark.musa
def test_musa_grouped_w13_matches_rounded_reference() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware required")
    torch.manual_seed(42)
    # Exact production K and I; small row count keeps the numerical test bounded.
    hidden = torch.randn(4, 256, dtype=torch.bfloat16)
    weights = (torch.randn(4, 2560, 256) / 16).bfloat16()
    slots, experts, count = metadata()
    reference = impl.sorted_w13(hidden, weights, slots, experts, count, 2, gemm=cpu_gemm)
    result = impl.sorted_w13(
        hidden.to("musa"),
        weights.to("musa"),
        slots.to("musa"),
        experts.to("musa"),
        count.to("musa"),
        2,
    ).cpu()
    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(result).all() and result.std() > 0
    assert torch.count_nonzero(result[2:128]) == 0
    assert torch.count_nonzero(result[130:]) == 0
