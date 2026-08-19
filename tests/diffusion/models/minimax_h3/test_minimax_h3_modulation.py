# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    indexed_scale_shift_,
    rms_norm_indexed_scale_shift,
)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float()
    variance = normalized.square().mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return (weight.float() * normalized).to(x.dtype)


def test_minimax_h3_modulation_cpu_fallback_with_strided_adaln() -> None:
    """The fallback preserves H3's two-step BF16 rounding contract."""
    dtype = torch.bfloat16
    eps = 1e-6
    x = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
        dtype=dtype,
    )
    branch = torch.tensor(
        [[0.5, -1.0, 1.5, -2.0], [2.5, -3.0, 3.5, -4.0]],
        dtype=dtype,
    )
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=dtype)
    indices = torch.tensor([0, 2], dtype=torch.int64)

    adaln = torch.arange(3 * 6 * x.shape[-1], dtype=torch.float32).reshape(3, 6 * x.shape[-1]).to(dtype) / 128
    shift, scale, gate, _, _, _ = adaln.chunk(6, dim=-1)
    assert not shift.is_contiguous()
    assert shift.stride() == (6 * x.shape[-1], 1)

    expected_scale_shift = (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)
    actual_scale_shift = indexed_scale_shift_(x.clone(), shift, scale, indices)
    torch.testing.assert_close(actual_scale_shift, expected_scale_shift, atol=0, rtol=0)

    expected_gate = (x + gate.index_select(0, indices) * branch).to(dtype)
    actual_gate = indexed_gate(x, gate, branch, indices)
    torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)

    normalized = _rms_norm(x, weight, eps)
    expected_rms_mod = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)
    actual_rms_mod = rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    torch.testing.assert_close(actual_rms_mod, expected_rms_mod, atol=0, rtol=0)

    expected_residual = expected_gate
    normalized_residual = _rms_norm(expected_residual, weight, eps)
    expected_residual_mod = (
        normalized_residual * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)
    actual_residual, actual_residual_mod = indexed_gate_rms_norm_scale_shift(
        x,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        eps,
    )
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(actual_residual_mod, expected_residual_mod, atol=0, rtol=0)
