# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from vllm_omni.diffusion.models.magi2.layers import swiglu7


def test_fused_swiglu7_keeps_cpu_fallback(monkeypatch):
    torch.manual_seed(17)
    x = torch.randn(9, 32, dtype=torch.bfloat16)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "0")
    expected = swiglu7(x)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    torch.testing.assert_close(swiglu7(x), expected, rtol=0, atol=0)


def test_fused_swiglu7_empty_and_odd_inputs_fall_back_safely(monkeypatch):
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    empty = torch.empty(0, 2560, dtype=torch.bfloat16)
    torch.testing.assert_close(
        swiglu7(empty), torch.empty(0, 1280, dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError):
        swiglu7(torch.empty(3, 7, dtype=torch.bfloat16))


def test_fused_swiglu7_boundary_values_use_reference_contract(monkeypatch):
    values = torch.tensor(
        [
            -float("inf"),
            -8.0,
            -7.0,
            -0.0,
            0.0,
            7.0,
            8.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    x = torch.stack((values, values.flip(0)), dim=-1).reshape(1, -1)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "0")
    output = swiglu7(x)
    assert torch.isnan(output).any()
    assert torch.isinf(output).logical_not().all()


@pytest.mark.parametrize("shape", [(33, 2560), (3, 16384), (3, 21840), (2, 514)])
@pytest.mark.skipif(
    not (hasattr(torch.version, "musa") and torch.version.musa is not None),
    reason="requires MUSA",
)
def test_fused_swiglu7_matches_reference_on_musa(monkeypatch, shape):
    visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    if visible:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    torch.musa.set_device(0)
    torch.manual_seed(23)
    x = torch.randn(*shape, device="musa", dtype=torch.bfloat16)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "0")
    expected = swiglu7(x)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    actual = swiglu7(x)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not (hasattr(torch.version, "musa") and torch.version.musa is not None),
    reason="requires MUSA",
)
def test_fused_swiglu7_matches_boundary_nan_mask_on_musa(monkeypatch):
    torch.musa.set_device(0)
    values = torch.tensor(
        [
            -float("inf"),
            -8.0,
            -7.0,
            -0.0,
            0.0,
            7.0,
            8.0,
            float("inf"),
            float("nan"),
        ],
        device="musa",
        dtype=torch.bfloat16,
    )
    x = torch.stack((values, values.flip(0)), dim=-1).reshape(1, -1)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "0")
    expected = swiglu7(x)
    monkeypatch.setenv("MAGI2_USE_FUSED_SWIGLU7", "1")
    actual = swiglu7(x)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
