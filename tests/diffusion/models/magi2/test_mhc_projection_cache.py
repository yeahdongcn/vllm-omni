# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.magi2.layers import MHCHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


@pytest.mark.cpu
def test_compiled_handler_keeps_projection_weights_distinct():
    handler = MHCHandler(4, 2)
    convert = torch.compile(handler._bf16_phi, backend="eager", fullgraph=True)
    attention = torch.ones(8, 24)
    moe = torch.full((8, 24), 2.0)
    for weight in (attention, moe, attention):
        torch.testing.assert_close(convert(weight), weight.bfloat16(), rtol=0, atol=0)
    assert not hasattr(handler, "_compiled_phi_fused_bf16")
    attention.add_(3)
    torch.testing.assert_close(convert(attention), attention.bfloat16(), rtol=0, atol=0)


@pytest.mark.cpu
def test_eager_projection_cache_remains_version_aware():
    handler = MHCHandler(4, 2)
    weight = torch.ones(8, 24)
    first = handler._bf16_phi(weight)
    assert handler._bf16_phi(weight) is first
    weight.add_(1)
    second = handler._bf16_phi(weight)
    assert second is not first
    torch.testing.assert_close(second, weight.bfloat16(), rtol=0, atol=0)


@pytest.mark.musa
def test_musa_graph_projection_is_not_persisted_on_handler():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    from torch_musa._inductor import musagraph_trees

    handler = MHCHandler(4, 2)
    convert = torch.compile(handler._bf16_phi, fullgraph=True, options={"triton.cudagraphs": True})
    with torch.inference_mode():
        weights = [torch.full((8, 24), float(i), device="musa") for i in (1, 2)]
        for i in range(8):
            musagraph_trees.mark_step_begin()
            weight = weights[i % 2]
            actual = convert(weight).clone()
            torch.musa.synchronize()
            torch.testing.assert_close(actual, weight.bfloat16(), rtol=0, atol=0)
            del actual
    assert not hasattr(handler, "_compiled_phi_fused_bf16")
