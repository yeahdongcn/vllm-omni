# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.magi2 import mh_moe
from vllm_omni.diffusion.models.magi2.sgl_fused_moe_kernels import (
    invoke_fused_moe_kernel,
)


def test_sorted_input_rejects_wrong_dtype():
    a = torch.zeros(8, 64)
    with pytest.raises(ValueError, match="plain BF16"):
        invoke_fused_moe_kernel(
            a, a, None, a, None, None, None,
            torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32),
            torch.arange(8, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32), False, 1, {}, torch.bfloat16,
            False, False, False, False, False, a_sorted=True,
        )


def test_sorted_input_requires_padding_capacity():
    a = torch.zeros(8, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="padded route capacity"):
        invoke_fused_moe_kernel(
            a, a, None, a, None, None, None,
            torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32),
            torch.arange(16, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32), False, 1, {}, torch.bfloat16,
            False, False, False, False, False, False, a_sorted=True,
        )


@pytest.mark.parametrize("rows,heads,skew", [(1, 1, False), (129, 2, False), (257, 2, True)])
@pytest.mark.parametrize("filtered_expert", [False, True])
def test_sorted_intermediate_matches_unsorted(rows, heads, skew, filtered_expert, monkeypatch):
    if not getattr(torch.version, "musa", None) or torch.musa.device_count() == 0:
        pytest.skip("requires a leased MUSA GPU")
    torch.manual_seed(42)
    device = "musa:0"
    top_k, local_experts, hidden, intermediate = 6, 8, 256, 1280
    experts = local_experts * heads
    x = torch.randn(rows, heads, hidden, device=device, dtype=torch.bfloat16) * 0.1
    wg = torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16) * 0.02
    wu = torch.randn_like(wg) * 0.02
    wd = torch.randn(experts, intermediate, hidden, device=device, dtype=torch.bfloat16) * 0.02
    ids = torch.stack([torch.randperm(local_experts)[:top_k] for _ in range(rows * heads)])
    if skew:
        ids[:] = torch.arange(top_k)
    ids = ids.reshape(heads, rows, top_k).to(device=device, dtype=torch.int32)
    weights = torch.softmax(torch.randn(heads, rows, top_k, device=device), dim=-1)
    packed = torch.stack([wg.transpose(1, 2), wu.transpose(1, 2)], dim=2).flatten(1, 2)
    packed_down = wd.transpose(1, 2).contiguous()
    if filtered_expert:
        original_align = mh_moe._magi2_align_block_size

        def align_with_filtered_block(*args, **kwargs):
            sorted_ids, expert_ids, num_padded = original_align(*args, **kwargs)
            expert_ids = expert_ids.clone()
            expert_ids[0] = -1
            return sorted_ids, expert_ids, num_padded

        monkeypatch.setattr(mh_moe, "_magi2_align_block_size", align_with_filtered_block)
    monkeypatch.setenv("MAGI2_SGL_SORTED_INTERMEDIATE", "0")
    reference = mh_moe._magi2_sgl_fused_moe_forward(
        x, weights, ids, wg, wu, wd, packed_w13=packed, packed_w2=packed_down
    )
    monkeypatch.setenv("MAGI2_SGL_SORTED_INTERMEDIATE", "1")
    actual = mh_moe._magi2_sgl_fused_moe_forward(
        x, weights, ids, wg, wu, wd, packed_w13=packed, packed_w2=packed_down
    )
    torch.musa.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
