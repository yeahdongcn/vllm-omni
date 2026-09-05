# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.magi2 import mh_moe


@pytest.mark.parametrize("block_m", [32, 64, 128])
def test_route_metadata_matches_selected_gemm_tile(monkeypatch, block_m):
    monkeypatch.setenv("MAGI2_SGL_BLOCK_M", str(block_m))

    class StopBeforeGemm(Exception):
        pass

    def check_align(ids, experts, actual_block):
        assert actual_block == block_m
        flat = ids.flatten()
        sorted_ids, expert_ids, padded = mh_moe._magi2_align_block_size_fixed_capacity(
            flat, experts, actual_block
        )
        published = int(padded.item())
        valid_routes = []
        for b, ex in enumerate(expert_ids[: published // actual_block].tolist()):
            routes = sorted_ids[b * actual_block : (b + 1) * actual_block]
            routes = routes[routes < flat.numel()].long()
            assert torch.all(flat[routes] == ex)
            valid_routes.extend(routes.tolist())
        assert sorted(valid_routes) == list(range(flat.numel()))
        raise StopBeforeGemm

    monkeypatch.setattr(mh_moe, "_magi2_align_block_size", check_align)
    tokens, heads, experts, hidden, intermediate = 37, 2, 7, 16, 16
    x = torch.zeros(tokens, heads, hidden)
    probs = torch.ones(heads, tokens, 6) / 6
    ids = (torch.arange(heads * tokens * 6) % experts).reshape(heads, tokens, 6)
    gate = torch.zeros(heads * experts, hidden, intermediate)
    down = torch.zeros(heads * experts, intermediate, hidden)
    with pytest.raises(StopBeforeGemm):
        mh_moe._magi2_sgl_fused_moe_forward(x, probs, ids, gate, gate, down)


@pytest.mark.parametrize(
    "variable,value",
    [
        ("MAGI2_SGL_BLOCK_M", "0"),
        ("MAGI2_SGL_BLOCK_M", "48"),
        ("MAGI2_SGL_BLOCK_N", "1"),
        ("MAGI2_SGL_BLOCK_K", "-16"),
        ("MAGI2_SGL_NUM_WARPS", "3"),
        ("MAGI2_SGL_NUM_STAGES", "0"),
        ("MAGI2_SGL_GROUP_M", "0"),
    ],
)
def test_invalid_tile_rejected_before_device_work(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        mh_moe._magi2_sgl_moe_config()
