# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_omni.diffusion.distributed.parallel_state as state
import vllm_omni.diffusion.models.magi2.parallel as magi_parallel
from tests.diffusion.distributed.test_head_expert_parallel import _initialize
from tests.diffusion.models.magi2.test_native_distributed_parity import (
    _initialize_model,
    _inputs,
    _patched_groups,
    _tiny_config,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.mh_moe import Magi2MultiHeadMoE, Magi2MultiHeadMoEConfig
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer
from vllm_omni.diffusion.models.magi2.parallel import (
    Magi2ParallelGroup,
    get_magi2_ep_group,
    get_magi2_ep_split_indices,
    get_magi2_ulysses_group,
)
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import _validate_native_topology

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _config(sp=4, cfg=1, ep=2, tp=1):
    return OmniDiffusionConfig(
        model_class_name="Magi2Pipeline",
        dtype=torch.float32,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp,
            ulysses_degree=sp,
            cfg_parallel_size=cfg,
            data_parallel_size=1,
            enable_expert_parallel=True,
            expert_parallel_size=ep,
        ),
    )


@pytest.mark.cpu
def test_opt_in_and_topology_guard():
    assert _config().use_head_expert_parallel
    _validate_native_topology(_config())
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _validate_native_topology(_config(tp=2))


@pytest.mark.cpu
def test_explicit_ep_does_not_fall_back_before_initialization():
    with set_forward_context(omni_diffusion_config=_config()):
        with patch.object(dist, "is_initialized", return_value=False):
            with pytest.raises(RuntimeError, match="Initialize framework"):
                get_magi2_ep_group()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "sp_members,ep_members,expected",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7], (4, 5, 6, 7)),
        ([4, 5, 6, 7], [4, 5, 6, 7], (0, 1, 2, 3)),
        ([1, 3, 5, 7], [5, 7], (2, 3)),
        ([4, 5, 6, 7], [6], (2,)),
    ],
)
def test_split_projection_uses_sp_coordinates(monkeypatch, sp_members, ep_members, expected):
    sp_pg, ep_pg = object(), object()
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda pg: sp_members if pg is sp_pg else ep_members)
    assert (
        get_magi2_ep_split_indices(
            Magi2ParallelGroup(ep_pg, len(ep_members), 0), Magi2ParallelGroup(sp_pg, len(sp_members), 0)
        )
        == expected
    )


@pytest.mark.cpu
def test_projection_rejects_cross_replica_group(monkeypatch):
    sp_pg, ep_pg = object(), object()
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda pg: [0, 1] if pg is sp_pg else [0, 2])
    with pytest.raises(ValueError, match="contained in its SP group"):
        get_magi2_ep_split_indices(Magi2ParallelGroup(ep_pg, 2, 0), Magi2ParallelGroup(sp_pg, 2, 0))
    assert (
        get_magi2_ep_split_indices(
            Magi2ParallelGroup(ep_pg, 2, 0, replicated_sequence=True), Magi2ParallelGroup(sp_pg, 2, 0)
        )
        is None
    )


def _moe_config(dtype=torch.float32):
    return Magi2MultiHeadMoEConfig(
        hidden_size=48,
        num_heads=3,
        num_experts=2,
        top_k=2,
        expert_intermediate_size=16,
        params_dtype=dtype,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("ep_size,rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_resident_and_mmap_head_slices_share_padding(ep_size, rank):
    module = Magi2MultiHeadMoE(_moe_config(), ep_group=Magi2ParallelGroup(None, ep_size, rank))
    for name, parameter in module.named_parameters():
        full = torch.arange(6 * parameter[0].numel(), dtype=parameter.dtype).reshape(6, *parameter.shape[1:])
        expected = torch.zeros_like(parameter)
        start = rank * module.local_num_heads * 2
        real = full[start : start + parameter.shape[0]]
        expected[: real.shape[0]] = real
        torch.testing.assert_close(module.ep_slice(full), expected, rtol=0, atol=0)
        transform = getattr(parameter, "mmap_weight_transform")
        torch.testing.assert_close(transform(full), expected, rtol=0, atol=0)


def _gloo_gather_sequence(tensor, split_sizes, group):
    # Gloo rejects uneven all_gather tensors. Pad only the final test transport;
    # all DiT/MoE compute and internal collectives still use production code.
    if group.world_size == 1:
        return tensor
    padded = tensor.new_zeros((max(split_sizes), *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in split_sizes]
    dist.all_gather(gathered, padded, group=group.group)
    return torch.cat([chunk[:size] for chunk, size in zip(gathered, split_sizes)])


def _model_case(od, token_count, cfg_rank):
    singleton = Magi2ParallelGroup(None, 1, 0)
    with _patched_groups(singleton, singleton):
        reference = Magi2PreviewTransformer(_tiny_config())
        _initialize_model(reference)
        checkpoint = [(name, value.detach().clone()) for name, value in reference.state_dict().items()]
        packed, coords, modalities, _ = _inputs()
        packed = packed[:token_count] + cfg_rank * 0.1
        coords, modalities = coords[:token_count], modalities[:token_count]
        cu = torch.tensor([0, token_count], dtype=torch.int32)
        inputs = (packed, coords, modalities, VarlenHandler(cu, cu, token_count, token_count))
        with torch.no_grad():
            expected = reference(*inputs)
    with set_forward_context(omni_diffusion_config=od):
        model = Magi2PreviewTransformer(_tiny_config())
        model.load_weights(checkpoint)
        sp = get_magi2_ulysses_group()
        ep = get_magi2_ep_group()
        quotient, remainder = divmod(token_count, sp.world_size)
        splits = [quotient + (index < remainder) for index in range(sp.world_size)]
        start = sp.rank // ep.world_size * ep.world_size
        expected_splits = splits[start : start + ep.world_size]
        seen = []

        def record(_module, _args, kwargs):
            seen.append(kwargs.get("sequence_split_sizes"))

        handles = [
            module.register_forward_pre_hook(record, with_kwargs=True)
            for module in model.modules()
            if isinstance(module, Magi2MultiHeadMoE)
        ]
        try:
            with torch.no_grad(), patch.object(magi_parallel, "gather_sequence", _gloo_gather_sequence):
                actual = model(*inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            assert seen and all(counts == expected_splits for counts in seen)
        finally:
            for handle in handles:
                handle.remove()


def _moe_case(od, token_count, cfg_rank, device):
    dtype = torch.bfloat16 if device == "musa" else torch.float32
    reference = Magi2MultiHeadMoE(_moe_config(dtype), ep_group=Magi2ParallelGroup(None, 1, 0))
    generator = torch.Generator().manual_seed(73)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
        reference.router.expert_bias.copy_(reference.router.expert_bias_ema)
    inputs = (torch.randn(token_count, 48, generator=generator) + cfg_rank * 0.1).to(dtype)
    with torch.no_grad():
        expected = reference(inputs)
    with set_forward_context(omni_diffusion_config=od):
        module = Magi2MultiHeadMoE(_moe_config(dtype)).to(device)
        for name, parameter in module.named_parameters():
            with torch.no_grad():
                parameter.copy_(module.ep_slice(reference.state_dict()[name]).to(device))
        sp, ep = get_magi2_ulysses_group(), module.ep_group
        quotient, remainder = divmod(token_count, sp.world_size)
        splits = [quotient + (index < remainder) for index in range(sp.world_size)]
        indices = get_magi2_ep_split_indices(ep, sp)
        counts = [splits[index] for index in indices]
        offset, count = sum(splits[: sp.rank]), splits[sp.rank]
        local = inputs[offset : offset + count].to(device)
        with (
            torch.no_grad(),
            patch.object(dist, "all_gather", side_effect=AssertionError("unexpected count discovery")),
        ):
            actual = module(local, sequence_split_sizes=counts).cpu()
        torch.testing.assert_close(
            actual,
            expected[offset : offset + count],
            rtol=0.03 if device == "musa" else 1e-5,
            atol=0.0004 if device == "musa" else 1e-5,
        )


def _worker(rank, rendezvous, device):
    torch.set_num_threads(1)
    backend = "gloo" if device == "cpu" else "mccl"
    if device == "musa":
        torch.accelerator.set_device_index(rank)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        state.init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend=backend)
        for sp, cfg, ep in ((4, 1, 2), (2, 2, 2), (4, 1, 4), (4, 1, 1)):
            od = _config(sp, cfg, ep)
            with set_forward_context(omni_diffusion_config=od):
                _initialize(od, backend)
            try:
                if device == "cpu":
                    for count in (5, 7):
                        _model_case(od, count, rank // sp)
                for count in (5, 1, 0):
                    _moe_case(od, count, rank // sp, device)
            finally:
                state.destroy_model_parallel()
            dist.barrier()
    finally:
        state.destroy_model_parallel()
        state.destroy_distributed_environment()


@pytest.mark.cpu
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_four_rank_native_model_and_head_ep_oracle(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'adapter-gloo'}", "cpu"), nprocs=4, join=True)


@pytest.mark.musa
def test_four_rank_musa_moe_head_ep_oracle(tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.musa.device_count() < 4:
        pytest.skip("requires four MUSA devices")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'adapter-mccl'}", "musa"), nprocs=4, join=True)
