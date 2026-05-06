# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

import vllm_omni.diffusion.model_loader.diffusers_loader as diffusers_loader
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DummyPipelineModel(nn.Module):
    def __init__(self, *, source_prefix: str):
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix=source_prefix,
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor.to(dtype=params[name].dtype))
            loaded.add(name)
        return loaded


class _TrackingQuantMethod(QuantizeMethodBase):
    uses_meta_device = True

    def __init__(self):
        self.processed_modules: list[nn.Module] = []

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        raise NotImplementedError

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.processed_modules.append(layer)


class _OnlineQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.linear.quant_method = _TrackingQuantMethod()

    def load_weights(self, weights):
        return set()


def _make_loader_with_weights(weight_names: list[str]) -> DiffusersPipelineLoader:
    loader = object.__new__(DiffusersPipelineLoader)
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0

    def _iter_weights(_model):
        for name in weight_names:
            yield name, torch.zeros((2, 2))

    loader.get_all_weights = _iter_weights  # type: ignore[assignment]
    return loader


def test_strict_check_only_validates_source_prefix_parameters():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights(["transformer.weight"])

    # Should not require VAE parameters because they are outside weights_sources.
    loader.load_weights(model)


def test_strict_check_raises_when_source_parameters_are_missing():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights([])

    with pytest.raises(ValueError, match="transformer.weight"):
        loader.load_weights(model)


def test_empty_source_prefix_keeps_full_model_strict_check():
    model = _DummyPipelineModel(source_prefix="")
    loader = _make_loader_with_weights(["transformer.weight"])

    with pytest.raises(ValueError, match="vae.weight"):
        loader.load_weights(model)


def test_qwen_model_class_selects_qwen_gguf_adapter():
    od_config = type(
        "Config",
        (),
        {
            "model_class_name": "QwenImagePipeline",
            "tf_model_config": {"model_type": "qwen_image"},
        },
    )()
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path="dummy",
        subfolder="transformer",
        revision=None,
        prefix="transformer.",
    )

    adapter = get_gguf_adapter("dummy.gguf", object(), source, od_config)

    assert adapter.__class__.__name__ == "QwenImageGGUFAdapter"


def test_online_quant_reloading_records_metadata_and_initializes_modules(monkeypatch):
    model = _OnlineQuantModel()
    recorded_models = []
    initialized_modules = []
    layer_infos = {}

    def _record_metadata(model_arg):
        recorded_models.append(model_arg)

    def _get_layerwise_info(module):
        return layer_infos.setdefault(module, type("LayerwiseInfo", (), {})())

    def _initialize_online_processing(module):
        initialized_modules.append(module)

    monkeypatch.setattr(diffusers_loader, "record_metadata_for_reloading", _record_metadata)
    monkeypatch.setattr(diffusers_loader, "get_layerwise_info", _get_layerwise_info)
    monkeypatch.setattr(diffusers_loader, "initialize_online_processing", _initialize_online_processing)

    assert diffusers_loader._prepare_online_quant_reloading(model, torch.device("cpu"))

    assert recorded_models == [model]
    assert initialized_modules == [model.linear]
    assert {info.restore_device for info in layer_infos.values()} == {torch.device("cpu")}


def test_load_model_finalizes_online_quant_and_skips_duplicate_post_load(monkeypatch):
    model = _OnlineQuantModel()
    loader = _make_loader_with_weights([])
    od_config = type(
        "Config",
        (),
        {
            "dtype": torch.float32,
            "quantization_config": object(),
            "parallel_config": type("ParallelConfig", (), {"use_hsdp": False})(),
        },
    )()
    finalized = []
    loaded_models = []
    prepared_devices = []
    target_device = torch.device("cpu:1")

    def _load_weights(model_arg):
        loaded_models.append(model_arg)

    def _prepare_online_quant_reloading(model_arg, device_arg):
        prepared_devices.append((model_arg, device_arg))
        return True

    def _finalize_layerwise_processing(model_arg, od_config_arg):
        finalized.append((model_arg, od_config_arg))

    monkeypatch.setattr(diffusers_loader, "initialize_model", lambda _od_config: model)
    monkeypatch.setattr(
        diffusers_loader, "_prepare_online_quant_reloading", _prepare_online_quant_reloading
    )
    monkeypatch.setattr(
        diffusers_loader, "finalize_layerwise_processing", _finalize_layerwise_processing
    )
    loader.load_weights = _load_weights

    loaded = loader.load_model(od_config, load_device="cpu", device=target_device)

    assert loaded is model
    assert finalized == [(model, od_config)]
    assert loaded_models == [model]
    assert prepared_devices == [(model, target_device)]
    assert model.linear.quant_method.processed_modules == []


def test_process_weights_after_loading_skips_online_quant_when_requested():
    model = _OnlineQuantModel()
    loader = _make_loader_with_weights([])

    loader._process_weights_after_loading(model, torch.device("cpu"), skip_online_quant=True)

    assert model.linear.quant_method.processed_modules == []


def test_hsdp_rejects_online_quantization(monkeypatch):
    model = _OnlineQuantModel()
    loader = _make_loader_with_weights([])
    od_config = type(
        "Config",
        (),
        {
            "dtype": torch.float32,
            "quantization_config": object(),
            "parallel_config": type(
                "ParallelConfig",
                (),
                {
                    "hsdp_replicate_size": 1,
                    "hsdp_shard_size": 1,
                    "use_hsdp": True,
                },
            )(),
        },
    )()

    monkeypatch.setattr(diffusers_loader, "initialize_model", lambda _od_config: model)

    with pytest.raises(ValueError, match="HSDP does not support online/meta-device quantization"):
        loader._load_model_with_hsdp(od_config)


def test_hsdp_rejects_unknown_load_format():
    loader = _make_loader_with_weights([])
    od_config = type(
        "Config",
        (),
        {
            "dtype": torch.float32,
            "parallel_config": type(
                "ParallelConfig",
                (),
                {
                    "hsdp_replicate_size": 1,
                    "hsdp_shard_size": 1,
                    "use_hsdp": True,
                },
            )(),
        },
    )()

    with pytest.raises(ValueError, match="Unknown load_format: diffusers"):
        loader._load_model_with_hsdp(od_config, load_format="diffusers")
