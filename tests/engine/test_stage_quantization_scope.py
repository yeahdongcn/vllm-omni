import copy
from dataclasses import dataclass

from vllm_omni.engine.arg_utils import _scope_prequantized_model_config


@dataclass
class _TextConfig:
    quantization_config: object | None = None


@dataclass
class _StageHFConfig:
    text_config: _TextConfig | None = None
    quantization_config: object | None = None


@dataclass
class _HFConfig:
    thinker_config: _StageHFConfig
    talker_config: _StageHFConfig
    code2wav_config: _StageHFConfig
    quantization_config: object | None = None


@dataclass
class _ModelArchConfig:
    quantization_config: object | None


@dataclass
class _ModelConfig:
    hf_config: _HFConfig
    quantization: str | None
    model_arch_config: _ModelArchConfig

    def get_model_arch_config(self) -> _ModelArchConfig:
        return _ModelArchConfig(
            quantization_config=self.hf_config.quantization_config,
        )


def _config() -> _ModelConfig:
    quantization_config = {"quant_method": "modelopt"}
    hf_config = _HFConfig(
        thinker_config=_StageHFConfig(
            text_config=_TextConfig(
                quantization_config=quantization_config,
            )
        ),
        talker_config=_StageHFConfig(text_config=_TextConfig()),
        code2wav_config=_StageHFConfig(),
        quantization_config=quantization_config,
    )
    return _ModelConfig(
        hf_config=hf_config,
        quantization="modelopt",
        model_arch_config=_ModelArchConfig(
            quantization_config=quantization_config,
        ),
    )


def _scope(
    config: _ModelConfig,
    *,
    model_stage: str,
    hf_config_name: str | None = None,
    explicit_quantization: str | None = None,
) -> _ModelConfig:
    return _scope_prequantized_model_config(
        config,
        model_stage=model_stage,
        hf_config_name=hf_config_name,
        explicit_quantization=explicit_quantization,
    )


def test_auto_modelopt_is_removed_during_talker_model_config_creation() -> None:
    config = _config()
    original = copy.deepcopy(config)

    scoped = _scope(
        config,
        model_stage="talker",
        hf_config_name="talker_config",
    )

    assert scoped is not config
    assert scoped.quantization is None
    assert scoped.hf_config.quantization_config is None
    assert scoped.model_arch_config.quantization_config is None
    assert config == original


def test_stage_local_modelopt_is_preserved_for_thinker() -> None:
    config = _config()

    assert (
        _scope(
            config,
            model_stage="thinker",
            hf_config_name="thinker_config",
        )
        is config
    )


def test_unknown_stage_preserves_top_level_modelopt() -> None:
    config = _config()

    assert _scope(config, model_stage="main") is config


def test_code2wav_uses_model_stage_before_hf_config_name() -> None:
    config = _config()

    scoped = _scope(
        config,
        model_stage="code2wav",
        hf_config_name="thinker_config",
    )

    assert scoped.quantization is None
    assert scoped.hf_config.quantization_config is None


def test_explicit_quantization_is_not_overridden() -> None:
    config = _config()

    assert (
        _scope(
            config,
            model_stage="talker",
            explicit_quantization="modelopt",
        )
        is config
    )
