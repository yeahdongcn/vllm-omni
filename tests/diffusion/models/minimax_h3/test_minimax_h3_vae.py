# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _mock_torchaudio(monkeypatch, *, result=None, error=None):
    load = Mock(return_value=result, side_effect=error)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=load))
    return load


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
def test_seeded_device_rng_uses_active_accelerator(monkeypatch, device_type):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    device = SimpleNamespace(type=device_type)
    fork_rng = Mock(return_value=nullcontext())
    device_context = Mock(return_value=nullcontext())
    manual_seed = Mock()
    device_module = SimpleNamespace(
        device=device_context,
        manual_seed=manual_seed,
    )
    cpu_rng_state = torch.get_rng_state()
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(torch, device_type, device_module, raising=False)

    try:
        with vae_module._seeded_device_rng(device, seed=42):
            pass
    finally:
        torch.set_rng_state(cpu_rng_state)

    fork_rng.assert_called_once_with(
        devices=[device],
        device_type=device_type,
    )
    device_context.assert_called_once_with(device)
    manual_seed.assert_called_once_with(42)


@pytest.mark.parametrize("device_type", ["cpu", "xpu", "npu"])
def test_seeded_device_rng_preserves_cpu_only_fallback(monkeypatch, device_type):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    device = SimpleNamespace(type=device_type)
    fork_rng = Mock(return_value=nullcontext())
    cpu_rng_state = torch.get_rng_state()
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)

    try:
        with vae_module._seeded_device_rng(device, seed=42):
            pass
    finally:
        torch.set_rng_state(cpu_rng_state)

    fork_rng.assert_called_once_with(devices=[])


def test_seeded_device_rng_restores_cpu_state():
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    device = torch.device("cpu")
    initial_state = torch.get_rng_state()

    with vae_module._seeded_device_rng(device, seed=42):
        first = torch.randn(8)
    assert torch.equal(torch.get_rng_state(), initial_state)

    with vae_module._seeded_device_rng(device, seed=42):
        second = torch.randn(8)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), initial_state)


def test_load_audio_file_prefers_torchaudio(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video

    expected = (torch.ones(2, 8), 32_000)
    torchaudio_load = _mock_torchaudio(monkeypatch, result=expected)
    soundfile_load = Mock()
    monkeypatch.setattr(reference_video, "_soundfile_to_waveform", soundfile_load)

    assert reference_video.load_audio_file("audio.wav") is expected
    torchaudio_load.assert_called_once_with("audio.wav")
    soundfile_load.assert_not_called()


def test_load_audio_file_falls_back_to_soundfile(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video

    expected = (torch.ones(2, 8), 32_000)
    _mock_torchaudio(monkeypatch, error=RuntimeError("TorchCodec is unavailable"))
    soundfile_load = Mock(return_value=expected)
    monkeypatch.setattr(reference_video, "_soundfile_to_waveform", soundfile_load)

    assert reference_video.load_audio_file("audio.wav") is expected
    soundfile_load.assert_called_once_with("audio.wav")


def test_load_audio_file_falls_back_to_ffmpeg(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video

    expected = (torch.ones(2, 8), 32_000)
    _mock_torchaudio(monkeypatch, error=RuntimeError("TorchCodec is unavailable"))
    soundfile_load = Mock(side_effect=[OSError("unsupported input"), expected])
    ffmpeg_run = Mock()
    monkeypatch.setattr(reference_video, "_soundfile_to_waveform", soundfile_load)
    monkeypatch.setattr(reference_video.subprocess, "run", ffmpeg_run)

    assert reference_video.load_audio_file("audio.m4a") is expected
    assert soundfile_load.call_count == 2
    assert soundfile_load.call_args_list[0].args == ("audio.m4a",)
    assert soundfile_load.call_args_list[1].args[0].endswith("/audio.wav")
    command = ffmpeg_run.call_args.args[0]
    assert command[:2] == ["ffmpeg", "-y"]
    assert command[command.index("-i") + 1] == "audio.m4a"
    assert command[-1].endswith("/audio.wav")
    assert ffmpeg_run.call_args.kwargs == {"check": True}
