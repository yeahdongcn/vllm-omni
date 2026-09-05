# MiniMax H3 on Moore Threads MUSA

> Joint video and audio generation on Moore Threads GPUs

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: 4x Moore Threads MTT S5000 for the validated Ref2VA profile; six
  GPUs on one eight-GPU S5000 node for the disaggregated T2VA/FastH3 profile
- Maintainer: Community

This recipe adapts [MiniMax-H3.md](MiniMax-H3.md) for MUSA environments.
The modular H3 pipeline can load both task-specific DiTs, but this MUSA recipe
uses a **single task partition**: `FL2VA` for `t2va`/`fl2va`, or `Ref2VA` for
`ref2va`. This avoids loading both DiTs.

## Prerequisites

### Checkpoint

Download the partition required by the selected startup task. The commands below
run a single-task server; do not use the modular default (combined) mode on MUSA
until its two-DiT memory and performance profile is validated.

For T2VA and FL2VA:

```bash
python -m pip install modelscope
export MODEL_ROOT=/path/to/MiniMax-H3
modelscope download MiniMax/MiniMax-H3 \
  --local_dir "${MODEL_ROOT}" \
  --max-workers 16 \
  --include 'FL2VA/**'
```

For Ref2VA, download `Ref2VA/**` instead.

### Environment

Install a compatible PyTorch/torch-musa, torchada, MATE, and vLLM-MUSA
stack before installing vLLM-Omni. Importing vLLM-Omni should report that the
`musa` vLLM platform plugin is active.

Install vLLM-Omni from a checkout containing MiniMax H3 support:

```bash
python -m pip install -e .
```

MiniMax H3 reference inputs and MP4 output require `soundfile`, `ffmpeg`, and
`ffprobe`. The Python dependency is installed by vLLM-Omni; install the two
executables with the operating system package manager and verify them:

```bash
python -c 'import soundfile; print(soundfile.__version__)'
ffmpeg -version
ffprobe -version
```

When torchaudio cannot load TorchCodec, vLLM-Omni automatically falls back to
soundfile. Formats that libsndfile cannot read are demuxed through ffmpeg.

## Start a server

### Current main: disaggregated T2VA/FL2VA

Current `main` resolves the modular MiniMax H3 model through an explicit deploy
configuration. On an eight-GPU S5000 node, use the MUSA configuration rather
than the CUDA-oriented TP1/Ulysses4 configuration:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export PORT=8092

MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 \
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=3600 \
vllm serve "${MODEL_ROOT}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_musa.yaml \
  --task-type fl2va \
  --init-timeout 1800
```

The MUSA async video-output path uses a backend-neutral `torch.Event` for the
device-to-host copy. This is required on MUSA: a native `torch.musa.Event` is
not observed by the generic output-copy stream and can produce valid MP4 files
whose pixels are mosaic/static on a subsequent request. Use a checkout that
contains the MUSA event fix; do not work around it by changing BF16 to FP32.

The text encoder uses TP2 on devices 0-1. The diffusion stage uses TP4 and
four-way Video VAE patch parallelism on devices 2-5. Devices 6-7 remain
available for runtime headroom. Replicating the full DiT with TP1/Ulysses4 can
exhaust S5000 memory when the Video VAE is materialized for 1344x768 decode.

### FastH3 four-step adapter

Download the dense adapter; it does not require the CUDA-only FastVideo VSA
extension:

```bash
export FASTH3_DIR=/path/to/FastVideo-FastH3-4-step-Preview-v1-LoRA
huggingface-cli download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --local-dir "${FASTH3_DIR}"

export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

Add the adapter to the disaggregated command above:

```bash
  --task-type fl2va \
  --lora-path "${FASTH3_LORA}"
```

FastH3 is T2VA-only and must be requested with
`extra_params={"task":"t2va","duration":...}` and
`num_inference_steps=4`. The adapter is fused into the DiT weights at startup;
CPU/offloaded DiT serving is not supported for this path.

`FASTVIDEO_VSA` is a separate optimization. The stock `fastvideo-kernel`
package is built for CUDA/CUTLASS Hopper and Blackwell kernels and is not a
MUSA dependency. Do not select `FASTVIDEO_VSA` unless a MUSA implementation is
installed and validated; the dense FastH3 adapter works with the MUSA
`FLASH_ATTN` backend.

An S5000 validation on 2026-09-05 used vLLM 0.28.0, vLLM-MUSA 0.1.28,
vLLM-Omni main at `0f9ee6af` plus the MUSA event fix, and torch
2.11.0/MUSA 5.2. For a 1344x768, 24 fps, 15-second T2VA request, regular H3 at
50 steps took 1152.22 seconds, while three measured dense FastH3 four-step
requests had a median client completion time of 110.89 seconds (10.39x
faster). The corrected service was then exercised with a 4-second warmup,
two consecutive 15-second FastH3 requests (seeds 1501 and 1502), and repeated
same-shape requests; all returned visually coherent frames, including the
second request. Before the event fix, the second identical request produced a
structurally valid but mosaic/static MP4, so MP4 metadata alone is not a
correctness gate. These figures describe this exact software and hardware
profile, not a general latency guarantee.

### Partition server (v0.28 release profile)

Use the partition path and set `--task-type` explicitly: `fl2va` for T2VA/FL2VA
or `ref2va` for Ref2VA.

The validated Ref2VA configuration uses tensor parallelism across four MTT S5000 GPUs and offloads inactive model components to CPU:

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
export PORT=8091

MUSA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --task-type ref2va \
  --tensor-parallel-size 4 \
  --vae-patch-parallel-size 4 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

On MUSA, `FLASH_ATTN` selects the MATE/FlashAttention-3 implementation. Do
not install the CUDA-only `fa4` optional dependency.

For request examples and parameter definitions, see
[MiniMax-H3.md](MiniMax-H3.md#http-api-examples).

## MUSA validation

The following weight-independent gates have been validated on one MTT S5000:

- packed variable-length attention selects MATE FlashAttention-3 and returns
  finite BF16 output;
- MiniMax H3 RoPE casts CPU FP64 position metadata to FP32 before device
  arithmetic and returns finite output;
- the video-VAE seeded RNG context is deterministic on MUSA and restores both
  CPU and MUSA RNG states;
- the soundfile fallback preserves stereo channel layout, FP32 samples, and
  the native sample rate when torchaudio/TorchCodec is unavailable.

Ref2VA image+audio generation was also validated end to end on one MTT S5000
with CPU offload and VAE tiling. The smoke used a 448x256 reference image, a
four-second 32 kHz stereo audio reference, two denoising steps, and seed 42.
The synchronous HTTP request completed in 50.6 seconds and returned an MP4
with the following decoded properties:

- 107 video frames at 448x256 and 24 FPS;
- non-static video (`pixel_std=13.07`, `temporal_delta=0.49`);
- finite stereo audio at 32 kHz (`audio_rms=0.0458`).

Official Ref2VA video-reference serving was validated with the expanded
MiniMax prompt at 1344x768, 5 seconds, seed 0, and 50 steps. A true TP4 run on
four MTT S5000 GPUs returned HTTP 200 in 638.6 seconds and produced 124 H.264
frames at 24 FPS with AAC stereo audio at 32 kHz.

Keep `--vae-use-tiling` enabled for this serving profile.

## Known limitations

- Keep Ring Attention at degree 1. The current Ring path does not preserve
  MiniMax H3 packed padding boundaries.
- H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.
- This MUSA recipe uses an explicitly selected single task partition; combined
  serving loads both DiTs.
- H3 currently executes one generation request per diffusion batch.
- FP8 quantization has not been enabled for MiniMax H3.
- MP3, M4A, MP4, and reference-video audio fallback requires `ffmpeg` on
  `PATH`; WAV inputs can be read directly through soundfile.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full model and API guide
- [MiniMax-H3-NPU.md](MiniMax-H3-NPU.md) — Ascend NPU deployment guide
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
