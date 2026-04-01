# vLLM-Omni MUSA Backend Test Results

**Test Date:** April 1, 2026
**Platform:** MUSA (Moore Threads S5000 GPUs)
**vLLM-Omni Version:** 0.1.dev1086+gf4b3fb603.musa
**GPU:** 8x MTT S5000 (80GB each)

## Summary

All tested models work correctly on MUSA backend after applying necessary fixes for MUSA compatibility.

| Model Type | Model Name | Status | Notes |
|------------|------------|--------|-------|
| Text-to-Image | Qwen/Qwen-Image | ✅ PASSED | Generated 512x512 image successfully |
| Text-to-Video | Wan-AI/Wan2.2-T2V-A14B-Diffusers | ✅ PASSED | Generated 5 frames at 480x640 |
| Omni (Multi-modal) | Qwen/Qwen2.5-Omni-3B | ✅ PASSED | All 3 stages initialized, text generation working |

## Required Environment Variables

```bash
export TORCHDYNAMO_DISABLE=1
export VLLM_ATTENTION_BACKEND=SDPA
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Model Test Details

### 1. Qwen-Image (Text-to-Image Diffusion)

**Command:**
```bash
python -c "
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model='/path/to/Qwen-Image',
    enforce_eager=True
)

outputs = omni.generate(
    'a photo of a cat sitting on a laptop keyboard',
    OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=4.0,
        seed=42,
    ),
)

images = outputs[0].request_output.images
images[0].save('output.png')
omni.close()
"
```

**Results:**
- Model loading: 53.74 GiB
- Peak GPU memory: 58.59 GB
- Generation time: ~2 seconds for 4 steps
- Output: 512x512 PNG image

### 2. Wan2.2-T2V (Text-to-Video Diffusion)

**Command:**
```bash
python -c "
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model='/path/to/Wan2.2-T2V-A14B-Diffusers',
    enforce_eager=True,
    boundary_ratio=0.875,
    flow_shift=5.0,
)

outputs = omni.generate(
    prompts='A cat sitting on a table',
    sampling_params_list=OmniDiffusionSamplingParams(
        height=480,
        width=640,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
    ),
)

frames = outputs[0].request_output.images[0]
# frames shape: (batch, num_frames, height, width, channels)
omni.close()
"
```

**Results:**
- Model loading: 64.44 GiB
- Peak GPU memory: 74.59 GB
- Generation time: ~1.3 seconds for 2 steps
- Output: 5 frames at 480x640

### 3. Qwen2.5-Omni-3B (Multi-modal Omni)

**Command:**
```bash
python -c "
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from vllm_omni import Omni

omni = Omni(model='/path/to/Qwen2.5-Omni-3B')

prompt = {
    'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of China?<|im_end|>\n<|im_start|>assistant\n',
}

outputs = omni.generate([prompt])
text_output = outputs[0].request_output.outputs[0].text
print(text_output)
omni.close()
"
```

**Results:**
- 3 stages initialized:
  - Stage 0 (Thinker): 8.84 GiB
  - Stage 1 (Talker): 8.84 GiB
  - Stage 2 (Code2Wav): 1.46 GiB
- Total initialization: ~82 seconds
- Output: Text response "The capital of China is Beijing."

## MUSA Compatibility Fixes Applied

### 1. RotaryEmbedding forward_musa (rope.py)

MUSA doesn't have flash_attn, so `forward_musa` must use native implementation instead of falling back to `forward_cuda`.

```python
def forward_musa(self, x, cos, sin):
    # MUSA doesn't have flash_attn, use native implementation
    return self.forward_native(x, cos, sin)
```

### 2. WanRotaryPosEmbed float64 Support (wan2_2_transformer.py)

MUSA doesn't support float64 (Double) operations. Use float32 for `freqs_dtype` when running on MUSA.

```python
from vllm_omni.platforms import current_omni_platform

if torch.backends.mps.is_available() or current_omni_platform.is_musa():
    freqs_dtype = torch.float32
else:
    freqs_dtype = torch.float64
```

## Known Limitations

1. **Flash Attention**: MUSA doesn't support Flash Attention. Must use SDPA backend (`VLLM_ATTENTION_BACKEND=SDPA`).

2. **Float64**: MUSA doesn't support float64 operations. Some models may need patches for float32 fallback.

3. **NVML**: NVML init fails on MUSA. The system falls back to profiling-based memory estimation.

4. **torch.compile**: Must be disabled via `TORCHDYNAMO_DISABLE=1` or `enforce_eager=True`.

## Online Serving

To start an OpenAI-compatible API server:

```bash
vllm-omni serve /path/to/model --omni --port 8091
```

For diffusion models:
```bash
vllm-omni serve /path/to/Qwen-Image --omni --port 8091
```

## Testing Commands Quick Reference

```bash
# Text-to-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model /home/dist/diffusion/yeahdongcn/Qwen/Qwen-Image \
    --prompt "a photo of a cat" \
    --height 512 --width 512 \
    --num-inference-steps 4

# Text-to-Video
python examples/offline_inference/text_to_video/text_to_video.py \
    --model /home/dist/diffusion/yeahdongcn/Wan-AI/Wan2.2-T2V-A14B-Diffusers

# Omni (Qwen2.5-Omni)
python examples/offline_inference/qwen2_5_omni/end2end.py \
    --model /home/dist/diffusion/yeahdongcn/Qwen/Qwen2.5-Omni-3B \
    --query-type text
```

## Available Models in Test Environment

| Path | Model Type |
|------|------------|
| `/home/dist/diffusion/yeahdongcn/Qwen/Qwen-Image` | Text-to-Image |
| `/home/dist/diffusion/yeahdongcn/Qwen/Qwen2.5-Omni-3B` | Omni (3B) |
| `/home/dist/diffusion/yeahdongcn/Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Text-to-Video |
| `/home/dist/diffusion/yeahdongcn/Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-Video |
