#!/usr/bin/env python
"""Test script for RoPE on MUSA device."""

import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding

print("RotaryEmbedding imported successfully")

# Test forward_cuda import path
rope = RotaryEmbedding(is_neox_style=True)
print("RotaryEmbedding instance created")

# Test on MUSA device
x = torch.randn(1, 10, 4, 64, device="musa:0", dtype=torch.float16)
cos = torch.randn(10, 32, device="musa:0", dtype=torch.float16)
sin = torch.randn(10, 32, device="musa:0", dtype=torch.float16)

print("Tensors created on MUSA device")

# Test forward_musa which now calls forward_cuda
try:
    result = rope.forward_musa(x, cos, sin)
    print(f"SUCCESS: forward_musa completed, output shape: {result.shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print("Test complete")
