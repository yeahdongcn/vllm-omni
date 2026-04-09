#!/usr/bin/env python
"""Test script for RoPE on MUSA device.

Direct import bypassing vllm_omni.__init__ to avoid vllm version mismatch issues.
"""

import importlib.util
import sys

import torch
import torch_musa  # noqa: F401  Required for MUSA device support

print("torch and torch_musa loaded")

# Direct module loading bypassing __init__.py chain
spec = importlib.util.spec_from_file_location("rope_module", "/root/vllm-omni/vllm_omni/diffusion/layers/rope.py")
rope_module = importlib.util.module_from_spec(spec)

# Need to preload dependencies
custom_op_spec = importlib.util.spec_from_file_location(
    "custom_op", "/root/vllm-omni/vllm_omni/diffusion/layers/custom_op.py"
)
custom_op_module = importlib.util.module_from_spec(custom_op_spec)
sys.modules["vllm_omni.diffusion.layers.custom_op"] = custom_op_module
custom_op_spec.loader.exec_module(custom_op_module)

# Now load rope module
spec.loader.exec_module(rope_module)

print("rope module loaded directly")

# Test RotaryEmbedding
RotaryEmbedding = rope_module.RotaryEmbedding
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
