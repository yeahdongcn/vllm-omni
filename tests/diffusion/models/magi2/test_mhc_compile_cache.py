# SPDX-License-Identifier: Apache-2.0
"""Regression for compiled BF16 mHC parameter aliasing/reload."""
import ast
import math
from pathlib import Path

import torch


def _handler():
    source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/magi2/layers.py"
    parsed = ast.parse(source.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "MHCHandler")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["MHCHandler"](4, 16)


def test_compiled_phi_uses_current_argument_after_reload():
    torch._dynamo.reset()
    handler = _handler()
    first = torch.ones(64, 24)
    second = torch.full_like(first, 2.0)
    compiled_first = torch.compile(lambda: handler._bf16_phi(first), backend="eager", fullgraph=True)
    compiled_second = torch.compile(lambda: handler._bf16_phi(second), backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled_first(), first.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(compiled_second(), second.bfloat16(), rtol=0, atol=0)
    second.copy_(3.0)
    torch.testing.assert_close(compiled_second(), second.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(compiled_first(), first.bfloat16(), rtol=0, atol=0)
    torch._dynamo.reset()
