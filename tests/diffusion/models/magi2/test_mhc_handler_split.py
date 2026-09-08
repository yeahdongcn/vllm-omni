# SPDX-License-Identifier: Apache-2.0
"""Structural regression for branch-specific compiled MHC handlers."""
from pathlib import Path


def test_model_owns_distinct_mhc_handlers():
    source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/magi2/modeling_magi2.py"
    text = source.read_text()
    assert "self.mhc_handler_attn = MHCHandler" in text
    assert "self.mhc_handler_mlp = MHCHandler" in text
    assert "def _mhc_handler(self, branch: str)" in text
