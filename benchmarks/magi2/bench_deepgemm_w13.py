# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cold-cache A/B of the complete production MoE wrapper, W13-only difference.

The production shape comes from magi2-omni-trace-r0.json.gz: 144 aten::bmm
events have inputs [3,14601,256] @ [3,256,256]. Routes are synthetic but
head-local (never whole-token/global-expert routing); both variants share them.
This is an eager operator diagnostic, not an E2E <=700 ms qualification.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torchada  # noqa: F401  # isort: skip

import torch
from mate.testing.utils import bench_gpu_time

from vllm_omni.diffusion.models.magi2.mh_moe import (
    _magi2_align_block_size,
    _magi2_sgl_fused_moe_forward,
)

logger = logging.getLogger(__name__)


def command_output(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(args.seed)
    h, t, e, d, n = args.heads, args.tokens, args.experts_per_head, args.head_dim, args.intermediate
    x = torch.randn(t, h, d, device="musa", dtype=torch.bfloat16)
    ids = torch.rand(h, t, e, device="musa").topk(args.top_k, dim=-1).indices.to(torch.int32)
    probabilities = torch.rand(h, t, args.top_k, device="musa").softmax(-1)
    packed_w13 = (torch.randn(h * e, 2 * n, d, device="musa") / math.sqrt(d)).bfloat16()
    packed_w2 = (torch.randn(h * e, d, n, device="musa") / math.sqrt(n)).bfloat16()
    # Both arms use exactly the production owned layouts; no per-call packing.
    gate = packed_w13[:, 0::2, :].transpose(1, 2)
    up = packed_w13[:, 1::2, :].transpose(1, 2)
    down = packed_w2.transpose(1, 2)
    return x, probabilities, ids, gate, up, down, packed_w13, packed_w2


def correctness(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    a, b = reference.float(), candidate.float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    nonconstant = bool((a.std() > 0) & (b.std() > 0))
    error = (a - b).abs()
    metrics = {
        "finite": finite,
        "nonconstant": nonconstant,
        "max_abs": error.max().item(),
        "relative_l2": (error.norm() / a.norm().clamp_min(1e-12)).item(),
        "max_relative": (error / a.abs().clamp_min(1e-3)).max().item(),
        "rtol": 0.01,
        "atol": 0.01,
    }
    if not finite or not nonconstant:
        raise AssertionError(f"Non-finite or poison output: {metrics}")
    torch.testing.assert_close(candidate, reference, rtol=0.01, atol=0.01)
    if metrics["relative_l2"] > 0.01:
        raise AssertionError(f"Relative L2 exceeds 1%: {metrics}")
    return metrics


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(samples) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(samples: list[float], flops: int, io_bytes: int) -> dict[str, object]:
    median_ms = statistics.median(samples)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "p50_ms": median_ms,
        "p90_ms": percentile(samples, 0.9),
        "max_ms": max(samples),
        "logical_tflops": flops / median_ms / 1e9,
        "minimum_io_gbps": io_bytes / median_ms / 1e6,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("This benchmark requires an authorized MUSA container")
    if args.top_k != 6 or args.experts_per_head < args.top_k:
        raise ValueError("The production MAGI2 MoE wrapper requires top_k=6")
    os.environ["MAGI2_SGL_SORTED_INTERMEDIATE"] = "1"
    os.environ["MAGI2_DEEPGEMM_W13_BACKEND"] = args.backend
    inputs = make_inputs(args)
    ids = inputs[2]
    flattened_ids = (
        ids + torch.arange(args.heads, device=ids.device, dtype=torch.int32)[:, None, None] * args.experts_per_head
    ).reshape(-1, args.top_k)
    sorted_ids, expert_ids, num_padded = _magi2_align_block_size(flattened_ids, args.heads * args.experts_per_head, 128)

    def invoke(candidate: bool) -> torch.Tensor:
        return _magi2_sgl_fused_moe_forward(*inputs, use_deepgemm_w13=candidate)

    reference, candidate = invoke(False), invoke(True)
    parity = correctness(reference, candidate)
    torch.musa.synchronize()
    order: list[str] = []

    def alternate() -> None:
        label = "baseline" if len(order) % 2 == 0 else "deepgemm_w13"
        order.append(label)
        invoke(label == "deepgemm_w13")

    # Reuse MATE events and cache flush, including sync outside the timed region.
    # Tag every invocation: helper warmup count need not be assumed by this bench.
    samples = bench_gpu_time(
        alternate,
        dry_run_iters=2,
        repeat_iters=2 * args.trials,
        l2_flush=True,
        l2_flush_size_mb=8192,
        use_musa_graph=False,
    )
    torch.musa.synchronize()
    samples = [float(value) for value in samples]
    measured_order = order[-len(samples) :]
    by_variant = {
        label: [sample for sample, variant in zip(samples, measured_order) if variant == label]
        for label in ("baseline", "deepgemm_w13")
    }
    if any(len(values) != args.trials for values in by_variant.values()):
        raise AssertionError("Benchmark helper did not preserve the interleaved A/B sample count")
    # Re-check after repeated reuse to catch stale-buffer/allocator-dependent errors.
    parity_after = correctness(invoke(False), invoke(True))
    m, routed, experts = args.heads * args.tokens, flattened_ids.numel(), args.heads * args.experts_per_head
    flops = 6 * routed * args.head_dim * args.intermediate
    minimum_bytes = sum(t.numel() * t.element_size() for t in (inputs[0], inputs[1], ids, inputs[6], inputs[7]))
    minimum_bytes += reference.numel() * reference.element_size()
    versions = {}
    for package in ("torch", "torch_musa", "torchada", "mate", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    return {
        "shape": {
            "tokens": args.tokens,
            "local_heads": args.heads,
            "experts_per_head": args.experts_per_head,
            "E": experts,
            "K": args.head_dim,
            "I": args.intermediate,
            "top_k": args.top_k,
            "input_rows": m,
            "routed_rows": routed,
            "padded_rows": num_padded.item(),
            "capacity_rows": sorted_ids.numel(),
            "expert_blocks": expert_ids.numel(),
            "W13": list(inputs[6].shape),
            "W2": list(inputs[7].shape),
            "dtype": "bf16",
        },
        "provenance": {
            "shape_source": "magi2-omni-trace-r0.json.gz: aten::bmm [3,14601,256]",
            "synthetic_routes": True,
            "seed": args.seed,
            "image": args.image,
            "baseline_sha": args.baseline_sha,
            "candidate_sha": args.candidate_sha,
            "checkout_sha": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--porcelain"]),
            "command": sys.argv,
            "versions": versions,
            "device": str(torch.musa.get_device_properties(0)),
            "mthreads_gmi": command_output(["mthreads-gmi"]),
            "route_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "env": {key: value for key, value in os.environ.items() if key.startswith("MAGI2_")},
        },
        "measurement": {
            "boundary": "complete MoE: layout, route sort, W13, activation, unchanged W2 and reduction",
            "harness": "mate.testing.utils.bench_gpu_time",
            "flush_l2": True,
            "flush_l2_bytes": 8192 * 1024 * 1024,
            "compile": False,
            "graph": False,
            "order": measured_order,
            "logical_flops": flops,
            "minimum_io_bytes": minimum_bytes,
            "io_note": "minimum tensor IO only; excludes intermediate traffic and padding work",
        },
        "correctness_before": parity,
        "correctness_after": parity_after,
        "timing": {label: timing_summary(values, flops, minimum_bytes) for label, values in by_variant.items()},
        "backend": args.backend,
        "result": "microbench_only_not_e2e_qualified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=14601)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--experts-per-head", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("mubin", "mutlass"), default="mubin")
    parser.add_argument("--image", required=True, help="Digest-qualified image provenance")
    parser.add_argument("--baseline-sha", required=True, help="Frozen production source SHA")
    parser.add_argument("--candidate-sha", required=True, help="Exact commit exported into the test source archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.experts_per_head, args.head_dim, args.intermediate, args.trials) < 1:
        parser.error("All dimensions and trials must be positive")
    logging.basicConfig(level=logging.INFO)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("MoE cold-cache A/B: %s", json.dumps(result["timing"]))


if __name__ == "__main__":
    main()
