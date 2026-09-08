# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cold-cache synthetic EP4 MoE scheduling sweep, not model E2E evidence."""
import argparse
import json
import os
import statistics

import torchada  # noqa: F401
import torch
from mate.testing.utils import bench_gpu_time_with_musa_event
from vllm_omni.diffusion.models.magi2.mh_moe import _magi2_sgl_fused_moe_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=15004)
    parser.add_argument("--repeat-iters", type=int, default=10)
    args = parser.parse_args()
    torch.musa.set_device(0)
    torch.manual_seed(42)
    h, e, k, n, topk = 3, 768, 256, 1280, 6
    w13 = (torch.randn(e, 2*n, k, device="musa")*.05).bfloat16()
    w2 = (torch.randn(e, k, n, device="musa")*.05).bfloat16()
    x = torch.randn(args.tokens, h, k, device="musa").bfloat16()
    scores = torch.randn(h, args.tokens, 256, device="musa")
    ids = scores.topk(topk, dim=-1).indices.int().contiguous()
    probs = torch.softmax(torch.randn(h, args.tokens, topk, device="musa"), -1)
    del scores
    print(json.dumps({"tokens": args.tokens, "E": e, "K": k, "I": n,
                      "slots": h*args.tokens*topk, "torch": torch.__version__,
                      "device": str(torch.musa.get_device_properties(0))}), flush=True)

    def run(pair):
        os.environ["MAGI2_SGL_GROUP_M"] = str(pair[0])
        os.environ["MAGI2_SGL_DOWN_GROUP_M"] = str(pair[1])
        return _magi2_sgl_fused_moe_forward(
            x, probs, ids, w13[:, 0::2, :].transpose(1, 2),
            w13[:, 1::2, :].transpose(1, 2), w2.transpose(1, 2), w13, w2)

    pairs = [(16,16), (1,1), (2,2), (4,4), (8,8), (32,32), (1,16), (16,1)]
    with torch.inference_mode():
        ref = run((16,16))
        for pair in pairs:
            got = run(pair)
            torch.musa.synchronize()
            if not torch.equal(ref, got):
                raise RuntimeError(f"Nonidentical output for group-M {pair}")
        del got, ref
        for repeat in range(3):
            order = pairs if repeat % 2 == 0 else list(reversed(pairs))
            for pair in order:
                vals = bench_gpu_time_with_musa_event(
                    lambda: run(pair), dry_run_iters=3, repeat_iters=args.repeat_iters,
                    l2_flush=True, l2_flush_size_mb=1024)
                print(json.dumps({"repeat": repeat, "group_m": pair,
                                  "p50_ms": statistics.median(vals),
                                  "mean_ms": statistics.mean(vals), "samples_ms": vals,
                                  "exact": True}), flush=True)
    print(json.dumps({"completed": True}), flush=True)


if __name__ == "__main__":
    main()
