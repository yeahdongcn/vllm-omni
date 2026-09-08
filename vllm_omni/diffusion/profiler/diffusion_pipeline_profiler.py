# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
import json
import os
import re
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def profiler(name: str, func: Callable, instance: Any) -> Callable:
    """Timing a function execution."""
    metric_name = f"{name.rsplit('.', 1)[0]}.diffuse" if name.endswith(".denoise_step") else name
    step_index = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        nonlocal step_index
        if os.environ.get("MAGI2_BENCH_GRAPH_REPLAYS", "0") == "1" and not hasattr(torch.musa, "_magi2_replay_count"):
            torch.musa._magi2_replay_count = 0
            torch.musa._magi2_replay_host_seconds = 0.0
            track_host_time = os.environ.get("MAGI2_BENCH_GRAPH_HOST_TIME", "0") == "1"
            original_replay = torch.musa.MUSAGraph.replay

            @functools.wraps(original_replay)
            def counted_replay(*a, **kw):
                torch.musa._magi2_replay_count += 1
                if track_host_time:
                    replay_start = time.perf_counter()
                    try:
                        return original_replay(*a, **kw)
                    finally:
                        torch.musa._magi2_replay_host_seconds += time.perf_counter() - replay_start
                return original_replay(*a, **kw)

            torch.musa.MUSAGraph.replay = counted_replay
        if name == f"{instance.__class__.__name__}.forward":
            instance.clear_profiler_records()
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if current_omni_platform.is_available():
                current_omni_platform.synchronize()
            duration = time.perf_counter() - start_time
            metrics_dir = os.environ.get("MAGI2_BENCH_STEP_METRICS_DIR")
            if metrics_dir and name.endswith(".denoise_step"):
                rank = dist.get_rank() if dist.is_initialized() else 0
                os.makedirs(metrics_dir, exist_ok=True)
                record = {"rank": rank, "step": step_index, "seconds": duration, "timer": metric_name}
                if os.environ.get("MAGI2_BENCH_GRAPH_REPLAYS", "0") == "1":
                    record["graph_replays_total"] = torch.musa._magi2_replay_count
                    if os.environ.get("MAGI2_BENCH_GRAPH_HOST_TIME", "0") == "1":
                        record["graph_replay_host_seconds_total"] = torch.musa._magi2_replay_host_seconds
                with open(os.path.join(metrics_dir, f"rank-{rank}.jsonl"), "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                step_index += 1
            logger.info(f"[DiffusionPipelineProfiler] {metric_name} took {duration:.6f}s")
            # record the profiling data: duration of stages
            with instance._profiler_lock:
                instance._stage_durations[metric_name] = instance._stage_durations.get(metric_name, 0.0) + duration

    return wrapper


def _parse_part(part: str) -> tuple[str, int | None]:
    """Parse 'att[num]' into ('att', num)."""
    if m := re.compile(r"(\w+)\[(\d+)\]").fullmatch(part):
        return m.group(1), int(m.group(2))
    return part, None


def _get_attribute_by_path(obj: Any, path: str) -> tuple[Any, str]:
    """Traverse an object by dotted path and return (parent_obj, attribute_name)."""
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        attr, idx = _parse_part(part)

        current = getattr(current, attr, None)
        if current is None:
            return None, parts[-1]
        if idx is not None:
            current = current[idx]

    return current, parts[-1]


def wrap_methods_by_paths(root_obj: Any, method_paths: list[str]) -> None:
    """Wrap specified methods of an object with profiler."""
    for path in method_paths:
        obj, method_name = _get_attribute_by_path(root_obj, path)
        if not obj or not hasattr(obj, method_name):
            logger.warning(f"[DiffusionPipelineProfiler] Method path {path} not found")
            continue

        original_method = getattr(obj, method_name)
        if not callable(original_method):
            logger.warning(f"[DiffusionPipelineProfiler] Attribute {path} is not callable")
            continue

        profiler_name = f"{root_obj.__class__.__name__}.{path}"
        setattr(obj, method_name, profiler(profiler_name, original_method, root_obj))


class DiffusionPipelineProfilerMixin:
    _PROFILER_TARGETS = ["vae.encode", "vae.decode", "diffuse", "text_encoder.forward", "tokenizer.forward"]

    def setup_diffusion_pipeline_profiler(
        self,
        profiler_targets: list[str] | None = None,
        enable_diffusion_pipeline_profiler: bool = False,
    ) -> None:
        self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
        if not enable_diffusion_pipeline_profiler:
            self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
            return
        self._profiler_lock = Lock()
        self._stage_durations: dict[str, float] = {}
        targets = profiler_targets if profiler_targets is not None else self._PROFILER_TARGETS
        if not targets:
            targets = []
        else:
            targets = list(targets)
        if not profiler_targets and hasattr(self, "denoise_step"):
            targets.append("denoise_step")

        targets = ["forward"] + [
            t for t in targets if t != "forward"
        ]  # ensure "forward" implement 'clear_profiler_records' at first place

        targets = list(dict.fromkeys(targets))
        wrap_methods_by_paths(
            self,
            targets,
        )

    @property
    def stage_durations(self) -> dict[str, float]:
        with self._profiler_lock:
            return self._stage_durations.copy()

    def clear_profiler_records(self) -> None:
        with self._profiler_lock:
            self._stage_durations.clear()
