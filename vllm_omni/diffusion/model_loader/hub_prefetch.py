# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Best-effort HuggingFace Hub prefetch for multi-subfolder pipelines.

This module exists to defend diffusion pipelines against a race condition we
hit after the `transformers` v5 rebase (see Buildkite vllm-omni-rebase
#1043 Qwen-Image-Edit-2509 failure):

When several diffusion worker processes start in parallel and each calls
``SomeModel.from_pretrained(model_id, subfolder="text_encoder", ...)`` with a
cold HuggingFace cache, transformers v5's cache-resolution (``cached_files``)
can observe a partially-written shard set written by a peer worker and raise
``OSError: <model_id> does not appear to have a file named
text_encoder/model-00002-of-00002.safetensors`` even though the peer will
eventually finish writing it.

Why ``origin/main`` does not need this helper
---------------------------------------------
The exact same ``__init__`` code lives on ``origin/main`` (e.g. the Qwen-Image
``pipeline_qwen_image_edit_plus.py`` in build vllm-omni#7412 passes without
any prefetch), so the race is NOT a behavioural change in vLLM-Omni itself.
Two environmental factors mask the race on main:

* ``origin/main`` is pinned (transitively, via vLLM main) to
  ``transformers`` 4.x. In 4.x the per-file ``cached_file`` path resolves
  shards **lazily**, one at a time, so each ``hf_hub_download`` blocks on its
  own single-file ``.lock`` and the second worker naturally waits for the
  first worker's atomic rename. ``transformers>=5.0`` rewrote this into
  ``cached_files`` (plural) which batch-resolves every shard listed in the
  index up-front via ``os.path.isfile`` and raises immediately if any shard
  is still sitting under its ``*.incomplete`` name. Same wave of v5 changes
  that introduced ``tie_weights(missing_keys=..., recompute_mapping=...)``
  (see the Dynin shim in ``dynin_omni_token2text.py``).
* CI shares ``HF_HOME=/fsx/hf_cache`` across pipelines (both the
  ``vllm-omni`` and ``vllm-omni-rebase`` pipelines mount the same FS). That
  cache is normally warm for long-lived repos like ``Qwen-Image-Edit-2509``,
  so most builds never go through the download path at all. Build 1043
  happened to hit a partially-evicted cache AND transformers v5's stricter
  resolver simultaneously, which is why the failure looks 'rebase-specific'
  but is really a latent race that main was absorbing via (1).

``huggingface_hub.snapshot_download`` guards its downloads with a per-repo
``.lock`` file, so invoking it up-front from every worker is safe: the first
worker into the critical section actually downloads, and subsequent workers
block until the shards are atomically renamed into place, after which their
own ``from_pretrained`` calls see a fully populated cache and succeed. For a
warm cache the snapshot call is a near-noop (it only stat()s the files), so
this is also cheap on ``origin/main`` should we ever backport it there.

The helper is intentionally best-effort: prefetch failures (offline, gated
repos, transient 5xx) are logged and swallowed so the subsequent
``from_pretrained`` call can surface the real, specific error to the user
rather than being masked here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def prefetch_subfolders(
    model: str,
    subfolders: Iterable[str],
    *,
    local_files_only: bool = False,
    include_root_metadata: bool = True,
) -> None:
    """Materialise ``model``'s ``subfolders`` in the HF cache before loading.

    Args:
        model: A HuggingFace Hub repo id (e.g. ``"Qwen/Qwen-Image-Edit-2509"``)
            or a local directory path. Local paths are a no-op.
        subfolders: Iterable of subfolder names (e.g. ``["text_encoder",
            "vae"]``) whose contents need to be fully present before any
            worker calls ``from_pretrained(subfolder=...)``.
        local_files_only: When True, skip the prefetch entirely. The caller
            has explicitly promised the cache is already populated (as happens
            for local model checkouts), so hitting the network would defeat
            the intent and may fail in air-gapped environments.
        include_root_metadata: When True, also pull ``*.json`` at the repo
            root so ``model_index.json`` / ``config.json`` resolution during
            ``from_pretrained`` also hits a warm cache.
    """
    if local_files_only or not model or os.path.isdir(model):
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        logger.debug("huggingface_hub unavailable; skipping prefetch of %s", model)
        return

    allow_patterns: list[str] = []
    for sub in subfolders:
        sub = (sub or "").strip("/")
        if not sub:
            continue
        # hf_hub globbing is shell-style: `text_encoder/*` catches the index +
        # any flat files, `text_encoder/**` catches nested safetensors shards
        # that some repos place under e.g. ``text_encoder/pytorch_model/``.
        allow_patterns.extend([f"{sub}/*", f"{sub}/**"])

    if include_root_metadata:
        allow_patterns.extend(["*.json", "*.txt"])

    if not allow_patterns:
        return

    try:
        snapshot_download(
            repo_id=model,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        # Best-effort: propagate only via logging. The subsequent
        # ``from_pretrained`` call will raise a clearer, call-site-specific
        # error (auth, 404, disk full, ...) that we'd rather surface - EXCEPT
        # for auth/gating, which we escalate here with an explicit hint so
        # readers of CI logs don't have to correlate the generic "OSError:
        # <repo> does not appear to have a file named ..." that
        # ``from_pretrained`` would otherwise emit much later with an
        # unrelated-looking message.
        if _looks_like_auth_error(exc):
            logger.error(
                "Hub prefetch for '%s' failed with an authentication / gated "
                "repository error (%s: %s). The CI HF_TOKEN must (1) be set "
                "in the step env, (2) be valid, and (3) belong to an account "
                "that has accepted the model license on huggingface.co. See "
                "docs/contributing/ci/hf_credentials.md.",
                model,
                type(exc).__name__,
                exc,
            )
        else:
            logger.warning(
                "Hub prefetch for repo '%s' subfolders %s failed (%s: %s); "
                "falling back to on-demand download in from_pretrained",
                model,
                list(subfolders),
                type(exc).__name__,
                exc,
            )


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Classify prefetch exceptions as auth/gating failures.

    ``huggingface_hub`` raises bespoke ``GatedRepoError`` /
    ``RepositoryNotFoundError`` subclasses when the token is missing or
    lacks license acceptance; but older releases (and third-party
    transport layers) sometimes only surface this as a generic
    ``HfHubHTTPError`` / ``requests.HTTPError`` with status 401/403. We
    check both code paths so the branch above is stable across
    ``huggingface_hub`` versions.
    """
    try:
        from huggingface_hub.errors import (  # type: ignore[import-not-found]
            GatedRepoError,
            RepositoryNotFoundError,
        )

        if isinstance(exc, GatedRepoError | RepositoryNotFoundError):
            return True
    except ImportError:  # pragma: no cover - very old huggingface_hub
        pass

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    # Last-resort string heuristic - ``snapshot_download`` on some
    # transports wraps the 401 as a plain OSError whose message is the
    # only load-bearing signal.
    msg = str(exc).lower()
    return "401 client error" in msg or "403 client error" in msg or "gatedrepo" in msg
