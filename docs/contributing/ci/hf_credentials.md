# HuggingFace credentials in CI

Several of our diffusion and audio pipelines load models from **gated**
HuggingFace repositories. When the CI job's `HF_TOKEN` is missing, invalid, or
belongs to an account that has not accepted the repo's license, the job fails
deep inside `from_pretrained` with a generic error like:

```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo for url https://huggingface.co/api/models/<repo>
```

or, worse, with the post-download `OSError: <repo> does not appear to have
a file named <subfolder>/model-00002-of-00002.safetensors` that `transformers`
v5's `cached_files` emits once the partial download aborts half-way through.

This page documents what needs to be true for those jobs to pass.

## 1. Required secret

The Buildkite pipeline and the Kubernetes pod specs expect a secret named
`HF_TOKEN` (for docker-based steps, a propagated env var; for k8s-based
steps, a `secretKeyRef` to the `hf-token-secret` / `token` entry). This
token must be a **read** token issued by a huggingface.co account that has:

1. Accepted the license for every gated repo we load, see the list below.
2. Not had its email verification lapse (HF periodically invalidates tokens
   belonging to unverified accounts).

When updating the token, update it in **both** places:

| CI surface | Where the token is referenced |
| --- | --- |
| `.buildkite/test-ready.yml` / `test-merge.yml` / `test-nightly.yml` (docker plugin steps) | `environment: [HF_TOKEN]` (propagated from the Buildkite agent's env) |
| k8s pod specs in the same files | `env.HF_TOKEN.valueFrom.secretKeyRef` -> secret `hf-token-secret` key `token` |

## 2. Gated repos the CI token must accept

Accept the license on each of these pages while logged in as the CI
account; the "Agree and access repository" button must be clicked on the
HF web UI for gated repos - an API-only token does not grant license
acceptance by itself.

### Diffusion (Black Forest Labs)

Used by the Diffusion X2I(&A&T) function tests on both L4 and H100, and
by the quantization / CPU-offload tests:

- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- https://huggingface.co/black-forest-labs/FLUX.2-dev
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

### Diffusion (Stability AI)

Used by the SD3 expansion tests and the Audio Generation test:

- https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
- https://huggingface.co/stabilityai/stable-audio-open-1.0

## 3. Diagnostics

- `vllm_omni/diffusion/model_loader/hub_prefetch.py` (`prefetch_subfolders`)
  is invoked at the start of every diffusion pipeline `__init__`. On auth /
  gating failures it logs a dedicated line pointing back to this doc, so
  `grep -F "HF_TOKEN"` on a failed build log should surface it without
  scrolling through the later transformers stack trace.
- On a dev box you can warm the cache (and verify token access in bulk)
  by running `hf download` against each repo listed in the previous
  section, e.g.:

  ```bash
  export HF_TOKEN=...   # or `hf auth login` once
  for repo in \
      black-forest-labs/FLUX.1-dev \
      black-forest-labs/FLUX.1-schnell \
      black-forest-labs/FLUX.1-Kontext-dev \
      black-forest-labs/FLUX.2-dev \
      black-forest-labs/FLUX.2-klein-4B \
      stabilityai/stable-diffusion-3.5-medium \
      stabilityai/stable-audio-open-1.0 ; do
    hf download "$repo" || echo "[FAIL] $repo"
  done
  ```

  The first 401 / GatedRepoError in that loop points at the exact license
  that still needs to be accepted.
- If you hit a `GatedRepoError` locally, `hf auth whoami` verifies the
  active token, and `hf auth login` re-seats one with `HF_TOKEN` env or
  interactive browser flow.

## 4. Adding a new gated dependency

1. Bump the license-acceptance list in this file with the new repo URL.
2. If the new repo is loaded per-subfolder via `from_pretrained(subfolder=...)`,
   wire `prefetch_subfolders(model, [...])` into the pipeline `__init__` as
   documented in `hub_prefetch.py` (the helper also guards against the
   `transformers` v5 multi-worker download race documented there).
3. Coordinate with the CI owner to accept the license on the CI account.
