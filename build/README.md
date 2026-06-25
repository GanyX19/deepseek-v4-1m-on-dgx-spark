# Building vLLM for DeepSeek-V4-Flash on GB10 (sm_121)

Stock vLLM (≤ 0.23.x) **does not load DeepSeek-V4-Flash on GB10**: the sm12x kernels / model wiring
land in an upstream PR that was not yet merged at the time of writing. You build vLLM from that PR.

## What you need from upstream

- **Base:** [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
- **sm12x enablement:** vLLM **PR #41834** — *"Add SM12x support for DeepSeek V4 Flash"* (from the
  `jasl` fork, branch `codex/ds4-sm120-min-enable`). Credit to its author. Pin the exact head commit
  you build (record it in [`../CHANGES.md`](../CHANGES.md)); the branch moves.
- Reference build used here: **vLLM `0.23.1rc1.dev407+g28fef2c70`**.

## The two gotchas that cost the most time

### 1. The PR is a large fork branch → check out its head, don't merge it

PR #41834 is effectively a **multi-commit fork branch** and reports `mergeable: dirty` — merging it
onto a current `main` always conflicts. **Don't merge it. Build its head directly.** Concretely, in
the build's "apply PR" step, *fetch then `checkout` the PR head* instead of `merge`:

```dockerfile
# fetch the PR head and BUILD it directly (self-contained tree, no merge conflicts)
RUN git fetch origin pull/41834/head:pr-41834 && git checkout pr-41834
```

GitHub serves a fork's PR head under the base repo's `pull/<N>/head` ref, so a plain
`git fetch origin pull/41834/head` works even though the branch lives on a fork.

### 2. The torch-CPU pin clobber

vLLM's main wheel may pin a `torch==X.Y.Z+cpu` build. If your wheel install runs *after* you install
the CUDA torch, it silently replaces it → at launch you get `ImportError: libtorch_cuda.so`.
**Re-pin the CUDA torch *after* the wheel install:**

```dockerfile
RUN pip install --no-cache-dir "torch==<X.Y.Z>" --index-url https://download.pytorch.org/whl/cu130
```

(Match the torch/CUDA versions to your GB10 driver stack.)

## Build (sketch)

This repo does not republish a full internal build harness. The essential steps:

```bash
# 1. Build the vLLM wheel/image from the PR head, for the GB10 arch.
#    Key build args: TORCH_CUDA_ARCH_LIST=12.1a  FLASHINFER_CUDA_ARCH_LIST=12.1a
#    and the PR-head checkout from gotcha #1.
#
# 2. Copy the resulting image to BOTH nodes (the TP=2 cluster needs it on each).
#    e.g. docker save <img> | ssh $WORKER_IP "docker load"
```

A minimal Dockerfile that clones vLLM, checks out the PR head, sets the GB10 arch, builds, and
re-pins CUDA torch is all that's required — the two gotchas above are the only non-obvious parts.

## Notes

- Build is fast (~10 min) once dependency wheels are cached; a cold build is much longer.
- **Pin the working commit.** Newer upstream commits regressed for us more than once — see
  [`../docs/known-issues.md`](../docs/known-issues.md).
- Do **not** ship the built image publicly without handling NVIDIA's CUDA redistribution terms —
  publishing the *recipe* (this) keeps you clear of that.
