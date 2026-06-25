# Changes relative to upstream vLLM

Per Apache-2.0 §4(b), the modifications this recipe applies to vLLM:

- **Base:** vLLM, built from the head of **PR #41834** ("Add SM12x support for DeepSeek V4 Flash",
  jasl fork, branch `codex/ds4-sm120-min-enable`).
  - Pinned build: **vLLM `0.23.1rc1.dev407+g28fef2c70`** (record the exact head commit you build).
- **Build process change:** the PR is a large fork branch (`mergeable: dirty`); instead of *merging*
  it onto `main` (which conflicts), the build **checks out the PR head directly**
  (`git fetch origin pull/41834/head:pr-41834 && git checkout pr-41834`). See `build/README.md`.
- **Torch pin fix:** re-pin the CUDA build of `torch` (cu130) *after* the vLLM wheel install, to
  avoid a `+cpu` torch clobbering it (`libtorch_cuda.so` ImportError at launch). See `build/README.md`.
- **No source patches beyond the above are applied by this repository.** All model/kernel changes
  come from PR #41834 upstream.

This repository itself adds only configuration, launch templates, benchmarks, and documentation —
no changes to vLLM source.
