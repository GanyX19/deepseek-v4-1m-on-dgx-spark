# Third-party components

This repository contains only configuration, scripts, and documentation (Apache-2.0). It does
**not** bundle or redistribute any third-party binaries, libraries, CUDA components, or model
weights. The components below are what the *recipe builds on* — you obtain each from its own source
under its own license.

| Component | Role | License | Source |
|---|---|---|---|
| vLLM | inference engine | Apache-2.0 | https://github.com/vllm-project/vllm |
| vLLM PR #41834 (jasl) | sm12x / DeepSeek-V4 enablement | Apache-2.0 (vLLM contribution) | https://github.com/vllm-project/vllm/pull/41834 |
| PyTorch | tensor runtime | BSD-3-Clause | https://github.com/pytorch/pytorch |
| FlashInfer | attention kernels | Apache-2.0 | https://github.com/flashinfer-ai/flashinfer |
| TileLang | kernel compilation | (check upstream) | https://github.com/tile-ai/tilelang |
| NCCL / CUDA / cuDNN | GPU runtime | **NVIDIA proprietary (CUDA EULA)** — not redistributed here | https://developer.nvidia.com |
| DeepSeek-V4-Flash | model weights | DeepSeek model license — not redistributed here | https://huggingface.co/deepseek-ai |

If you choose to build and then **redistribute a binary/image**, you become the distributor and must
satisfy each component's terms — in particular NVIDIA's CUDA redistribution terms and a complete
`THIRD_PARTY_LICENSES` of everything bundled. Verify licenses marked "(check upstream)" against the
exact versions you ship.
