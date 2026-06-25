# DeepSeek-V4-Flash on 2× DGX Spark (GB10) — a reproducible serving recipe

Running **DeepSeek-V4-Flash (FP8)** with up to **1 M token context** across **two NVIDIA DGX Spark
(GB10 / sm_121)** boxes with vLLM, tensor-parallel over a RoCEv2 interconnect.

This repo documents the **hardware bring-up and the vLLM serving recipe** — the parts that run *on
the Spark cluster*. It is a **reference / recipe, not a turnkey product**: it is specific to GB10
hardware and a particular point-in-time vLLM build. It does **not** ship binaries or model weights
(see [Licensing](#licensing)).

> Scope: cluster + build + serve only. Any request-routing / proxy / failover / monitoring layer you
> put in front of the engine is intentionally out of scope.

## Why this is non-trivial

- **GB10 is sm_121** — stock vLLM 0.23 will not load DeepSeek-V4-Flash on it; you need the sm12x
  enablement from an upstream PR (see [`build/`](build/README.md)).
- **Unified memory (UMA)** — there is no separate VRAM pool; the model, KV cache, CUDA graphs and
  every co-located process share the same ~120 GB. `--gpu-memory-utilization` and page-cache
  behaviour matter a lot (see [`docs/hardware-bringup.md`](docs/hardware-bringup.md)).
- **Two nodes, tensor-parallel** — TP=2 over a RoCEv2 (CX7) link, with a known post-reboot GID-index
  quirk.
- **Known firmware/driver foot-guns** — a GSP hard-lock under sustained 1 M load, an MTP
  speculative-decoding crash mode, Marlin-MoE lock-ups — all documented in
  [`docs/known-issues.md`](docs/known-issues.md) so you don't rediscover them the hard way.

## Layout

| Path | What |
|---|---|
| [`build/`](build/README.md) | How to build the sm_121 vLLM image (PR ref, the "checkout-not-merge" trick, the torch-pin gotcha). |
| [`launch/`](launch/) | `vllm serve` templates: `serve-1m.sh` (1 M context) and `serve-256k.sh` (higher aggregate throughput). |
| [`docs/hardware-bringup.md`](docs/hardware-bringup.md) | Prereqs, UMA, RoCE/NCCL, the GID-index fix, the drop-caches trick. |
| [`docs/known-issues.md`](docs/known-issues.md) | The foot-guns + workarounds. |
| [`docs/benchmark-results.md`](docs/benchmark-results.md) | Single-stream & aggregate throughput, quality numbers, the context↔throughput trade-off, the validation gates. |
| [`bench/`](bench/README.md) | Direct-to-engine benchmark approach. |

## Quickstart (once the image is built — see `build/`)

```bash
# fill in your two node IPs and (if gated) HF token first
cp .env.example .env && $EDITOR .env

# launch the 1M-context cluster (TP=2) — run from the head node
bash launch/serve-1m.sh
# wait ~80 s for boot (instanttensor load + KV alloc + cudagraph capture)
curl -s http://localhost:8000/v1/models | jq .
```

## The context ↔ throughput trade-off (TL;DR)

| Config | Single-stream | Aggregate (saturated) | seqs cap |
|---|---|---|---|
| 1 M context | ~37 tok/s | ~100 tok/s | 6 |
| 256 K context | ~40 tok/s | ~150 tok/s | 24 |

1 M context forces a low `--max-num-seqs` (each sequence reserves a lot of KV), which caps aggregate
throughput. Pick the config that matches your workload. Details + measurement method in
[`docs/benchmark-results.md`](docs/benchmark-results.md).

## Licensing

- This repo's own text/scripts: **Apache-2.0** (see [`LICENSE`](LICENSE)).
- It builds on **vLLM** (Apache-2.0) plus an upstream **sm12x PR** — see [`NOTICE`](NOTICE),
  [`CHANGES.md`](CHANGES.md), [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
- **No binaries, no CUDA, no model weights are distributed here.** You build the image yourself
  (so NVIDIA's CUDA redistribution terms stay between you and NVIDIA) and pull the model from its
  official source under its own license. See `build/` and `docs/`.

## Disclaimer

Provided **as-is, without warranty** (Apache-2.0 §7). Hardware-specific (GB10). Not affiliated with
or endorsed by NVIDIA, the vLLM project, or DeepSeek.
