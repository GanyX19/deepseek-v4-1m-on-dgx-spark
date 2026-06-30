# DeepSeek-V4-Flash on 2× DGX Spark (GB10) — a reproducible serving recipe

Running **DeepSeek-V4-Flash (FP8)** with up to **1 M token context** across **two NVIDIA DGX Spark
(GB10 / sm_121)** boxes with vLLM, tensor-parallel over a RoCEv2 interconnect.

This repo documents the **hardware bring-up and the vLLM serving recipe** — the parts that run *on
the Spark cluster*. It is a **reference / recipe, not a turnkey product**: it is specific to GB10
hardware and a particular point-in-time vLLM build. It does **not** ship binaries or model weights
(see [Licensing](#licensing)).

> Scope: cluster + build + serve only. Any request-routing / proxy / failover / monitoring layer you
> put in front of the engine is intentionally out of scope.

## Why this exists

I spent weeks trying to get DeepSeek-V4-Flash running *stably* at long context on this hardware —
many vLLM builds and commits, MTP on and off, FP8 and other quantizations, context lengths from
128 K to 1 M, `--gpu-memory-utilization` and `--max-num-seqs` sweeps, and driver/firmware updates.

The recurring wall was a **GSP firmware hard-lock under sustained long-context load** — it barely
cared which recipe I used, and newer drivers/firmware did not make it disappear. I chased it to the
bottom: it is a **hardware/firmware-level lockup (NVIDIA GB10 / sm_121, tracked as #1111), not a
vLLM setting you can tune away.** What actually held, what didn't, and the practical escapes
(shorter context, capped concurrency, an external power-cycle watchdog) are written down honestly in
[`docs/known-issues.md`](docs/known-issues.md).

I'm publishing the whole thing — the build, the serve recipe, and above all the **landmines** — so
you don't have to burn the same weeks rediscovering them.

## Why this is non-trivial

- **GB10 is sm_121** — stock vLLM 0.23 will not load DeepSeek-V4-Flash on it; you need the sm12x
  enablement from an upstream PR (see [`build/`](build/README.md)).
- **Unified memory (UMA)** — there is no separate VRAM pool; the model, KV cache, CUDA graphs and
  every co-located process share the same ~120 GB. `--gpu-memory-utilization` and page-cache
  behaviour matter a lot: too high a util **silently hard-freezes the host** after a while (no
  traceback, no kernel OOM line), so the serve scripts default to a headroom-leaving **0.78** and
  honour `GPU_MEM_UTIL` to override. See [`docs/hardware-bringup.md`](docs/hardware-bringup.md) and
  [`docs/known-issues.md`](docs/known-issues.md) #3.
- **Two nodes, tensor-parallel** — TP=2 over a RoCEv2 (CX7) link, with a known post-reboot GID-index
  quirk.
- **Known firmware/driver foot-guns** — a GSP hard-lock under sustained 1 M load, a silent UMA-OOM
  hard-freeze that turned out to be a **UCX RDMA-cache leak** (fixed by two env vars — the single most
  important finding here), an MTP speculative-decoding crash mode, Marlin-MoE lock-ups — all
  documented in [`docs/known-issues.md`](docs/known-issues.md) so you don't rediscover them the hard
  way. **If you run multi-node TP=2, read [known-issues #3](docs/known-issues.md) before anything
  else.**

## Layout

| Path | What |
|---|---|
| [`build/`](build/README.md) | How to build the sm_121 vLLM image (PR ref, the "checkout-not-merge" trick, the torch-pin gotcha). |
| [`launch/`](launch/) | `vllm serve` templates: `serve-256k.sh` (max throughput), `serve-512k.sh` (middle ground), `serve-1m.sh` (max context). All three set the **UCX leak-fix** env vars. |
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
| 256 K context | ~40 tok/s | ~150 tok/s | 24 |
| 512 K context | ~40 tok/s | ~130 tok/s¹ | 12–16 |
| 1 M context | ~37 tok/s | ~100 tok/s | 6 |

A larger context forces a lower `--max-num-seqs` (each sequence reserves more KV), which caps
aggregate throughput and concurrency — so **pick the smallest window that covers your requests** for
the most parallel users. ¹512 K aggregate is interpolated (boot + serving verified; not separately
benchmarked). With the UCX leak fixed ([known-issues #3](docs/known-issues.md)) the choice is now
purely this context-vs-concurrency trade-off — **not** a stability question. Note `--max-num-seqs` is
a *ceiling*, not a forced load: a lone request runs at full speed regardless; the cap only shapes
behaviour once you actually hit that many concurrent requests (high cap = no queuing/low TTFT under
load; low cap = guaranteed per-request speed but queuing). Method in
[`docs/benchmark-results.md`](docs/benchmark-results.md).

> The aggregate numbers above were measured at a throughput-peak `--gpu-memory-utilization` (0.85).
> The serve scripts now default to **0.78** for stable sustained operation, which trades a little
> aggregate throughput (smaller KV pool) for not silently freezing the host — see
> [`docs/known-issues.md`](docs/known-issues.md) #3. Push util back up with `GPU_MEM_UTIL` only on a
> clean, dedicated, monitored node.

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
