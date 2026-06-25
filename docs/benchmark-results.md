# Benchmark results & validation method

Numbers are from our 2× GB10 cluster on the sm12x build (vLLM `0.23.1rc1.dev407`). Treat them as
**order-of-magnitude for this hardware**, not guarantees.

## Measurement method (so the numbers mean something)

- Hit the engine **directly** (`:8000`), not through any proxy.
- **Warm up first** and discard the cold-start run (Triton JIT + flashinfer autotune — see
  [`known-issues.md`](known-issues.md#6-cold-start-jit--autotune-skews-first-requests)).
- For clean decode throughput: short outputs (~128 tokens), thinking off, varied prompts (so prefix
  caching doesn't trivialize it).
- Sweep concurrency; report aggregate tok/s, per-stream tok/s, and p50/p99 latency.

## Throughput — 1 M config (`--max-num-seqs 6`)

| Concurrency | Aggregate tok/s | Per-stream | p50 latency |
|---|---|---|---|
| 1 | ~37 | 37 | low |
| 6 | ~98 | ~16 | moderate |
| 12 | ~98 | ~8 | higher |
| 24 | ~101 | ~4 | high |

**Aggregate saturates at the `--max-num-seqs` cap (6).** Beyond 6 concurrent requests, throughput is
flat and only latency grows — extra clients just queue.

## Throughput — 256 K config (`--max-num-seqs 24`)

~**150 tok/s** aggregate at saturation; ~32 seqs regressed (scheduler/KV pressure). Single-stream
~40 tok/s.

## The context ↔ throughput trade-off

1 M context forces a low seqs cap because **each sequence reserves a large slice of the KV pool**
(pool ≈ 2.0 M tokens at 1 M ⇒ only ~2× full-context concurrency). For **short** requests the engine
is therefore **seqs-bound, not KV-bound** — raising `--max-num-seqs` can lift aggregate toward the
256 K numbers, at the cost of how many full-1 M requests can run at once. Sweep it for your workload.

## Quality

A 33-prompt battery (coding / summarization / structured-output / extraction), judged by a strong
LLM judge, scored **~8.0–8.7 / 10** on the sm12x build — on par with the older build (no regression
from the version bump). Single-stream speed with MTP ≈ 37–40 tok/s. JSON-mode output was valid and
no language drift was observed in the German/English prompts tested.

## Validation gates (how to qualify a new build)

Before promoting any new image, gate it (we run prod traffic on a fallback during this):

- **a — Boot + KV:** `/health` 200, model loads, full KV pool allocates without OOM / CUDA error /
  unknown-parser.
- **b — MTP crash test:** sustained **non-greedy + greedy concurrency** with MTP on; watch for
  `EngineDeadError` / shm-wedge / TP hang. This is the test that catches the
  [MTP crash](known-issues.md#2-mtp-speculative-decoding-crash-on-older-vllm).
- **c — Freeze soak:** ~30 min zero-gap concurrency; watch `dmesg` for Xid / host hang
  ([GSP lock-up](known-issues.md#1-gsp-firmware-hard-lock-under-sustained-1-m-load)).
- **d — Quality smoke:** a judged prompt subset + explicit JSON-validity and output-language checks.

A build that fails b or c does **not** ship.
