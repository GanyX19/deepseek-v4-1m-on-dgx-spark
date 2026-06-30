# Known issues & foot-guns (GB10 + DeepSeek-V4-Flash + vLLM)

Hard-won; documented so you don't pay for them twice. Severity is for a production serving setup.

## 1. "1 M GSP firmware hard-lock" — mostly the UCX leak (#3), re-evaluated  ·  [HIGH]

Under **sustained load at 1 M context** we used to see the box **hard-lock the host** — the whole
node stops responding, not just the engine. It looked like a GSP firmware lockup and we filed it as
driver/firmware (newer drivers/firmware didn't eliminate it).

**Re-interpretation (2026-06):** most of this was almost certainly the **UCX registration-cache leak
(#3)**, not firmware. 1 M context has the **tightest UMA headroom** of any config, so the same
per-request UCX leak fills it **fastest** and freezes **soonest** — which reads exactly like a
"1 M-only" firmware lock. After applying the UCX fix (#3), **both 1 M and 512 K boot and serve
cleanly** on our 2× GB10 (1 M: KV pool 1.43 M tokens, util 0.80, ~8 GB free; 512 K: 1.18 M tokens,
~7 GB free).

**Mitigations:**
- **Apply the UCX fix (#3) first** — it is very likely the actual cause; then re-test 1 M.
- Any genuine firmware portion is now **unproven and probably rarer than this entry implied**. We
  have not run a **>12 h** sustained-saturation 1 M soak *with the UCX fix* yet — do your own long
  soak before trusting 1 M under continuous saturation, and keep an external watchdog + the
  clean-crash hardening (#3) as a backstop.

## 2. MTP speculative-decoding crash on older vLLM  ·  [HIGH]

On **vLLM 0.22.x with MTP** (`--speculative-config method=mtp`) under **non-greedy concurrent
traffic**, the engine hits a **shm_broadcast / rejection-sampler wedge** and the EngineCore process
dies (connection-refused, *no* OOM/Xid). For us this fired ~twice a day in production.

**Fixed** in the sm12x build referenced here (vLLM `0.23.1rc1.dev407`). If you must stay on an older
build, **disable MTP**. Validate with the MTP crash test in
[`benchmark-results.md`](benchmark-results.md#validation-gates).

## 3. Gradual UMA-OOM host hard-freeze — the UCX RDMA-cache leak  ·  [HIGH]

**Root cause (found 2026-06):** a **per-request GPU/unified-memory leak (~14 MB/request under load),
freed only on container restart**, that fills the unified-memory pool until the host **silently
hard-freezes**. The leak is in **UCX** — the inter-node RDMA transport that multi-node TP=2 uses
(under NCCL) to move tensors between the two GB10 nodes every step. UCX hooks **every `mmap`** to
maintain its memory-registration cache, and with `UCX_RCACHE_MAX_UNRELEASED=inf` (the default) the
unreleased-region queue grows **unbounded, per request**, on the shared UMA pool → OOM → freeze.
Same mechanism as Mistral's vLLM-memory-leak writeup; it just lands harder here because GPU and host
RAM are one physical pool.

**THE FIX — two env vars** (now set in `launch/serve-*.sh`):
```
UCX_MEM_MMAP_HOOK_MODE=none      # stop UCX intercepting mmap for its rcache
UCX_RCACHE_MAX_UNRELEASED=1024   # bound the unreleased-region queue
```
A/B-proven on our box (256 K, 5 concurrent, varied request shapes): **without** the vars, free UMA
fell 10 → 6 GB and OOM-aborted in ~18 min (~14 MB/request); **with** them, the slope is flat (no
abort, same load). This is the actual fix — not a util tweak.

**The freeze is SILENT** — plan observability around it. No vLLM traceback, no kernel
`NV_ERR_NO_MEMORY` line (the host locks up before anything flushes). The only signatures:
- the **other** TP rank logs `ProcessGroupNCCL ... HeartbeatMonitor ... TCPStore server has shut
  down` (rank-0's host died), and
- the head node goes to **"no route to host"** (SSH dies).

So don't wait for a clean error — treat "rank lost / no route to host" as a UMA freeze.

**About `--gpu-memory-utilization`:** lowering util used to look like the fix — it isn't. More util
headroom just gives the leak more room to eat, **delaying** the freeze, never stopping it (we chased
0.85 → 0.74 for a week before finding UCX). With the UCX fix above, util is bounded **only** by the
per-node UMA headroom your runtime working set needs — not by a leak. On GB10 we measured: **0.85
leaves only ~1 GB free/node** (too tight — a load burst OOMs the working set); **~0.80–0.82 is the
practical max** (≈8–13 GB free). Note `--max-num-seqs` does **not** change this baseline reservation.

**Belt-and-suspenders (recommended regardless):** make a UMA-OOM **recoverable** instead of a host
wedge. Set `sysctl vm.min_free_kbytes=3145728` (≈3 GB reserve) and `swapoff -a` on each node, so that
if the pool is ever exhausted the kernel **OOM-kills the process cleanly** (your watchdog relaunches
it) instead of starving itself into a freeze. NVIDIA acknowledges the UMA-OOM-host-wedge as a known
GB10 issue (better OOM handling promised in a future Spark OS). Also: **alert before** the freeze —
watch free UMA (see [`monitoring/uma-headroom-check.sh`](../monitoring/uma-headroom-check.sh)) — and
run benchmarks at modest concurrency.

## 4. Marlin WNA16-MoE hangs on GB10  ·  [MEDIUM]

Marlin WNA16 MoE kernels have hung the host on GB10 for some models/quantizations. Prefer the
quant/kernel path validated for your model (for DeepSeek-V4-Flash, the FP8 path in this recipe).

## 5. NVFP4 @ 128 K OOMs the KV cache  ·  [LOW]

An NVFP4 variant at 128 K context OOM'd KV on this hardware. Stick to the `kv-cache-dtype fp8`
recipe here.

## 6. Cold-start JIT / autotune skews first requests  ·  [LOW]

The first requests after boot pay a one-time **Triton JIT + flashinfer autotune** cost. A naive
benchmark that starts measuring immediately will report a wildly low first data point. **Warm up
and discard** before measuring.

## 7. Newer upstream commits regressed  ·  [MEDIUM]

We bumped to newer vLLM commits more than once and got **slower and/or functionally broken** builds
(tool-calling / arithmetic / language drift). **Pin the exact commit you validated**, and gate every
build (next doc). Don't chase `main`.

## 8. Language drift on some prompt shapes  ·  [LOW]

DeepSeek-V4-Flash can drift to another language on certain non-thinking + system-prompt shapes.
Enabling the model's thinking mode for the affected request types removed it for us. Check output
language explicitly in your quality gate.
