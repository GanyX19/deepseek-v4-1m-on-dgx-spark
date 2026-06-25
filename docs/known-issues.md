# Known issues & foot-guns (GB10 + DeepSeek-V4-Flash + vLLM)

Hard-won; documented so you don't pay for them twice. Severity is for a production serving setup.

## 1. GSP firmware hard-lock under sustained 1 M load  ·  [HIGH]

Under **sustained, zero-gap load at 1 M context**, the box can **hard-lock the host** (GSP firmware
lockup; tracked upstream around NVIDIA GB10/sm_121 GSP issues). It is **driver-independent** — newer
drivers + firmware did not eliminate it in our testing. Symptoms: the whole node stops responding,
not just the engine.

**Mitigations:**
- Run **256 K** instead of 1 M if you push sustained saturation — we did not reproduce the freeze
  there.
- If you must run 1 M, put an **external watchdog with a hard power-cycle** in front (out of scope
  for this repo) and treat a freeze as expected-but-rare.
- Our 1 M build survived a **30-minute** zero-gap soak cleanly (0 Xid), but **>12 h is unproven** —
  do your own long-soak before trusting 1 M under continuous saturation.

## 2. MTP speculative-decoding crash on older vLLM  ·  [HIGH]

On **vLLM 0.22.x with MTP** (`--speculative-config method=mtp`) under **non-greedy concurrent
traffic**, the engine hits a **shm_broadcast / rejection-sampler wedge** and the EngineCore process
dies (connection-refused, *no* OOM/Xid). For us this fired ~twice a day in production.

**Fixed** in the sm12x build referenced here (vLLM `0.23.1rc1.dev407`). If you must stay on an older
build, **disable MTP**. Validate with the MTP crash test in
[`benchmark-results.md`](benchmark-results.md#validation-gates).

## 3. UMA-OOM host hang under parallel benchmark load  ·  [MEDIUM]

Hammering the engine with **high parallel load while other things run** can drive the unified-memory
pool to OOM and **hang the host**. Run benchmarks **sequentially or at modest concurrency**, and
keep `--gpu-memory-utilization` honest (see [`hardware-bringup.md`](hardware-bringup.md)).

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
