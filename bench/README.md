# Benchmarking

Benchmark the engine **directly** (`http://<head>:8000/v1/...`), never through a routing/proxy layer
— you want the engine's numbers, not your stack's.

A minimal throughput probe (decode tok/s vs concurrency) is all you need:

- Fire `N` concurrent chat-completions with short outputs (~128 tokens), `temperature>0`, varied
  prompts, **thinking off** for clean decode.
- **Warm up and discard** one round first (cold-start JIT/autotune).
- Sweep concurrency (e.g. 1, 6, 12, 24, 32); report aggregate tok/s, per-stream tok/s, p50/p99.

See [`../docs/benchmark-results.md`](../docs/benchmark-results.md) for our numbers and the full
validation-gate method (boot → MTP-crash test → freeze soak → quality smoke).

> A quality battery (multi-category prompts + an LLM judge) is also worth running against `:8000`
> directly; keep it **sequential** on this hardware (parallel judge+gen load can UMA-OOM the host).
