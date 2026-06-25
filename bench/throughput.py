#!/usr/bin/env python3
"""Decode-throughput benchmark for a local OpenAI-compatible inference engine.

This script measures *decode* (generation) throughput of the model server by
firing short chat-completion requests at increasing concurrency levels and
reporting how many output tokens per second the engine sustains.

It talks DIRECTLY to the engine at http://localhost:8000/v1/chat/completions
(no proxy, no API key, no secrets) so the numbers reflect the engine itself
rather than any routing/auth layer in front of it.

Why a discarded warmup round matters
-------------------------------------
The first requests an engine sees after start (or after a config change) pay
one-time costs that have nothing to do with steady-state throughput: CUDA/JIT
kernel compilation, autotuning of GEMM/attention kernels, cuBLAS/cuDNN
algorithm selection, graph capture, and allocator warmup. If you fold those
cold-start requests into the measurement they drag the numbers down and make
runs non-reproducible. So we always do one WARMUP round at each concurrency
level and throw its timings away; only the second, hot round is measured.

To keep the engine honest we also vary the prompts. Identical prompts let
prefix caching serve most of the work from cache, which inflates throughput
and stops being a decode benchmark. Varied prompts force real prefill+decode.

Metrics reported per concurrency level:
  * aggregate tok/s   - total output tokens / wall-clock of the round
  * per-stream tok/s  - mean of each request's own output-tokens / latency
  * req/s             - completed requests / wall-clock of the round
  * p50 / p99         - request latency percentiles (seconds)

Stdlib only: urllib + concurrent.futures. No third-party deps.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4"
DEFAULT_SWEEP = [1, 6, 12, 24, 32]

# Varied prompt fragments so prefix caching can't trivialize the run. Each
# request gets a distinct topic + index, forcing real prefill and decode.
PROMPT_TOPICS = [
    "the water cycle",
    "how a bicycle gear system works",
    "the difference between TCP and UDP",
    "why the sky is blue",
    "how sourdough bread rises",
    "the basics of double-entry bookkeeping",
    "how a heat pump moves heat",
    "the role of mitochondria in a cell",
    "how GPS determines position",
    "what causes ocean tides",
    "how a transistor switches current",
    "the idea behind public-key cryptography",
    "how vaccines train the immune system",
    "why metals conduct electricity",
    "how a jet engine produces thrust",
    "the concept of compound interest",
]


def make_prompt(i: int) -> str:
    topic = PROMPT_TOPICS[i % len(PROMPT_TOPICS)]
    return (
        f"In about four sentences, explain {topic}. "
        f"Keep it concrete and clear. (variation #{i})"
    )


def one_request(url, model, prompt, max_tokens, temperature, timeout):
    """Fire a single chat completion. Returns (latency_s, completion_tokens, ok)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        # Disable "thinking" so we measure plain decode, not reasoning tokens.
        "chat_template_kwargs": {"thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return (time.perf_counter() - t0, 0, False, str(e))
    except Exception as e:  # noqa: BLE001 - benchmark robustness
        return (time.perf_counter() - t0, 0, False, str(e))

    usage = body.get("usage") or {}
    # Prefer the engine-reported completion token count; fall back to a rough
    # whitespace estimate if usage is missing.
    ctoks = usage.get("completion_tokens")
    if ctoks is None:
        try:
            text = body["choices"][0]["message"]["content"] or ""
            ctoks = max(1, len(text.split()))
        except (KeyError, IndexError, TypeError):
            ctoks = 0
    return (dt, int(ctoks), True, None)


def run_round(url, model, model_args, concurrency, n_requests, start_index):
    """Run one round of n_requests at the given concurrency. Returns a dict of stats."""
    max_tokens, temperature, timeout = model_args
    latencies = []
    total_tokens = 0
    errors = 0
    per_stream_rates = []

    t_round = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                url,
                model,
                make_prompt(start_index + i),
                max_tokens,
                temperature,
                timeout,
            )
            for i in range(n_requests)
        ]
        for fut in as_completed(futures):
            dt, ctoks, ok, _err = fut.result()
            if not ok:
                errors += 1
                continue
            latencies.append(dt)
            total_tokens += ctoks
            if dt > 0 and ctoks > 0:
                per_stream_rates.append(ctoks / dt)
    wall = time.perf_counter() - t_round

    ok_count = len(latencies)
    agg_toks = total_tokens / wall if wall > 0 else 0.0
    per_stream = (sum(per_stream_rates) / len(per_stream_rates)) if per_stream_rates else 0.0
    reqs = ok_count / wall if wall > 0 else 0.0
    return {
        "concurrency": concurrency,
        "wall": wall,
        "ok": ok_count,
        "errors": errors,
        "agg_toks": agg_toks,
        "per_stream": per_stream,
        "req_s": reqs,
        "p50": percentile(latencies, 50),
        "p99": percentile(latencies, 99),
    }


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # Linear-interpolation percentile.
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Direct decode-throughput benchmark for a local OpenAI-compatible engine."
    )
    p.add_argument(
        "concurrency",
        nargs="*",
        type=int,
        default=DEFAULT_SWEEP,
        help=f"Concurrency levels to sweep (default: {DEFAULT_SWEEP}).",
    )
    p.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint (default: {DEFAULT_URL}).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--max-tokens", type=int, default=128, help="Output tokens per request (default: 128).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7).")
    p.add_argument(
        "--requests-per-level",
        type=int,
        default=None,
        help="Requests per measured round. Default: max(concurrency*4, 16).",
    )
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout seconds (default: 300).")
    args = p.parse_args(argv)

    levels = args.concurrency or DEFAULT_SWEEP
    model_args = (args.max_tokens, args.temperature, args.timeout)

    print(f"endpoint : {args.url}")
    print(f"model    : {args.model}")
    print(f"settings : max_tokens={args.max_tokens} temperature={args.temperature} thinking=off")
    print(f"sweep    : {levels}")
    print()

    # Rolling prompt index so every request across the whole run is distinct.
    idx = 0
    header = (
        f"{'conc':>5} {'agg tok/s':>11} {'per-strm tok/s':>15} "
        f"{'req/s':>8} {'p50 s':>8} {'p99 s':>8} {'errs':>5}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for c in levels:
        n = args.requests_per_level if args.requests_per_level else max(c * 4, 16)

        # WARMUP round (discarded): absorbs cold-start JIT/autotune costs.
        _ = run_round(args.url, args.model, model_args, c, n, idx)
        idx += n

        # MEASURED round (hot).
        r = run_round(args.url, args.model, model_args, c, n, idx)
        idx += n
        results.append(r)

        print(
            f"{r['concurrency']:>5} {r['agg_toks']:>11.1f} {r['per_stream']:>15.1f} "
            f"{r['req_s']:>8.2f} {r['p50']:>8.2f} {r['p99']:>8.2f} {r['errors']:>5}"
        )
        sys.stdout.flush()

    print()
    if any(r["errors"] for r in results):
        print("note: some requests errored; per-level numbers cover successful requests only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
