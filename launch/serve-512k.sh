#!/usr/bin/env bash
# DeepSeek-V4-Flash · 512 K context · TP=2 — the middle-ground variant.
#
# Half the per-sequence KV of 1 M, double the context of 256 K. Pick this when you want a long window
# but still more concurrency than 1 M allows (≈12–16 concurrent vs ~6 at 1 M). Same recipe as the
# others; only --max-model-len / --max-num-seqs differ.
set -euo pipefail
: "${HEAD_IP:?set HEAD_IP}" "${WORKER_IP:?set WORKER_IP}"
IMAGE="${VLLM_IMAGE:-vllm-gb10-dsv4}"

DROP='docker rm -f dsv4-dropcaches >/dev/null 2>&1; docker run -d --privileged --name dsv4-dropcaches --entrypoint sh '"$IMAGE"' -c "while true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 60; done"'
bash -c "$DROP"
ssh -o BatchMode=yes "$WORKER_IP" "$DROP"

exec env \
  TILELANG_CLEANUP_TEMP_FILES=1 \
  VLLM_TRITON_MLA_SPARSE=1 VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  VLLM_USE_B12X_MOE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  UCX_MEM_MMAP_HOOK_MODE=none UCX_RCACHE_MAX_UNRELEASED=1024 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4 --trust-remote-code \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000 \
    --max-model-len 524288 --max-num-seqs 12 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" --kv-cache-dtype fp8 --block-size 256 \
    --load-format instanttensor \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 --tokenizer-mode deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --enable-prefix-caching --max-cudagraph-capture-size 8

# UCX_* vars: THE fix for the gradual UMA-OOM freeze on multi-node TP=2 — see docs/known-issues.md #3.
# Keep them; without them the inter-node UCX registration cache leaks per request until the host freezes.
#
# --max-num-seqs 12: each 512 K sequence reserves ~half the KV of a 1 M one, so you can roughly double
# the concurrency vs the 1 M recipe. With the UCX leak fixed you can raise GPU_MEM_UTIL to 0.82 and
# --max-num-seqs to ~16 on a clean/dedicated node for more parallel sessions (the seqs cap is a
# ceiling, not a forced load — it only affects per-request latency once you actually hit that many
# concurrent requests; a single request runs at full speed regardless). Verify boot headroom stays
# above your clean-crash reserve (see known-issues #3).
#
# --gpu-memory-utilization 0.80: util is bounded by per-node UMA headroom, NOT by a leak (fixed). At
# 1 M/512 K the KV pool is util-driven (~1.2–1.4 M tokens); the smaller window just yields a higher
# "Maximum concurrency Nx". The weights (~74.5 GB/node) are fixed.
#
# Production status: this 512 K / util 0.80 / seqs 16 config is what we run in LIVE PRODUCTION on
# 2x GB10 — stable, no leak (UCX fix #3), KV pool ~1.18 M tokens, ~7 GB idle headroom per node.
# util 0.80 is the proven production point: headroom is tight by design (util is UMA-bounded), so we
# run 0.80 (not 0.82) and keep the UMA monitor (monitoring/uma-headroom-check.sh) as the standard
# guardrail. Load bursts narrow the idle headroom (we see ~7 GB -> ~4 GB) but the clean-crash reserve
# (#3) plus the monitor keep that safe. 0.82/clean-node stays an optional headroom-permitting bump.