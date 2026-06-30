#!/usr/bin/env bash
# DeepSeek-V4-Flash · 256 K context · TP=2 — the higher-aggregate-throughput variant.
#
# Same recipe as serve-1m.sh but a smaller context lets each sequence reserve far less KV,
# so you can run many more concurrent sequences → ~150 tok/s aggregate (vs ~100 at 1 M).
# Pick this if you serve many short/medium requests and don't need the full 1 M window.
set -euo pipefail
: "${HEAD_IP:?set HEAD_IP}" "${WORKER_IP:?set WORKER_IP}"
IMAGE="${VLLM_IMAGE:-vllm-gb10-dsv4}"

DROP='docker rm -f dsv4-dropcaches >/dev/null 2>&1; docker run -d --privileged --name dsv4-dropcaches --entrypoint sh '"$IMAGE"' -c "while true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 60; done"'
bash -c "$DROP"
ssh -o BatchMode=yes "$WORKER_IP" "$DROP"

exec env \
  TILELANG_CLEANUP_TEMP_FILES=1 \
  VLLM_TRITON_MLA_SPARSE=1 VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  VLLM_USE_B12X_MOE=1 \
  UCX_MEM_MMAP_HOOK_MODE=none UCX_RCACHE_MAX_UNRELEASED=1024 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4 --trust-remote-code \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --max-num-seqs 24 --max-num-batched-tokens 6144 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" --kv-cache-dtype fp8 --block-size 256 \
    --load-format instanttensor \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 --tokenizer-mode deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --enable-prefix-caching --max-cudagraph-capture-size 8

# UCX_MEM_MMAP_HOOK_MODE / UCX_RCACHE_MAX_UNRELEASED: THE fix for the gradual UMA-OOM freeze. On
# 2-node TP=2 the inter-node RDMA transport (UCX) leaks its registration cache ~per request until the
# unified memory is exhausted and the host silently hard-freezes. These two vars bound it. Keep them.
# Full story + the A/B that proved it: docs/known-issues.md #3.

# --max-num-seqs: aggregate throughput peaked around 24 in our tests; ~32 regressed
# (scheduler/KV pressure). Sweep it for your workload.

# --gpu-memory-utilization: with the UCX leak fixed (above), util is NO LONGER constrained by a
# creeping leak — only by the per-node UMA headroom your runtime working set needs. We measured on
# GB10: 0.85 leaves only ~1 GB free/node (too tight — a load burst OOMs the working set); ~0.80–0.82
# is the practical max (≈8–13 GB free). The weights (~74.5 GB/node) are fixed; util sizes the KV pool.
# Default 0.80; GPU_MEM_UTIL=0.82 on a clean/dedicated node, lower if you co-locate other services.
# (Earlier versions of this recipe lowered util "for stability" — that was treating the symptom; the
# real cause was the UCX leak above.)
