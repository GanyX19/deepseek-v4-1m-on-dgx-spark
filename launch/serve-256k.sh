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
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4 --trust-remote-code \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --max-num-seqs 24 --max-num-batched-tokens 6144 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.78}" --kv-cache-dtype fp8 --block-size 256 \
    --load-format instanttensor \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 --tokenizer-mode deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --enable-prefix-caching --max-cudagraph-capture-size 8

# --max-num-seqs: aggregate throughput peaked around 24 in our tests; ~32 regressed
# (scheduler/KV pressure). Sweep it for your workload.

# --gpu-memory-utilization: default 0.78 leaves ~15 GB of the ~120 GB UMA free, which we found
# REQUIRED for stable sustained operation. 0.85 maximizes the KV pool / aggregate throughput but
# left only ~7 GB free and crept into a silent UMA-OOM host hard-freeze after ~1 h even under light
# (~1 req/min) load — see docs/known-issues.md #3. Override GPU_MEM_UTIL=0.85 only on a clean,
# dedicated, monitored node. The weights (~74.5 GB/node) are fixed; util controls the KV pool — and
# note: lowering --max-num-seqs does NOT reduce this baseline UMA, only runtime concurrency spikes.
