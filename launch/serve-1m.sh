#!/usr/bin/env bash
# DeepSeek-V4-Flash · 1 M context · TP=2 across two GB10 nodes.
#
# This shows the `vllm serve` invocation + environment + the drop-caches trick.
# Multi-node TP=2 is orchestrated by launch/launch-cluster.sh (included in this repo);
# this file is the serve command it runs on the head node. See ../INSTALL.md for the full flow.
#
# Fill HEAD_IP / WORKER_IP / VLLM_IMAGE via .env (see ../.env.example).
set -euo pipefail
: "${HEAD_IP:?set HEAD_IP}" "${WORKER_IP:?set WORKER_IP}"
IMAGE="${VLLM_IMAGE:-vllm-gb10-dsv4}"

# --- drop-caches sidecar on BOTH nodes DURING load ---
# On UMA the Linux page cache competes with the model/KV for the same ~120 GB; while loading
# ~149 GB of weights the page cache can push the free-memory check over the edge. A privileged
# sidecar drops caches every 60 s during load. Remove it once :8000 is ready (post-boot hook).
DROP='docker rm -f dsv4-dropcaches >/dev/null 2>&1; docker run -d --privileged --name dsv4-dropcaches --entrypoint sh '"$IMAGE"' -c "while true; do sync; echo 3 > /proc/sys/vm/drop_caches; sleep 60; done"'
bash -c "$DROP"
ssh -o BatchMode=yes "$WORKER_IP" "$DROP"

# --- the serve command (run on the head; worker joined via your TP launcher) ---
exec env \
  TILELANG_CLEANUP_TEMP_FILES=1 \
  VLLM_TRITON_MLA_SPARSE=1 VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  VLLM_USE_B12X_MOE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  UCX_MEM_MMAP_HOOK_MODE=none UCX_RCACHE_MAX_UNRELEASED=1024 \
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-v4 --trust-remote-code \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000 \
    --max-model-len 1000000 --max-num-seqs 6 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" --kv-cache-dtype fp8 --block-size 256 \
    --load-format instanttensor \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 --tokenizer-mode deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --enable-prefix-caching --max-cudagraph-capture-size 8

# After :8000 answers, remove the sidecars:
#   docker rm -f dsv4-dropcaches ; ssh "$WORKER_IP" docker rm -f dsv4-dropcaches
#
# Flags worth understanding (see ../docs/):
#   --max-model-len 1000000   1 M context; needs VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
#   --max-num-seqs 6          KV-bound at 1 M (each seq reserves a lot of KV) → low concurrency cap
#   --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}"  conservative: UMA is shared, do NOT use 0.9 on a non-clean node
#   --kv-cache-dtype fp8 --block-size 256   smaller KV footprint
#   --speculative-config mtp/2   MTP speculative decoding (~+50% single-stream); see known-issues for the crash history
#   --tokenizer/--tool-call/--reasoning-parser deepseek_v4   DS4-specific parsers (provided by the sm12x build)
#   --load-format instanttensor   fast weight load (~80 s cold)
