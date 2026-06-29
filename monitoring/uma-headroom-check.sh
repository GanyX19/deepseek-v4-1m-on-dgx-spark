#!/usr/bin/env bash
# uma-headroom-check.sh — preventive UMA-OOM guard for GB10 nodes.
#
# The failure mode (docs/known-issues.md #3) is a SILENT host hard-freeze: once the unified-memory
# headroom is eaten (model + KV + CUDA graphs + page cache + uvm fragmentation over time), the box
# locks up with NO vLLM traceback and NO kernel OOM line — the freeze happens before anything flushes.
# A post-mortem log won't help; you must catch the *creep* BEFORE the freeze. This reports free UMA
# and exits non-zero (optionally POSTs a webhook) when it drops below a threshold, so you can wire it
# into your own alerting and/or step --gpu-memory-utilization down.
#
# Run on each serving node via cron/systemd (~60 s), or from a monitoring host over SSH.
set -euo pipefail
THRESHOLD_GB="${UMA_FREE_MIN_GB:-12}"   # alert below this many GB free
WEBHOOK_URL="${UMA_WEBHOOK_URL:-}"      # optional: your own endpoint (Alertmanager / Slack / ntfy / ...)

free_gb=$(free -g | awk '/Mem:/{print $4}')
total_gb=$(free -g | awk '/Mem:/{print $2}')
host=$(hostname)

if [ "${free_gb}" -lt "${THRESHOLD_GB}" ]; then
  msg="UMA headroom LOW on ${host}: ${free_gb}G free of ${total_gb}G (< ${THRESHOLD_GB}G) — OOM-hard-freeze risk. Lower --gpu-memory-utilization or shed load."
  echo "ALERT: ${msg}" >&2
  if [ -n "${WEBHOOK_URL}" ]; then
    curl -fsS -m 10 -X POST "${WEBHOOK_URL}" -H 'Content-Type: application/json' \
         -d "{\"text\":\"${msg}\"}" >/dev/null || true
  fi
  exit 1
fi
echo "OK: ${host} ${free_gb}G free of ${total_gb}G (>= ${THRESHOLD_GB}G headroom)"
