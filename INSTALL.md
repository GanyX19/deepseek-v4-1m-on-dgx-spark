# INSTALL — DeepSeek-V4-Flash @ up to 1 M context on 2× DGX Spark (GB10), vLLM TP=2

This is a step-by-step recipe to serve **DeepSeek-V4-Flash** with a context window of up to
**1 000 000 tokens** across **two NVIDIA DGX Spark (GB10)** nodes, using a self-built vLLM image and
tensor parallelism (**TP=2**) over a high-speed interconnect.

The flow is: **build the image once → copy it to both nodes → fill in `.env` → launch the TP=2
cluster with a serve template → verify → (troubleshoot)**.

> Throughout, replace every `<...>` placeholder with your own values. The examples use
> `10.0.0.1` (head) and `10.0.0.2` (worker) on a dedicated `/30` link — substitute your own IPs.
> Do **not** commit your real `.env`.

---

## 0. Prerequisites

Hardware and host setup (see [`docs/hardware-bringup.md`](docs/hardware-bringup.md) for detail):

- **2× NVIDIA DGX Spark (GB10)** — Grace-Blackwell, compute capability **sm_121**, ~120 GB
  **unified memory (UMA)** each. There is **no separate VRAM**: weights, KV cache, CUDA graphs and
  every other process share the same ~120 GB pool.
- A recent **NVIDIA driver + CUDA stack** for GB10 on both nodes (match your `torch`/CUDA versions
  to it — see the torch-pin note in [`build/README.md`](build/README.md)).
- **Docker** installed and working on **both** nodes.
- A **high-speed interconnect** between the two nodes for tensor parallelism. We used a **RoCEv2**
  link over ConnectX-7 NICs on a dedicated `/30` subnet. You need:
  - the two node IPs on that link (→ `HEAD_IP` / `WORKER_IP`),
  - the NIC/interface names (Ethernet iface + IB HCA),
  - passwordless **SSH from the head to the worker** (`ssh -o BatchMode=yes <WORKER_IP>` must work —
    the build copy step and the drop-caches sidecar use it).
- A **Hugging Face token** with access to the **gated** `deepseek-ai/DeepSeek-V4-Flash` model
  (request access on the model page first). The weights are **~149 GB** and must be cached on
  **both** nodes.
- Disk headroom for the weights (~149 GB) and the built image on each node.

Quick sanity check on each node:

```bash
nvidia-smi                       # driver + GB10 visible
docker info >/dev/null && echo docker-ok
ssh -o BatchMode=yes <WORKER_IP> 'echo worker-ssh-ok'   # from the head node
```

---

## 1. Build the vLLM image (from PR #41834 head)

Stock vLLM (≤ 0.23.x) **does not load DeepSeek-V4-Flash on GB10** — the sm12x kernels and model
wiring live in an unmerged upstream PR. You build vLLM from that PR's head.

> **Why the PR head, not a merge:** PR **#41834** ("Add SM12x support for DeepSeek V4 Flash", from
> the `jasl` fork, branch `codex/ds4-sm120-min-enable`) is effectively a large multi-commit fork
> branch and reports `mergeable: dirty` — merging it onto current `main` always conflicts.
> The build therefore **fetches and checks out the PR head directly** instead of merging it:
>
> ```dockerfile
> RUN git fetch origin pull/41834/head:pr-41834 && git checkout pr-41834
> ```
>
> GitHub serves a fork's PR head under the base repo's `pull/<N>/head` ref, so this works even
> though the branch lives on a fork. See [`build/README.md`](build/README.md) for this and the
> torch-CPU-pin clobber (re-pin the CUDA `torch` *after* the wheel install).

Run the build on one node (typically the head). `build/build-and-copy.sh` builds the **sm_121a**
vLLM image and can copy it to both nodes in the same run:

```bash
cd build

./build-and-copy.sh \
  --apply-vllm-pr 41834 \
  --rebuild-vllm \
  --vllm-ref <pinned-vllm-commit> \
  --copy-to <WORKER_IP>
```

- `--apply-vllm-pr 41834` — applies the sm12x PR (fetch + checkout the PR head, per the note above).
- `--rebuild-vllm` — forces a fresh vLLM wheel build (don't reuse a stale cached wheel).
- `--vllm-ref <pinned-vllm-commit>` — **pin the exact commit you validate.** Newer upstream commits
  regressed for us more than once (see [issue #7](docs/known-issues.md)). The reference build used
  here is **vLLM `0.23.1rc1.dev407+g28fef2c70`** — record your actual head commit in
  [`CHANGES.md`](CHANGES.md). If you omit `--vllm-ref` it defaults to `main` (not recommended).
- The default image tag is **`vllm-gb10-dsv4`** (override with `-t <tag>`; it must match
  `VLLM_IMAGE` in your `.env`).
- The build target arch is `12.1a` (`TORCH_CUDA_ARCH_LIST=12.1a` / `FLASHINFER_CUDA_ARCH_LIST=12.1a`).

Build time: **~10 min** once dependency wheels are cached; a cold build is much longer.

> Useful variants:
> - Build only, copy later: omit `--copy-to`, then `./build-and-copy.sh --no-build --copy-to <WORKER_IP>`.
> - `-c/--copy-to` accepts a comma- or space-delimited list of hosts.
> - `./build-and-copy.sh --help` lists all options.

---

## 2. The image lands on both nodes

The `--copy-to <WORKER_IP>` flag above streams the built image to the worker (effectively
`docker save <tag> | ssh <WORKER_IP> docker load`). **TP=2 needs the identical image present on
each node.** Confirm on **both** the head and the worker:

```bash
docker images | grep vllm-gb10-dsv4            # on the head
ssh <WORKER_IP> 'docker images | grep vllm-gb10-dsv4'   # on the worker
```

Both must show the same tag (and ideally the same image ID).

You also need the **model weights (~149 GB) cached on both nodes**. The gated download uses your
`HF_TOKEN`; the launcher mounts your `~/.cache/huggingface` into the container, so pre-pulling the
weights into that cache on each node avoids paying the download at launch:

```bash
# on EACH node
export HF_TOKEN=<your_hf_token>
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash   # or let the first launch pull it
```

---

## 3. Configure `.env`

Copy the template and fill in your topology, token, and image tag. **Never commit `.env`** (it's in
`.gitignore`):

```bash
cp .env.example .env
$EDITOR .env
```

Minimum to set (from [`.env.example`](.env.example)):

```ini
# --- Cluster topology (your two DGX Spark nodes) ---
HEAD_IP=<HEAD_IP>            # e.g. 10.0.0.1 on the RoCEv2 link
WORKER_IP=<WORKER_IP>       # e.g. 10.0.0.2
MASTER_PORT=29501           # NCCL master port for the TP group

# --- Model source (gated model: required to pull weights) ---
HF_TOKEN=<your_hf_token>

# --- Image tag produced by build/ (must match the tag you built) ---
VLLM_IMAGE=vllm-gb10-dsv4
```

`launch/launch-cluster.sh` also reads a `.env` for cluster wiring — set the interface names and node
list it expects (auto-detected if omitted, but explicit is safer):

```ini
CLUSTER_NODES=<HEAD_IP>,<WORKER_IP>     # e.g. 10.0.0.1,10.0.0.2
ETH_IF=<eth_iface>                      # Ethernet interface name
IB_IF=<ib_iface>                        # RoCE/IB interface (e.g. ib0)
MASTER_PORT=29501
CONTAINER_NAME=vllm_node
CONTAINER_HF_TOKEN=<your_hf_token>      # CONTAINER_* vars are passed into the container as -e
```

---

## 4. Launch the TP=2 cluster + serve

The serve command itself lives in a template — pick the context window you want:

- [`launch/serve-1m.sh`](launch/serve-1m.sh) — **1 M context** (`--max-model-len 1000000`,
  `--max-num-seqs 6`, `--gpu-memory-utilization 0.83`). KV-bound: each sequence reserves a lot of KV,
  so concurrency is low. ~100 tok/s aggregate.
- [`launch/serve-256k.sh`](launch/serve-256k.sh) — **256 K context** (`--max-model-len 262144`,
  `--max-num-seqs 24`, `--gpu-memory-utilization 0.85`). Smaller per-seq KV → far more concurrency →
  ~150 tok/s aggregate. Prefer this if you serve many short/medium requests and don't need the full
  1 M window — and if you push **sustained saturation** (see GSP note in troubleshooting).

Both templates:
- require `HEAD_IP` / `WORKER_IP` (sourced from `.env`),
- start a **privileged drop-caches sidecar** on **both** nodes *during load* (on UMA the Linux page
  cache competes with the model/KV for the same pool; it's removed once `:8000` is ready), and
- run `vllm serve` with TP=2 and the DeepSeek-V4 parsers/MTP config.

Drive the multi-node cluster with `launch/launch-cluster.sh`, pointing it at the serve template
(`--launch-script` copies the script into the container on the head and runs it, with the worker
joined to the same TP group):

```bash
# from the repo root, with .env filled in
set -a; source .env; set +a       # export HEAD_IP / WORKER_IP / VLLM_IMAGE for the template

# 1 M context:
./launch/launch-cluster.sh \
  --nodes "$HEAD_IP,$WORKER_IP" \
  -t "$VLLM_IMAGE" \
  --launch-script launch/serve-1m.sh

# ...or 256 K context:
./launch/launch-cluster.sh \
  --nodes "$HEAD_IP,$WORKER_IP" \
  -t "$VLLM_IMAGE" \
  --launch-script launch/serve-256k.sh
```

Other actions: `./launch/launch-cluster.sh status`, `./launch/launch-cluster.sh stop`,
`./launch/launch-cluster.sh --check-config`, and `--help` for the full flag list. If interface names
aren't in `.env`, pass `--eth-if <iface> --ib-if <iface>`.

**Boot characteristics:** ~**80 s cold boot** with `--load-format instanttensor` (weight load + KV
alloc + CUDA-graph capture). The **first long prompt after boot** pays a one-time autotune/JIT cost
(up to ~+70 s TTFT) — **warm the engine before measuring anything.**

Once `:8000` answers, remove the drop-caches sidecars (the templates print the exact command):

```bash
docker rm -f dsv4-dropcaches
ssh "$WORKER_IP" docker rm -f dsv4-dropcaches
```

---

## 5. Verify

From the head node, confirm the model is serving with the expected context length:

```bash
curl -s http://localhost:8000/v1/models | jq '.data[0] | {id, max_model_len}'
# 1 M:    { "id": "deepseek-v4", "max_model_len": 1000000 }
# 256 K:  { "id": "deepseek-v4", "max_model_len": 262144 }
```

Then a real (warm-up) completion — discard the first response's timing, it pays the JIT/autotune cost:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "deepseek-v4",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 16
      }' | jq -r '.choices[0].message.content'
```

If `/v1/models` is empty or the curl is refused, the engine likely didn't come up — most often a
2-node NCCL init issue (see RoCE GID below) or a UMA free-memory shortfall.

---

## 6. Troubleshooting

Full detail and severities in [`docs/known-issues.md`](docs/known-issues.md). The ones that bite first:

- **2-node launch hangs at NCCL init after a reboot → RoCEv2 GID index.** After a node reboot the
  worker's RoCEv2 **GID index can come back wrong**; NCCL init then stalls (looks like a hang, not a
  clear error). **Check this first.** Re-bounce the interface address on the worker:
  ```bash
  # on the WORKER
  sudo ip addr del <worker_ip>/<prefix> dev <iface>
  sudo ip addr add <worker_ip>/<prefix> dev <iface>
  ```
  (See [`docs/hardware-bringup.md`](docs/hardware-bringup.md#the-post-reboot-gid-index-gotcha).)

- **Engine won't start / free-memory check refuses (UMA).** Unified memory is shared by everything on
  the box. Keep `--gpu-memory-utilization` **conservative** (0.83 @ 1 M, 0.85 @ 256 K — **not** 0.9 on
  a non-clean node), and make sure the **drop-caches sidecar ran on both nodes during load** (the page
  cache can push the check over the edge mid-load). See known-issues
  [#3 UMA-OOM](docs/known-issues.md#3-uma-oom-host-hang-under-parallel-benchmark-load).

- **GSP firmware hard-lock under sustained 1 M load.** Under zero-gap saturation at 1 M context the
  **whole host can hard-lock** (GSP firmware lockup; **driver-independent**). Mitigations: run **256 K**
  for sustained saturation (we did not reproduce the freeze there); if you must run 1 M continuously,
  put an external hard-power-cycle watchdog in front and long-soak it yourself (>12 h is unproven).
  See [#1](docs/known-issues.md#1-gsp-firmware-hard-lock-under-sustained-1-m-load--high).

- **MTP speculative-decoding crash (older vLLM).** On vLLM **0.22.x** with MTP under non-greedy
  concurrent traffic the EngineCore wedges and dies (connection-refused, no OOM/Xid). **Fixed** in the
  sm12x build referenced here (`0.23.1rc1.dev407`). If you're stuck on an older build, **disable MTP**
  (drop the `--speculative-config` from the serve template). See
  [#2](docs/known-issues.md#2-mtp-speculative-decoding-crash-on-older-vllm--high).

- **First requests after boot look absurdly slow (cold-start JIT/autotune).** One-time Triton JIT +
  flashinfer autotune cost. **Warm up and discard** before measuring. See
  [#6](docs/known-issues.md#6-cold-start-jit--autotune-skews-first-requests--low).

- **A newer vLLM build is slower or functionally broken.** Don't chase `main`; **pin the exact commit
  you validated** in [`CHANGES.md`](CHANGES.md) and re-build from it. See
  [#7](docs/known-issues.md#7-newer-upstream-commits-regressed--medium).

Also relevant: [#4 Marlin WNA16-MoE hangs](docs/known-issues.md#4-marlin-wna16-moe-hangs-on-gb10--medium)
(prefer the FP8 path in this recipe), [#5 NVFP4 @ 128 K OOMs KV](docs/known-issues.md#5-nvfp4--128-k-ooms-the-kv-cache--low),
and [#8 language drift on some prompt shapes](docs/known-issues.md#8-language-drift-on-some-prompt-shapes--low).
