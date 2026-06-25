# Hardware bring-up (2× DGX Spark / GB10)

## Prerequisites

- **2× NVIDIA DGX Spark (GB10)** — Grace-Blackwell, compute capability **sm_121**, ~120 GB unified
  memory each.
- A **high-speed interconnect** between the nodes for tensor parallelism — we used a **RoCEv2**
  link over ConnectX-7 NICs (a dedicated /30 subnet).
- Recent NVIDIA driver + CUDA stack for GB10; Docker on both nodes.
- The vLLM image from [`../build/`](../build/README.md) present on **both** nodes.
- The model weights (~149 GB) cached on both nodes (gated download — set `HF_TOKEN`).

## Unified memory (UMA) — the thing that bites first

GB10 has **no separate VRAM**. Weights, KV cache, CUDA graphs, *and every other process on the box*
share the same ~120 GB pool. Consequences:

- **`--gpu-memory-utilization` must be conservative.** If anything else runs on the node (other
  services, sidecars), they eat into the same budget. We run **0.83 at 1 M / 0.85 at 256 K**, not
  the 0.9 you might use on a clean discrete GPU. vLLM's own free-memory check will refuse to start
  if the budget is short.
- **The page cache competes for the pool.** Loading ~149 GB of weights fills the Linux page cache,
  which can push the free-memory check over the edge mid-load. Fix: a **privileged drop-caches
  sidecar** (`sync; echo 3 > /proc/sys/vm/drop_caches` in a loop) on **both** nodes during load,
  removed once `:8000` is ready. Both `launch/` scripts do this.

## RoCE / NCCL for TP=2

Tensor-parallel across two nodes runs NCCL over the RoCEv2 link. Relevant NCCL env (set on each
node): `NCCL_IB_HCA=<your_hca>`, `NCCL_SOCKET_IFNAME=<iface>`, `NCCL_IB_GID_INDEX=3`,
`NCCL_NET=IB`, `MASTER_ADDR=$HEAD_IP`.

### The post-reboot GID-index gotcha

After a node reboot, the worker's **RoCEv2 GID index can come back wrong**, and NCCL init then
fails on the 2-node launch (looks like a hang, not an obvious error). The fix is to re-bounce the
interface address on the worker so the right RoCEv2 GID is selected:

```bash
# on the worker, if 2-node launch hangs at NCCL init after a reboot
sudo ip addr del <worker_ip>/<prefix> dev <iface>
sudo ip addr add <worker_ip>/<prefix> dev <iface>
```

When a 2-node launch fails post-reboot, **check this first** — it is far more often the GID index
than anything in the serving config.

## Boot characteristics

- **~80 s cold boot** with `--load-format instanttensor` (weight load + KV alloc + CUDA-graph
  capture). KV pool for 1 M context is ~2.0 M tokens (≈ 2× concurrency at full context).
- The **first long prompt after boot pays a one-time autotune/JIT cost** (flashinfer autotune +
  Triton JIT), up to ~+70 s TTFT. Warm the engine before you measure anything
  (see [`benchmark-results.md`](benchmark-results.md)).

## Validating a node is healthy before serving

```bash
curl -s http://localhost:8000/v1/models | jq '.data[0] | {id, max_model_len}'
# expect: { "id": "deepseek-v4", "max_model_len": 1000000 }
```
