# VeloceIO: Software-Defined Multipath Memory Access Engine

Transparently break the single-link PCIe bottleneck in multi-GPU nodes. VeloceIO multiplexes Host-to-Device memory transfers across idle peer-GPU PCIe lanes, using the high-speed NVLink backplane for final assembly.

**Result:** Up to **245 GB/s** effective host-to-device bandwidth on standard 8x GPU nodes, cutting model cold starts and long-context (vLLM/SGLang) KV cache prefill loading times in half.

```
                  [ Host System Memory (DRAM) ]
                  /      |            |      \
        (PCIe 0) /       | (PCIe 1)   | ...   \ (PCIe N)
                v        v            v        v
            [GPU 0]   [GPU 1]      [GPU 2]  [GPU N]  <-- Peer Relay Ports
               ^         |            |        /
               |         | (NVLink)   | (NVLink)
                \________v____________v_______/
                         |
                [ Target GPU (GPU 0) ]

```

---

> **Drop-in Acceleration:** VeloceIO operates entirely in user-space via `LD_PRELOAD`. It requires **zero code changes** to your application, zero kernel driver modifications, and automatically passes through small copies or unsupported streams.

---

## The Hidden Bottleneck

In multi-GPU clusters (like an NVIDIA HGX H100/H20 8-GPU chassis), individual host-to-device memory transfers (such as swapping model weights or fetching offloaded KV caches) are bound to the physical PCIe link of the target GPU.

At PCIe Gen 5 x16, this limits real-world transfers to ~53 GB/s. While the target GPU's PCIe lane is completely choked, **the other 7 PCIe links sit completely idle**.

VeloceIO treats the node as a software-defined mesh network. It intercepts memory copies, slices them into micro-chunks, distributes them across all available peer PCIe links in parallel, and streams them to the destination GPU using the ultra-fast intra-node NVLink fabric (900 GB/s bidirectional).

---

## Performance Index

*Benchmarks evaluated on an 8 × NVIDIA H20 Node (PCIe Gen 5 x16, NVLink 4.0, Dual AMD EPYC 9654).*

| Workload / Metric | Native CUDA | VeloceIO Optimized | Performance Delta |
| --- | --- | --- | --- |
| **Peak Host-to-Device Throughput** | 53 GB/s | 245 GB/s | **4.62x Bandwidth Boost** |
| **P99 TTFT (64K Context Fetch)** | 2.62 seconds | 1.10 seconds | **2.38x Latency Reduction** |
| **P99 TTFT (128K Context Fetch)** | 5.10 seconds | 2.15 seconds | **2.37x Latency Reduction** |
| **Model Wake-Up (Qwen-32B, 64GB)** | 2.10 seconds | 0.88 seconds | **2.38x Faster Cold Start** |
| **Co-located Decode Kernel Tax** | 14.8% delay | <2.7% delay | **Congestion-Isolated** |

---

## Key Features

* **Zero-Code Integration:** Installed via a single environment variable injection (`LD_PRELOAD=libveloce.so`). Compatible with PyTorch, vLLM, SGLang, and TensorRT-LLM.
* **Dual-Pipeline Overlay Engine:** Uses lock-free double-buffered circular scratchpads (16MB slices) to completely hide the NVLink hop behind the slower PCIe transfer step.
* **Low-Occupancy PTX Assembly Signaling:** Bypasses heavy CPU context-switching loops. Launches a 1-thread placeholder sync kernel directly on the user's stream that polls the release signature using cache-bypassing global memory loads (`ld.global.cg.volatile`) with an exponential backoff `nanosleep`.
* **Congestion-Aware Active Yielding:** Dynamically balances traffic using outstanding queue depth. If a peer GPU experiences heavy inference decode execution load, the path selector pulls traffic away to protect production P99 latency windows.
* **GDS Alignment Engine:** Realignment kernel that intercepts unaligned GPUDirect Storage chunks (e.g., misaligned Safetensors blocks) and straightens data configurations directly in VRAM over 64-bit word-coalesced loops.

---

## Architecture Blueprint

```
       =========================================================
      |     vLLM / PyTorch Application Space                    |
       =========================================================
                                  |
               (Intercept Calls via LD_PRELOAD wrapper)
                                  |
                                  v
       =========================================================
      |     VeloceIO Runtime Layer (libveloce.so)              |
       =========================================================
              |                                         |
              | (Intercepts Copy)                       | (Spins up)
              v                                         v
   ==============================            ==============================
  |   Transfer Task Interceptor  |          |   PTX Async Spin Barrier     |
  |  - Intercepts MemcpyAsync    |          |  - Target stream placeholder |
  |  - Bypasses CUDA Graphs      |          |  - Cache-blind volatile loop |
   ==============================            ==============================
              |                                         |
              | (Dispatches fragments)                  | (Flips signal flag)
              v                                         v
   ==============================            ==============================
  |   MPMC Bounded Queue Ring    | --------> |   FlagPool Slot Tracker     |
  |  - LRD Work-Stealing Workers |           |  - Pre-allocated HBM slabs  |
  |  - Dual-Pipeline P2P copies  |           |  - Clean lock-free reuse    |
   ==============================            ==============================

```

---

## Quick Start

### 1. Prerequisites

* **Operating System:** Linux (Ubuntu 20.04+ or Rocky Linux 8+)
* **Toolchain:** GCC 11+, CMake 3.18+, CUDA Toolkit 11.8+
* **Hardware Profile:** Multi-GPU node featuring high-speed peer-to-peer interconnects (NVLink or equivalent unified backplanes).

### 2. Building the Project

Clone the repository and compile the optimized production release binary:

```bash
git clone https://github.com/MohdAatifSiddi/VeloceIO.git
cd VeloceIO

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

```

This generates:

* `libveloce.so`: The core shared runtime interception library.
* `veloce_test`: The high-concurrency multi-threaded verification stress engine.

### 3. Running Verification Tests

Execute the local multi-threaded test harness to validate memory layout and data path integrity across your system nodes:

```bash
./veloce_test

```

### 4. Direct Production Deployment

Inject the runtime optimization layer straight into your existing container or environment serving pipelines without changing a line of source code:

```bash
# Inject the interposition architecture layer into execution runtimes
export LD_PRELOAD=/path/to/build/libveloce.so

# Start your inference server normally
vllm serve --model deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8

```

---

## Structural Safeguards & Fallbacks

To ensure absolute safety in production deployments, VeloceIO includes a zero-overhead defensive fallback tier:

* **Bypass Threshold:** Any memory copy smaller than **32MB** bypasses VeloceIO's internal daemon entirely. This prevents fine-grained, low-latency operations from incurring micro-task scheduling overhead.
* **CUDA Graph Protection:** The interception layer queries stream capture flags at entry. If a framework is actively capturing a CUDA Graph, VeloceIO steps aside and transparently routes the call through the native CUDA driver path.
* **Graceful Degradation:** If the pre-allocated FlagPool is exhausted or peer memory access rules are disabled on anomalous node configurations, the transaction falls back to a clean single-path `cudaMemcpyAsync` routine, protecting the workload from unexpected crashes.

---

## License

VeloceIO is distributed under the Apache 2.0 License. See `LICENSE` for details.
