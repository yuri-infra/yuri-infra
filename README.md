# Hongwei Zhu · 朱鸿伟

**M.Eng. Student in Electronic Information, Fudan University (复旦大学)**  
**Focus:** AI Infra · CUDA C++ & Triton Kernels · PyTorch / TorchInductor Compiler · LLM Serving · Long-Context Distributed Training

Building reproducible LLM systems and GPU kernels from scratch on consumer Blackwell hardware (`NVIDIA RTX 5060 8GB`, measured `312.6 GB/s` sustained 256MB D2D ceiling), verified against real `Qwen2.5-0.5B-Instruct` weights with 5-round ABA median benchmarks, PTX disassembly checks, and end-to-end correctness tests.

> **Open to AI Infra R&D Internships** (Tencent TEG/WXG, ByteDance Seed/AML, etc.) · Available starting **Dec 2026**, 4–5 days/week for 3–6 months.

---

## Core AI Infra Projects

| Repository | Architecture & Engineering Scope | Verified Benchmark Highlights (`RTX 5060 8GB`, 5-Round ABA Median) |
| :--- | :--- | :--- |
| **[`fused-cuda-triton-inductor`](https://github.com/yuri-infra/fused-cuda-triton-inductor)**<br><sub>CUDA C++ · Triton · TorchInductor</sub> | Native `.cu` `FusedAddRMSNorm` (`uint4` 128-bit loads, register residency, Warp shuffle reduction; `cuobjdump`-verified `80 ld.v4` / `140 shfl.bfly`; `TORCH_LIBRARY_IMPL` → `c10::Dispatcher`); dual-grid single-launch Triton GQA RoPE & SwiGLU; cache-isolated `TorchInductor` Post-Grad 2-stage Pass (topological sort + dynamic ancestor bitmask cycle guard); zero-`.item()` 128K chunked Triton `Linear+CE`. | • **512MB HBM regime:** `311.0 GB/s` (**99.5%** of `312.6 GB/s` sustained D2D ceiling)<br>• **M=1 Decode (launch-bound):** `13.8 μs` (**1.07x vs `torch.compile`** / `10.1x vs Eager`)<br>• **GQA RoPE (16Q/4KV, B=1 Decode):** `11.6 μs` (**1.40x vs `compile`**); Prefill `1.02x vs compile`<br>• **8L LLaMA-3 (B=8 Decode):** **1.03x vs `compile`** (`2.40x vs Eager`), zero `FXGraphCache` cross-talk<br>• **128K `Linear+CE`:** **-74.2%** intermediate activations vs `compile` (`0.99x` latency), FP16 grad underflow `>96% → ~11%` |
| **[`qwen-paged-serving`](https://github.com/yuri-infra/qwen-paged-serving)**<br><sub>Triton PagedAttention · Radix-Tree · SSE</sub> | End-to-end serving engine for 24-layer `Qwen2.5-0.5B-Instruct` (BF16, 14Q/2KV GQA): 7:1 GQA `BLOCK_G=16` Tensor Core (`tl.dot`) + adaptive Split-KV Triton PagedAttention, cross-project `.cu`/Triton fusion, `O(log N)` min-heap edge-splitting `Radix-Tree` + CoW lock downgrade, Decode-first `Chunked-Prefill` + event-safe zero-leak LIFO pinned Swap, `asyncio (epoll)` GPU-thread-decoupled SSE gateway. | • **Paged Decode (`B=16, S=1024`):** **1.76x** vs `.flatten()+SDPA` (`115.1 → 65.4 μs`), scratch copy `8.22 → 0.22 MB`<br>• **Cross-project fusion:** single-step launches `814 → 400` (`-50.9%`), prefix-hit TTFT **-23.0%** vs Eager (`19.9 → 15.3 ms`, analyzed against `2.94 ms` DRAM bound)<br>• **Scheduling & Prefix Reuse:** `1056`-tok hit TTFT **-72.0%** (`54.6 → 15.3 ms` vs contiguous SDPA); burst ITL P99 **-90.5%** (`289.0 → 27.5 ms` vs atomic prefill)<br>• **Concurrency (`t=0` empty-tree cold start, 16K budget):** 24-way multi-turn **7.37x** (`114 → 840 tok/s`, `83.8%` hit rate), KV internal fragmentation `36.0% → 0.63%`; SSE event-loop jitter `41.7 → 2.5 ms` |
| **[`qwen-32k-longctx-train`](https://github.com/yuri-infra/qwen-32k-longctx-train)**<br><sub>Triton FA2 · ZigZag Ring · SAC · Offload</sub> | Single-8GB-GPU 32K long-context training & sequence parallelism for 24-layer `Qwen2.5-0.5B` (152K vocab): zero-`atomic_add` register-reduced GQA Triton FlashAttention-2 fwd/bwd, multi-process even/odd interleaved P2P `ZigZag Ring Attention`, intra-layer SAC, full-sequence `S-1` in-place `dw_f32.addmm_` 152K chunked CE, dual-stream pinned async activation offload + `HybridCPUAdamW`. | • **ZigZag Ring Attention:** `P=4` causal load imbalance **6.99x → 1.00x** (`world_size=2` bwd max diff `8.58e-6` vs global FA2)<br>• **Intra-layer SAC (`S=4096`):** **-80.6%** resident activations vs SDPA Eager (`3813 → 739 MB`) at **15.13 TFLOPS** (beats `14.87 TFLOPS` full recompute)<br>• **152K Chunked CE (`S=16K`):** bwd workspace cut to **520 MB** (**-93%+** vs unchunked) with overlapped last-layer H2D prefetch<br>• **Dual-stream Offload & 2-Step Steady State:** masks `52.5 ms` raw PCIe (`1225.6 → 1170.3 ms`, zero-tamper verified); 8GB single-GPU 2-step steady-state context **1K → 32K** (**32x** vs GPU AdamW) |

---

## Technical Stack

- **Languages & GPU Kernels:** Python, C / C++, CUDA C++ (`uint4` vectorized memory access, Warp shuffle reduction, PTX inspection), OpenAI Triton (`tl.dot` Tensor Core tiling)
- **Compiler, Serving & Training:** PyTorch 2.x (`FX` / `Core ATen` / `TorchInductor` / `c10::Dispatcher`), PagedAttention, Radix-Tree, Chunked-Prefill, FlashAttention-2, ZigZag Ring Attention, Selective Activation Checkpointing (SAC) & Async Offload
- **Systems & Performance Profiling:** GPU memory hierarchy & Roofline analysis, CUDA multi-stream / event synchronization, Pinned Memory DMA, `torch.distributed` P2P, `asyncio (epoll)`, `cuobjdump`, `torch.profiler`

---

## Academic & Industry Background

- **Publication & Academic Service:** Sole Corresponding Author & Invited Reviewer, **IEEE/RSJ IROS 2025** (robotics mechanism design and mechatronic system validation). B.Eng. Rank **1 / 59** (GPA `3.92 / 4.00`, Tianjin Municipal Government Scholarship), admitted to Fudan University via graduate recommendation.
- **Industry Internship:** Embedded Software Intern, **Zhejiang Uniview Technologies (宇视科技)** (6 months, 2025.11 – 2026.05) — embedded C module development and peripheral bring-up for intelligent traffic vision and thermal imaging terminals.

---

## Contact

- **Email:** [zhuhw26@m.fudan.edu.cn](mailto:zhuhw26@m.fudan.edu.cn)
- **Location:** Shanghai, China (Fudan University)
