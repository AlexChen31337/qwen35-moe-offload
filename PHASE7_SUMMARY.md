# Phase 7 Summary: PolarQuant + QJL End-to-End Benchmarks

## Mission
Test custom KV cache compression algorithms (PolarQuant from arXiv:2502.02617, QJL from arXiv:2406.03482) against llama.cpp's native quantization for Qwen3.5-35B-A3B on RTX 3070 8GB.

## All-Time Best (unchanged)
**12.114 tok/s** (Phase 6, exp 117: n_gpu=16, batch=256/64, q8_0, n_ctx=512, n_threads=12, gen=256)

## Key Findings

### 1. Rust Implementation Throughput
Both PolarQuant and QJL Rust crates compile and pass all tests (27/27).

| Algorithm | Throughput | Per-Token Overhead | Max tok/s | CPU Budget Used |
|-----------|-----------|-------------------|-----------|----------------|
| PolarQuant 4-bit (Rust) | 150K heads/s | 3.24 ms | 11.657 | 3.9% |
| PolarQuant 6-bit (Rust) | 161K heads/s | 3.10 ms | 11.676 | 3.8% |
| QJL sketch=64 (Rust) | 27K heads/s | 10.27 ms | 10.774 | 12.4% |
| QJL sketch=128 (Rust) | 12K heads/s | 19.53 ms | 9.797 | 23.7% |
| **Uniform quant 4-bit (NumPy)** | **vectorised** | **0.13 ms** | **12.095** | **0.2%** |

### 2. ⚡ DISCOVERY: Simple Uniform Quantization Beats PolarQuant

| Bit Width | PolarQuant cos_sim | Uniform cos_sim | Winner | Margin |
|-----------|-------------------|-----------------|--------|--------|
| 2-bit | 0.264 | 0.800 | **Uniform** | +0.536 |
| 3-bit | 0.610 | 0.803 | **Uniform** | +0.193 |
| 4-bit | 0.891 | 0.917 | **Uniform** | +0.026 |
| 5-bit | 0.973 | 0.979 | **Uniform** | +0.006 |
| 6-bit | 0.993 | 0.995 | **Uniform** | +0.002 |
| 8-bit | 0.9996 | 0.9997 | **Uniform** | +0.0001 |

**Why?** PolarQuant's spherical coordinate conversion assumes structured (non-Gaussian) distributions. For approximately i.i.d. Gaussian KV cache values, the polar conversion introduces unnecessary error. Simple radius-preserving uniform quantization on the unit sphere is more direct and loses less information.

### 3. Quality Degradation with Head Dimension (PolarQuant)
PQ 4-bit quality drops as head_dim increases (spherical coordinate curse):
- d=32: cos_sim=0.944
- d=128: cos_sim=0.897
- d=256: cos_sim=0.816
- d=512: cos_sim=0.677

### 4. q8_0 + PolarQuant Stacking: Zero Additional Loss
Applying PolarQuant to q8_0-dequantized values produces **identical** quality to direct PolarQuant. The q8_0 quantization noise doesn't compound with PolarQuant.

### 5. QJL Attention Speedup: Non-Existent
QJL sketch-based attention is **250-16,000x SLOWER** than full-precision matmul, even in Rust. The Gaussian matrix generation + projection cost vastly exceeds the benefit of smaller vectors. QJL only becomes viable with CUDA kernel acceleration at extreme context lengths (>100K).

### 6. Hybrid Offloading Analysis
With 10GB RAM budget for KV offload:

| Config | Total Context | Speed | Quality |
|--------|--------------|-------|---------|
| iq4_nl VRAM + Uniform 4-bit RAM | ~310K tokens | 12.09 tok/s | 0.917 cos_sim on offloaded tokens |
| iq4_nl VRAM + PQ6 Rust RAM | ~210K tokens | 11.67 tok/s | 0.993 cos_sim on offloaded tokens |
| iq4_nl VRAM + QJL64 Rust RAM | ~328K tokens | 10.77 tok/s | Pearson r=0.54 (lower quality) |

**Blocker:** llama.cpp provides no API to extract/inject individual KV cache entries.

## Files Created
- `scripts/polar_kv.py` — Python reference implementation of PolarQuant + QJL
- `bench_polarquant.py` — PolarQuant Python benchmark
- `bench_qjl.py` — QJL Python benchmark
- `bench_hybrid.py` — Hybrid offloading analysis
- `bench_rust_ffi.py` — Rust/Python integration benchmarks
- `crates/polarquant/src/bin/bench_throughput.rs` — Rust PQ throughput bench
- `crates/polarquant/src/bin/quality_sweep.rs` — Rust PQ quality analysis
- `crates/qjl/src/bin/bench_throughput.rs` — Rust QJL throughput bench
- `results_phase7.tsv` — 31 experiments logged

## Conclusions

1. **For inference speed: llama.cpp's native q8_0/iq4_nl remains king.** Zero CPU overhead, integrated with CUDA attention kernels. Phase 6's 12.114 tok/s is the ceiling without deeper llama.cpp modifications.

2. **For KV offloading: vectorised uniform quantization wins.** 0.13ms overhead (0.2% of budget), better quality than PolarQuant, trivial to implement. NumPy's SIMD/BLAS makes it faster than Rust PolarQuant's per-head processing.

3. **PolarQuant is not suitable for Gaussian-distributed KV values.** The spherical coordinate transform adds error rather than reducing it. The paper's gains likely come from structured (non-Gaussian) activations in specific architectures.

4. **QJL is a research tool, not a practical compression scheme.** The Gaussian matrix generation is the bottleneck. Without CUDA kernels, it's thousands of times slower than dense matmul.

5. **The real bottleneck is llama.cpp integration**, not compression speed. Building a KV cache offloading system requires patching llama.cpp's KV management API — a significant engineering effort beyond compression algorithm benchmarking.

## Phase 7 Verdict
No new speed record (12.114 tok/s holds). But we now have:
- Complete understanding of PolarQuant/QJL capabilities and limitations
- Discovery that simple uniform quantization dominates for Gaussian KV distributions
- Theoretical framework for 310K+ context via RAM offloading (~12.09 tok/s)
- Comprehensive Rust crates with 27 passing tests
- Clear roadmap: the compression algorithms work; llama.cpp API is the gap
