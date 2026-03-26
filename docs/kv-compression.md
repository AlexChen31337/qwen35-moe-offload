# KV Cache Compression — Research Overview

This document covers the KV cache compression techniques implemented and planned for this project.

## Why KV Cache is a Bottleneck

The KV cache stores key and value vectors for every past token in every attention layer. Size:
```
KV cache bytes = 2 × n_layers × n_heads × head_dim × n_ctx × dtype_bytes
```

For Qwen3.5-35B-A3B at n_ctx=4096, fp16:
- 40 layers × 40 heads × 128 head_dim × 4096 tokens × 2 bytes × 2 (K+V) = **~3.3GB**

At n_ctx=512 this is ~410MB — manageable. At 4096+ it competes with expert weights for VRAM.

## Compression Techniques

### Phase 4 (Current) — llama.cpp Built-in Block Quantization
- `--cache-type-k q8_0 --cache-type-v q8_0`: 2× compression
- `--cache-type-k q4_0 --cache-type-v q4_0`: 4× compression
- **Overhead:** +1-2 bits/value for stored scale and zero-point per block
- **Status:** ✅ Active. Best result: 10.20 tok/s (exp 14)

---

### Phase 5 Track A — PolarQuant (arXiv:2502.02617, AISTATS 2026)
**Authors:** Insu Han (KAIST), Praneeth Kacham, Vahab Mirrokni, Amir Zandieh (Google Research), Amin Karbasi (Yale)

**Algorithm:**
1. Apply random Hadamard preconditioner H to KV vector: `v = H @ x`
2. After preconditioning, angles in polar representation concentrate near π/2 with variance O(1/d)
3. Recursive polar decomposition: `(x₁,...,xₙ) → (r, θ₁,...,θₙ₋₁)` where r = ‖x‖₂
4. Quantize only angles θᵢ on uniform grid [0, π] — no per-block normalization needed
5. Store: r in fp16, angles in 3-4 bits

**Why zero overhead:** Standard block quantization must store scale + zero-point in full precision per block (+1-2 bits/value). PolarQuant's angle distribution is analytically known after preconditioning, so no normalization constants needed.

**Results (Python prototype, validated):**
- 4-bit: **3.91× compression**, cosine similarity 0.89
- 5-bit: 3.15× compression, cosine similarity 0.97
- Angle mean after preconditioning: 1.569 ≈ π/2 ✅ (confirms theory)

**Planned implementation:** Rust crate `crates/polarquant/`
- CUDA kernel for Hadamard transform via `cudarc`
- FFI bridge to llama.cpp via `#[no_mangle] extern "C"`
- New KV cache type: `--cache-type-k polar_q4`

---

### Phase 5 Track B — QJL (arXiv:2406.03482)
**Authors:** Same group — Amir Zandieh (Google Research) et al.
**Code:** https://github.com/amirzandieh/QJL ✅ (CUDA kernel exists)

**Algorithm:**
1. Apply Johnson-Lindenstrauss transform to KV vectors: dimensionality reduction preserving inner products
2. Sign-bit quantization: each element → +1 or -1 (1 bit)
3. Asymmetric estimator for attention scores: high-precision query × low-precision key
4. Zero memory overhead: no quantization constants stored

**Results (Python prototype):**
- sketch=512: **3.88× compression**, attention correlation 0.78, **0.34ms GPU** (65× faster than PolarQuant Python)
- sketch=256: 7.53× compression, attention correlation 0.67
- Custom CUDA kernels: quantize + inner product estimator

**Planned implementation:** Rust crate `crates/qjl/`
- Wrap existing CUDA kernels from github.com/amirzandieh/QJL
- Safe Rust FFI boundary (borrow checker prevents KV cache lifecycle bugs)
- New KV cache type: `--cache-type-k qjl_3bit`

---

## Comparison

| Method | Compression | Quality | GPU Speed | Status |
|--------|------------|---------|-----------|--------|
| llama.cpp q8_0 | 2× | best | native | ✅ Phase 4 |
| llama.cpp q4_0 | 4× | good | native | ✅ Phase 4 |
| PolarQuant 4-bit | 3.91× | 0.89 cos sim | 33ms (Python) | 🔨 Phase 5 |
| PolarQuant 5-bit | 3.15× | 0.97 cos sim | 33ms (Python) | 🔨 Phase 5 |
| **QJL sketch=512** | **3.88×** | 0.78 attn corr | **0.34ms CUDA** | 🔨 Phase 5 |
| QJL sketch=256 | 7.53× | 0.67 attn corr | 0.32ms CUDA | 🔨 Phase 5 |

## Why Rust for Phase 5

C++ alternative risks:
- Use-after-free in KV cache tensor lifecycle (silent corruption, hard to debug)
- Undefined behavior in template metaprogramming

Rust advantages:
- Borrow checker catches KV cache lifecycle bugs at compile time
- Zero-cost abstractions — identical machine code to C for inner loops
- `cudarc` crate (Hugging Face) for safe CUDA kernel dispatch
- `#[no_mangle] extern "C"` for clean llama.cpp FFI
- Publishable as standalone crates (`polarquant`, `qjl-rs`)
- Aligns with ClawChain/EvoClaw Rust-native infrastructure stack

## Context Length Impact

At Phase 4 best config (q8_0, n_ctx=512, VRAM=1387MB):

| n_ctx | q8_0 KV VRAM | polar_q4 KV VRAM | headroom |
|-------|-------------|-----------------|---------|
| 512 | ~200MB | ~51MB | saves 149MB |
| 4096 | ~1.6GB | ~410MB | saves 1.2GB |
| 16384 | ~6.4GB | ~1.6GB | fits in 8GB |
| 32768 | OOM | ~3.3GB | potentially viable |

**The autoresearch loop will discover the actual ceiling — these projections are illustrative only.**

## References

| Paper | arXiv | Venue | Code |
|-------|-------|-------|------|
| PolarQuant | 2502.02617 | AISTATS 2026 | — |
| QJL | 2406.03482 | — | github.com/amirzandieh/QJL |
| TurboQuant | 2504.19874 | ICLR 2026 | — |
| Apple Flash | 2312.11514 | ACL 2024 | — |
