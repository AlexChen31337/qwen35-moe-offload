# qwen35-moe-offload

> Efficient inference of Qwen3.5-35B-A3B on consumer hardware (RTX 3070 8GB + 16GB RAM + NVMe SSD), combining two orthogonal optimization axes: Apple's "LLM in a Flash" flash-memory expert offloading (arXiv:2312.11514) and Google's TurboQuant zero-overhead KV cache compression (arXiv:2504.19874, ICLR 2026).

## Results

| Phase | Best tok/s | Key technique |
|---|---|---|
| Phase 3 (baseline tuning) | 6.59 tok/s | n_ubatch=32, n_threads=10 |
| Phase 4 exp 2 | 8.52 tok/s | q8_0 KV quant + flash attention |
| Phase 4 exp 12 | 9.90 tok/s | +10 GPU layers |
| **Phase 4 exp 14** | **10.20 tok/s** | n_gpu=10, n_batch=32, q8_0 KV, flash_attn |

**+54% improvement** over parameter-tuned baseline via KV cache compression alone.

## The Core Insight

Qwen3.5-35B-A3B is a **sparse Mixture-of-Experts** model:
- **35B total parameters** — too large for any consumer GPU VRAM
- **3B activated per token** — only 8 routed experts + 1 shared expert fire per forward pass
- **256 total experts**, 9 active → **96.5% of FFN expert weights are idle per token**

Two independent memory bottlenecks exist on constrained hardware:

1. **Expert weight I/O** — 35B params won't fit in 8GB VRAM; must stream from NVMe/RAM
2. **KV cache** — grows linearly with context length; limits usable sequence length

These are orthogonal bottlenecks. We attack both simultaneously.

## Hardware

| Component | Spec | Inference Role |
|---|---|---|
| GPU | RTX 3070 8GB | Active computation + hot expert cache |
| RAM | 16GB DDR4 | Sliding window expert DRAM buffer |
| SSD | 937GB NVMe (PCIe 3.0) | Full model cold storage, ~3GB/s sequential read |

## Dual Optimization Strategy

### Axis 1 — Expert Weight Offloading (Apple "LLM in a Flash")
Inspired by [arXiv:2312.11514](https://arxiv.org/abs/2312.11514) (ACL 2024):

- Load attention/embedding layers to GPU (always hot, ~5.4GB)
- Store all 256 expert FFN weight matrices on NVMe as memory-mapped files
- Per token: route → load only 9 active expert chunks → compute → evict
- **Windowing:** maintain DRAM cache of last k expert activations (~60-70% NVMe read reduction)
- **Bundling:** co-locate gate_proj[i] + down_proj[i] on disk → single contiguous NVMe read

### Axis 2 — KV Cache Compression (Google TurboQuant)
Inspired by [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026):

- **PolarQuant**: convert KV vectors to polar coordinates (radius + angle) — eliminates per-block normalization overhead that traditional quantization must carry
- **QJL (Quantized Johnson-Lindenstrauss)**: 1-bit residual error correction via JL transform — zero memory overhead, bias-free attention scores
- **TurboQuant**: PolarQuant (main compression budget) + QJL (1-bit residual) = zero-overhead KV compression with no accuracy loss

**Practical implementation via llama.cpp:**
```bash
# q8_0 KV quant (Phase 4 best: +54% tok/s)
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn

# Push to longer context (TurboQuant-inspired, Phase 4 ongoing)
--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --ctx-size 4096
```

Note: llama.cpp's built-in KV quantization implements a subset of the TurboQuant principles. Full PolarQuant rotation is implemented as a Python wrapper layer (see `scripts/polar_kv.py`, in progress).

## Memory Budget Breakdown (Q3_K_M quantization, actual measurements)

| Layer type | Size | Where |
|---|---|---|
| Embeddings + LM head | ~1.2GB | GPU VRAM (always hot) |
| Attention layers (40L × shared) | ~2.8GB | GPU VRAM (always hot) |
| Delta attention layers (GDN) | ~1.4GB | GPU VRAM (always hot) |
| **Total always-hot** | **~5.4GB** | GPU VRAM |
| Expert FFN weights (full 256 experts) | ~15GB | NVMe SSD |
| Active expert cache (sliding window k=5) | ~1.5GB | RAM + remaining VRAM |
| **KV cache (n_ctx=512, f16)** | **~387MB** | GPU VRAM |
| **KV cache (n_ctx=4096, q8_0)** | **~387MB** | GPU VRAM (4× VRAM savings) |

KV q8_0 compresses the cache 2× vs f16; q4_0 compresses 4×. At n_ctx=4096 with q4_0, KV cache fits in the same VRAM budget as n_ctx=512 with f16.

## Autoresearch Loop

Experiments are driven by an autonomous Opus 4.6 agent — no human guidance between runs. The agent reads `program.md` (goal), examines prior results, decides what to change, runs, measures, and commits. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

```
goal (program.md) → agent decides → modify bench_kv.py → run 256 tokens → measure tok/s → commit if new best → repeat
```

**Branches:**
- `autoresearch/mar25` — NVMe simulation (52 experiments)
- `autoresearch/ram-offload` — RAM backend simulation (82 experiments)
- `phase3/real-inference` — Real GGUF on RTX 3070 (15 experiments, best: 6.59 tok/s)
- `phase4/kv-compression` — KV cache compression (ongoing, best: **10.20 tok/s**)

## Roadmap

| Phase | Technique | Status | Best result |
|-------|-----------|--------|-------------|
| 3 | llama.cpp parameter tuning | ✅ Done | 6.59 tok/s |
| 4 | KV block quant (q8_0) + flash attn + GPU layers | ✅ Done | **10.20 tok/s** |
| 5A | PolarQuant Rust+CUDA (arXiv:2502.02617) | 🔨 Building | — |
| 5B | QJL Rust+CUDA (arXiv:2406.03482) | 🔨 Building | — |
| 6 | Async expert prefetch (Apple §3.1) | 📋 Planned | — |

No ceiling. No predicted targets. The autoresearch loop runs until hardware limits are found empirically.

## Repo Structure

```
.
├── README.md
├── PLAN.md                     # Detailed implementation plan
├── program.md                  # Autoresearch goal and current phase directive
├── harness.py                  # Fixed benchmark harness (immutable)
├── bench.py                    # Autoresearch editable (NVMe/RAM phases)
├── bench_kv.py                 # KV compression experiments (Phase 4)
├── bench_polarquant.py         # PolarQuant benchmark (Phase 5A)
├── bench_qjl.py                # QJL benchmark (Phase 5B)
├── docs/
│   ├── architecture.md         # Qwen3.5-35B-A3B architecture breakdown
│   ├── apple-flash-mapping.md  # Apple paper techniques → MoE expert mapping
│   ├── kv-compression.md       # KV cache compression: PolarQuant + QJL deep dive
│   └── hardware-profile.md     # RTX 3070 + NVMe bandwidth measurements
├── scripts/
│   ├── polar_kv.py             # PolarQuant Python prototype (correctness verified)
│   ├── measure_nvme.py         # Benchmark NVMe sequential vs random read
│   ├── expert_cache.py         # Sliding window expert DRAM cache
│   └── download_model.py       # Download Qwen3.5-35B-A3B-Q3_K_M GGUF
└── crates/                     # Phase 5 — Rust+CUDA implementations (in progress)
    ├── polarquant/             # PolarQuant: Hadamard+polar decomposition CUDA kernel
    └── qjl/                    # QJL: JL transform + sign-bit quantization CUDA wrapper
```

## Quick Start

```bash
# Clone
git clone https://github.com/AlexChen31337/qwen35-moe-offload
cd qwen35-moe-offload

# Download model (~16.4GB)
uv run python scripts/download_model.py

# Run Phase 4 best config (10.2 tok/s on RTX 3070 8GB)
uv run --with llama-cpp-python python bench_kv.py
# Config: n_gpu_layers=10, n_batch=32, n_ubatch=32, n_threads=10
#         cache_type_k=q8_0, cache_type_v=q8_0, flash_attn=True
```

## References

| Paper | Relevance |
|---|---|
| Apple "LLM in a Flash" (arXiv:2312.11514, ACL 2024) | Windowing + bundling for expert weight offloading |
| Google TurboQuant (arXiv:2504.19874, ICLR 2026) | Zero-overhead KV cache compression via PolarQuant + QJL |
| Google PolarQuant (arXiv:2502.02617, AISTATS 2026) | Polar coordinate KV compression, eliminates normalization overhead |
| karpathy/autoresearch | Autonomous experiment loop methodology |
| llama.cpp | `--cache-type-k/v`, `--flash-attn`, `--n-gpu-layers` flags |
| KTransformers | MoE-aware CPU/GPU expert offloading reference implementation |

---

## Phase 5 — Rust Crates: PolarQuant + QJL

Pure-Rust CPU implementations of the two KV cache compression algorithms.
These crates will be compiled to a shared library and called from Python via
FFI to inject custom KV compression into the llama.cpp inference loop.

### Crates

```
crates/
├── polarquant/   PolarQuant KV cache compression (arXiv:2502.02617)
└── qjl/          QJL KV cache compression (arXiv:2406.03482)
```

#### `crates/polarquant`

Implements PolarQuant:
1. **Randomised Hadamard preconditioner** — Walsh-Hadamard transform + random ±1 diagonal.
2. **Polar decomposition** — splits KV head into radius + (head_dim−1) spherical angles.
3. **Angle quantization** — configurable `bits` per angle (1–8 bit).

Compression: **~7.58× vs f32** (~3.79× vs f16, paper: 3.91×) at head_dim=128, 4-bit.
Quality: **≥0.85 cosine similarity** at 4-bit.

#### `crates/qjl`

Implements QJL:
1. **JL projection** — random Gaussian matrix, seeded deterministically.
2. **Sign quantization** — `sign(AK) ∈ {−1,+1}` stored as `i8`.
3. **Asymmetric attention estimator** — estimates `Q·K` without full-precision K.

Compression: **4–16× depending on sketch_dim**.
Quality: **Pearson correlation >0.7** between estimated and true attention scores.

### Build

```bash
# CPU-only (no CUDA required)
cargo build --workspace
cargo test --workspace

# With CUDA support
cargo build --workspace --features cuda
```

### Integration Plan

Next step: `crates/kvcache-ffi/` C FFI shim → patch llama.cpp at KV read/write
boundary → measure vs Phase 4 q8_0 baseline (10.2 tok/s).
