# qwen35-moe-offload

> Efficient inference of Qwen3.5-35B-A3B on consumer hardware (RTX 3070 8GB + 16GB RAM + NVMe SSD) using flash-memory offloading techniques inspired by Apple's "LLM in a Flash" paper (arXiv:2312.11514).

## The Core Insight

Qwen3.5-35B-A3B is a **sparse Mixture-of-Experts** model:
- **35B total parameters** — too large for any consumer GPU VRAM
- **3B activated per token** — only 8 routed experts + 1 shared expert fire per forward pass
- **256 total experts**, 9 active → **96.5% of FFN expert weights are idle per token**

This is *exactly* the sparsity structure that Apple's flash offloading exploits. We don't need 35B in VRAM — we need ~3B active weights + the attention/embedding layers (~6GB), and we can stream expert weights on demand from NVMe.

## Hardware

| Component | Spec | Inference Role |
|---|---|---|
| GPU | RTX 3070 8GB | Active computation + hot expert cache |
| RAM | 16GB DDR4 | Sliding window expert DRAM buffer |
| SSD | 937GB NVMe (PCIe 3.0) | Full model cold storage, ~3GB/s sequential read |

## Memory Budget Breakdown (Q4_K_M quantization)

| Layer type | Size | Where |
|---|---|---|
| Embeddings + LM head | ~1.2GB | GPU VRAM (always hot) |
| Attention layers (40L × shared) | ~2.8GB | GPU VRAM (always hot) |
| Delta attention layers (GDN) | ~1.4GB | GPU VRAM (always hot) |
| **Total always-hot** | **~5.4GB** | GPU VRAM |
| Expert FFN weights (full 256 experts) | ~15GB | NVMe SSD |
| Active expert cache (sliding window k=5) | ~1.5GB | RAM + remaining VRAM |

**Target:** ≤ 7.5GB VRAM peak, ≤ 12GB RAM peak, ≥ 5 tok/s on RTX 3070.

## Techniques (layered, each experiment builds on prior)

### Phase 1 — Baseline Measurement
Measure naive inference speed and VRAM ceiling. Establish what breaks without offloading.

### Phase 2 — Expert-Aware Selective Loading
- Load attention/embedding layers to GPU (always hot)
- Store all 256 expert FFN weight matrices on NVMe as memory-mapped files
- Per token: route → load only 9 expert weight chunks → compute → evict

### Phase 3 — Sliding Window Expert Cache (Windowing)
- Maintain a DRAM cache of the last k=5 tokens' active expert weights
- Per new token: only load the *delta* experts not in the window
- Expected: ~60-70% reduction in NVMe reads vs naive per-token loading

### Phase 4 — Row-Column Bundling (Flash Paper Technique)
- In each MoE layer: co-locate gate_proj row[i] + down_proj col[i] on disk
- Read both in a single contiguous NVMe chunk (2x chunk size → 2x throughput)
- NVMe sequential read: ~3GB/s; random 4K read: ~50MB/s → bundling is critical

### Phase 5 — Sparsity Predictor (Low-Rank Expert Router Lookahead)
- Train a tiny low-rank predictor (similar to Apple paper §3.1) on top of attention output
- Predicts which experts will activate *before* loading them
- Enables async prefetch: start NVMe read while previous token is still computing
- Expected: hide most of the NVMe latency behind compute

### Phase 6 — Full Stack: Quantization + Offload + Async Prefetch
- Q4_K_M quantization reduces model from ~70GB (fp16) to ~18GB on disk
- Combine all above: windowing + bundling + async prefetch + quantized weights
- Target: ≥ 5 tok/s sustained generation

## Autoresearch-Inspired Experiment Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): each experiment is a fixed-budget (5 min) benchmark run with a single variable changed. An agent evaluates the result, decides keep/discard, and logs to `results/`.

```
experiment → benchmark (5 min) → evaluate metric → log result → next experiment
```

**Metric:** tokens/second on a fixed 512-token prompt → 256-token generation.  
**Secondary:** peak VRAM, peak RAM, NVMe read bytes per token.

## Repo Structure

```
.
├── README.md
├── PLAN.md                    # Detailed implementation plan (this file)
├── docs/
│   ├── architecture.md        # Qwen3.5-35B-A3B architecture breakdown
│   ├── hardware-profile.md    # RTX 3070 + NVMe bandwidth measurements
│   └── apple-flash-mapping.md # How Apple paper techniques map to MoE experts
├── scripts/
│   ├── measure_nvme.py        # Benchmark NVMe sequential vs random read speeds
│   ├── measure_baseline.py    # Baseline llama.cpp / transformers throughput
│   ├── build_expert_index.py  # Extract + index expert weights to disk
│   ├── expert_cache.py        # Sliding window expert DRAM cache
│   ├── bundled_loader.py      # Row-col bundled NVMe loader
│   ├── router_predictor.py    # Low-rank expert activation predictor
│   └── benchmark.py           # Unified benchmark runner (all phases)
├── experiments/
│   ├── baseline/              # Phase 1 results
│   ├── windowing/             # Phase 3 results
│   ├── bundling/              # Phase 4 results
│   └── full_stack/            # Phase 6 results
└── results/
    └── experiment_log.jsonl   # All experiment runs with metrics
```

## Quick Start

```bash
# 1. Measure your hardware
uv run python scripts/measure_nvme.py

# 2. Build expert index from GGUF model
uv run python scripts/build_expert_index.py --model path/to/Qwen3.5-35B-A3B.Q4_K_M.gguf

# 3. Run baseline (naive llama.cpp with full offload to RAM)
uv run python scripts/measure_baseline.py

# 4. Run full optimized stack
uv run python scripts/benchmark.py --phase full_stack --prompt "Your prompt here"
```

## Expected Results (projected)

| Configuration | tok/s | VRAM | RAM |
|---|---|---|---|
| Naive (transformers, RAM offload) | ~0.3 | 8GB | 16GB full | 
| llama.cpp default (mmap) | ~1.2 | 7.5GB | 8GB |
| Phase 2 (selective expert load) | ~2.0 | 6GB | 4GB |
| Phase 3 + windowing | ~3.5 | 6GB | 6GB |
| Phase 4 + bundling | ~4.5 | 6.5GB | 6GB |
| Phase 6 + async prefetch | **≥ 5.0** | 7.5GB | 8GB |

## References

- Apple "LLM in a Flash" (arXiv:2312.11514) — windowing + bundling techniques
- Qwen3.5-35B-A3B model card — architecture: 256 experts, 9 active, hidden_dim=2048
- llama.cpp `--no-mmap` + `--gpu-layers` flags for layer-level GPU/CPU split
- KTransformers — MoE-aware CPU/GPU expert offloading (key related work)
