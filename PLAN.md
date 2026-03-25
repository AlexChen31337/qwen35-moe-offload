# Implementation Plan — Qwen3.5-35B-A3B Flash Offload on RTX 3070

**Hardware target:** Dell XPS, RTX 3070 8GB, 16GB RAM, NVMe SSD (~3GB/s seq read)  
**Goal:** ≥ 5 tok/s sustained generation on a 35B MoE model with only 8GB VRAM

---

## Phase 0: Environment Setup (Day 1, ~2h)

### 0.1 Measure NVMe throughput
```bash
# Sequential read (large chunks — what we want)
dd if=/dev/nvme0n1 of=/dev/null bs=1G count=4 iflag=direct 2>&1

# 4K random read (the bad case we want to avoid)
fio --name=rand4k --rw=randread --bs=4k --size=1G --numjobs=4 --iodepth=32 --runtime=30 --time_based --direct=1
```
Expected: ~3 GB/s sequential, ~50 MB/s random 4K. Document in `docs/hardware-profile.md`.

### 0.2 Install dependencies
```bash
uv add transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
uv add numpy safetensors huggingface_hub
```

### 0.3 Download model (Q4_K_M GGUF)
```bash
# ~18GB on disk — fits on NVMe, NOT in VRAM
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GGUF \
  --include "Qwen3.5-35B-A3B-Q4_K_M*.gguf" \
  --local-dir ./models/
```

---

## Phase 1: Baseline Measurement (Day 1, ~2h)

### What to measure
Run the model in three naive configurations and record tok/s, VRAM, RAM:

1. **HuggingFace Transformers + CPU offload** (`device_map="auto"`)
2. **llama.cpp default** (`-ngl 0` = pure CPU, all layers in RAM)
3. **llama.cpp with GPU layers** (`-ngl 20` = push as many layers to GPU as VRAM allows)

### Script: `scripts/measure_baseline.py`
```python
import subprocess, json, time

CONFIGS = [
    {"name": "llama_cpu", "cmd": ["llama-cli", "-m", "models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
                                   "-ngl", "0", "-p", PROMPT, "-n", "256", "--log-disable"]},
    {"name": "llama_gpu20", "cmd": ["llama-cli", "-m", "models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
                                    "-ngl", "20", "-p", PROMPT, "-n", "256", "--log-disable"]},
]
# measure tok/s from llama.cpp output, VRAM from nvidia-smi polling
```

### Expected findings
- llama.cpp CPU-only: ~0.8-1.5 tok/s (bottlenecked by RAM bandwidth)
- llama.cpp GPU 20L: ~2-3 tok/s (attention on GPU, experts on CPU)
- This is the baseline to beat.

---

## Phase 2: Expert Weight Index (Day 2, ~4h)

### Core idea
Extract all 256 expert FFN weight matrices from the GGUF and store them as individually-addressable files on NVMe. Build an index mapping `expert_id → file_offset`.

### Architecture deep-dive: Qwen3.5-35B-A3B MoE layers

Per MoE layer (40 layers total):
- `gate_proj`: shape [256, 512, 2048] — 256 experts × (intermediate=512, hidden=2048)  
- `up_proj`: shape [256, 512, 2048]
- `down_proj`: shape [256, 2048, 512]
- Router (gate): shape [2048, 256] — always hot in VRAM

Expert weight size (Q4_K_M): ~512 × 2048 × 1.5 bytes ≈ **1.5MB per expert per layer direction**  
Total per expert (gate+up+down × 40 layers): ~1.5MB × 3 × 40 = **180MB per expert**  
Total all 256 experts: **~46GB** (but stored efficiently in GGUF)

### Script: `scripts/build_expert_index.py`
```python
# Parse GGUF format, extract expert weight tensors
# Store as memory-mapped numpy arrays: experts/{layer}/{expert_id}.bin
# Build: index.json = {layer: {expert_id: {offset, size, shape}}}
```

### Bundling optimization (from Apple paper §3.2)
When storing expert weights:
- Co-locate `gate_proj[i]` and `down_proj[i]` in the same file, contiguous
- This doubles the effective NVMe chunk size per read
- Single read latency amortized over 2x data = 2x throughput

---

## Phase 3: Sliding Window Expert Cache (Day 3, ~4h)

### Core idea (Apple paper §3.1 — "Windowing")
Maintain a DRAM + VRAM buffer of the last k=5 tokens' active experts.  
For each new token, only load experts NOT already in the cache.

### Why it works for MoE
- Adjacent tokens in the same document share similar context → similar experts activate
- Measured on Qwen-style MoE: ~40-60% expert reuse between consecutive tokens
- k=5 window → expect ~70% cache hit rate → 70% reduction in NVMe reads

### Data structure: `scripts/expert_cache.py`
```python
class SlidingWindowExpertCache:
    """
    Maintains DRAM buffer of (layer, expert_id) → weight tensor for last k tokens.
    Uses pointer-swap eviction: O(c) evictions without copying existing data.
    """
    def __init__(self, k=5, max_experts_per_layer=30, device="cuda"):
        self.window = deque(maxlen=k)  # list of sets of (layer, expert_id)
        self.cache = {}                # (layer, expert_id) → tensor on device
        self.max_size = max_experts_per_layer
    
    def get_or_load(self, layer, expert_ids, index):
        needed = [(l, e) for l, e in expert_ids if (l, e) not in self.cache]
        if needed:
            self._load_from_nvme(needed, index)
        self.window.append(set(expert_ids))
        self._evict_old()
        return {(l, e): self.cache[(l, e)] for l, e in expert_ids}
```

### Expected improvement: Phase 2 → Phase 3
- Reduce NVMe reads by ~60-70%
- Expected: 2.0 tok/s → 3.0-3.5 tok/s

---

## Phase 4: Async NVMe Prefetch (Day 4, ~4h)

### Core idea
While GPU is computing token T's forward pass, prefetch experts for token T+1.

**The overlap:** GPU compute time >> NVMe read time at 3GB/s
- GPU: 9 experts × 3 directions × 40 layers = 1080 matmuls → ~80-100ms at 8GB VRAM
- NVMe read for fresh experts: ~20-30ms at 3GB/s
- **There's compute time to hide the IO behind**

### Implementation
```python
import asyncio, concurrent.futures

class AsyncExpertPrefetcher:
    def __init__(self, pool_size=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
        self.pending = {}  # future_id → Future
    
    def prefetch(self, layer, expert_ids, index):
        """Fire off NVMe reads while GPU is busy."""
        for layer_id, expert_id in expert_ids:
            key = (layer_id, expert_id)
            if key not in self.pending:
                self.pending[key] = self.executor.submit(self._load, key, index)
    
    def collect(self, expert_ids):
        """Block on results, should be nearly instant if prefetch won."""
        return {k: self.pending.pop(k).result() for k in expert_ids}
```

### Requires: sparsity predictor (Phase 5) to know which experts to prefetch

---

## Phase 5: Low-Rank Expert Predictor (Day 5, ~6h)

### Core idea (Apple paper §3.1 — "Anticipating ReLU Sparsity")
Train a tiny linear probe on top of each layer's attention output to predict which experts will be selected by the router — *before* the router runs.

### Adaptation for Qwen3.5 MoE
Qwen3.5 uses a learned router (softmax gating), not ReLU. We learn to predict the top-8 experts from the attention output:
- Input: attention output, shape [seq, 2048]  
- Output: predicted top-8 expert indices (as logits over 256 experts)
- Loss: binary cross-entropy on whether each expert is in top-8

### Training cost
- 2 epochs on C4 validation (10K samples)
- One predictor per layer (40 predictors total)
- Each predictor: 2048 → 256 linear layer = 524K params → tiny
- Training time on RTX 3070: ~2-3 hours total

### Integration
```python
# At each layer:
# 1. Run attention → attn_out (already on GPU)
# 2. Run predictor(attn_out) → predicted_experts (5ms on GPU)
# 3. Fire async prefetch for predicted experts
# 4. Run actual router → true_experts (very fast)
# 5. Compute: true_experts already loaded (prefetch won) or stall (miss)
```

### Expected predictor accuracy: ~85-90% top-8 recall (per Apple paper, similar architecture)

---

## Phase 6: Full Stack Integration + Benchmarks (Day 6, ~4h)

### Integration order
1. Load attention + embedding + router layers to GPU (always hot, ~5.4GB)
2. Initialize expert cache (k=5, ~1.5GB DRAM)
3. Initialize async prefetcher (4 IO threads)
4. Load + deploy expert predictors (all 40, on GPU)

### Per-token inference loop
```
for each token:
  1. Embed → GPU
  for each layer:
    2. DeltaNet/Attention → attn_out (GPU compute)
    3. predictor(attn_out) → predicted_experts (GPU, fast)
    4. async_prefetch(predicted_experts) → fire NVMe reads
    5. router(attn_out) → true_experts (GPU, fast)
    6. collect(true_experts) → expert weights (from cache or NVMe)
    7. moe_ffn(attn_out, expert_weights) → layer_out (GPU compute)
    8. update_cache(true_experts, attn_out)
  3. LM head → logits → sample
```

### Benchmark: `scripts/benchmark.py`
Fixed protocol (same as autoresearch spirit):
- Prompt: 512 tokens from C4 validation
- Generate: 256 tokens
- Metric: tok/s (wall clock / 256)
- Report: VRAM peak, RAM peak, NVMe bytes read per token, cache hit rate

---

## Autoresearch Experiment Loop

Inspired by karpathy/autoresearch — each phase is an autonomous experiment:

```bash
# Run one 5-minute experiment, log results
uv run python scripts/benchmark.py --phase $PHASE --config $CONFIG --log results/experiment_log.jsonl

# View results table
uv run python scripts/summarize.py results/experiment_log.jsonl
```

An agent can iterate on cache parameters (k size, prefetch depth, quantization level) autonomously, keeping what improves tok/s, discarding what doesn't.

---

## Success Criteria

| Metric | Target | Stretch |
|---|---|---|
| Tokens/second | ≥ 5 tok/s | ≥ 8 tok/s |
| Peak VRAM | ≤ 7.5GB | ≤ 6.5GB |
| Peak RAM | ≤ 12GB | ≤ 8GB |
| Expert predictor recall | ≥ 80% | ≥ 90% |
| Cache hit rate (k=5) | ≥ 60% | ≥ 75% |

---

## Timeline

| Day | Phase | Deliverable |
|---|---|---|
| 1 | 0+1 | Environment, baseline measurement, docs |
| 2 | 2 | Expert index built, bundled storage, validated |
| 3 | 3 | Sliding window cache, Phase 2+3 benchmark |
| 4 | 4 | Async prefetcher, Phase 4 benchmark |
| 5 | 5 | Expert predictor trained + deployed |
| 6 | 6 | Full stack integrated, all benchmarks, README final |

**Total: 6 days of focused work (or 1-2 days with parallel sub-agents)**

---

## Key Related Work

| Paper/Tool | Relevance |
|---|---|
| Apple "LLM in a Flash" (2312.11514) | Windowing + bundling directly applicable to MoE experts |
| KTransformers | Production MoE CPU/GPU expert routing, reference implementation |
| llama.cpp `--no-mmap` + `--split-mode` | Layer-level GPU/CPU split, our starting point |
| DeepSeek-V2 expert offloading | Shows MoE offload is viable at scale with proper caching |
| ExpertFlow (2024) | Expert-level NVMe offloading, most related prior work |
