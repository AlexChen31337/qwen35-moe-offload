# Mapping "LLM in a Flash" Techniques to Qwen3.5 MoE

## Original Paper (Apple, arXiv:2312.11514)

Designed for dense models with ReLU-induced sparsity (OPT 6.7B, Falcon 7B).
Key observation: 90-97% of FFN neurons have zero output per token.

## Our Case: MoE Sparsity is Better

| Property | Dense + ReLU | Qwen3.5 MoE |
|---|---|---|
| Sparsity source | ReLU zero-out | Expert routing (top-8/256) |
| Sparsity rate | 90-97% | 96.5% (247/256 experts idle) |
| Predictability | Low-rank predictor | Router predictor (easier — it's a learned gate) |
| Granularity | Neuron-level (fine) | Expert-level (coarser, simpler) |
| Chunk size | Neuron row/col | Expert FFN block (larger → better NVMe throughput) |

MoE is actually *better suited* for flash offloading than dense+ReLU because:
1. The routing decision is explicit and learned → easier to predict
2. Expert granularity (~60MB Q4 blocks) >> neuron granularity → fewer NVMe seeks
3. Expert co-activation is highly structured → predictable caching patterns

## Technique Mapping

### Windowing → Sliding Window Expert Cache
**Apple paper:** Keep last k tokens' active neurons in DRAM, only load deltas.  
**Our adaptation:** Keep last k tokens' active expert (layer, expert_id) in DRAM+VRAM cache.

Difference: our "neurons" are entire expert FFN blocks (60MB each), not individual neurons.
This means fewer cache entries but each entry is larger → DRAM budget managed differently.

```
Apple: k=5 tokens × ~3% active neurons × model_dim = small DRAM footprint
Ours:  k=5 tokens × 9 experts × 40 layers × 60MB/expert = ~108GB theoretical max
→ In practice: due to expert reuse, actual cache ~1-2GB for k=5
```

### Row-Column Bundling → Expert Up/Down Bundling
**Apple paper:** Co-locate up_proj column[i] and down_proj row[i] on flash.  
**Our adaptation:** Co-locate gate_proj[expert_id] and down_proj[expert_id] per layer per expert.

Same principle: single NVMe read fetches both matrices needed for one expert.
Our chunk size: 2 × 512 × 2048 × 1.5 bytes (Q4) ≈ **3MB per expert per layer** per read.
At 3GB/s sequential: **~1ms per expert per layer** — very fast.

### Low-Rank Predictor → Expert Activation Predictor
**Apple paper:** Train tiny linear probe on attention output to predict which FFN neurons activate.  
**Our adaptation:** Train tiny linear probe to predict which of 256 experts the router will select.

Key advantage: The router itself is a linear layer over 256 experts — our predictor is essentially
a smaller, faster approximation of the router that runs one token ahead.
Training: 40 predictors (one per layer), each 2048→256, binary CE loss, 2 epochs on C4.

### Context-Adaptive Loading → Expert Window Sizing
**Apple paper:** Larger window k reduces IO but increases DRAM usage — tune to available memory.  
**Our adaptation:** k is tunable at runtime based on measured DRAM available.

With 16GB RAM and 5.4GB always-hot in VRAM:
- Available DRAM for cache: ~8GB
- Per cached expert: ~60MB (Q4_K_M, 40 layers folded)
- Max cacheable experts: ~133
- k=5 with 9 active experts/token = 45 slots → easily fits

## What's New vs Apple Paper

1. **MoE routing as predictor target** — cleaner than ReLU prediction (routing is explicit)
2. **Expert-granularity bundling** — coarser than neuron bundling but larger chunks → better IO
3. **Async GPU/NVMe overlap** — GPU compute on token T while prefetching experts for T+1
4. **GDN layers** — Qwen3.5 uses Gated Delta Networks instead of standard attention (novel architecture — need to profile separately)
