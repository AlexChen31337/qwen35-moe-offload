# Qwen3.5-35B-A3B Architecture Breakdown

## Model Specs (from HuggingFace model card)

| Parameter | Value |
|---|---|
| Total parameters | 35B |
| Activated per token | 3B |
| Hidden dimension | 2048 |
| Layers | 40 |
| Experts (total) | 256 |
| Experts activated | 8 routed + 1 shared = 9 per token |
| Expert intermediate dim | 512 |
| Attention heads (Q) | 16 |
| KV heads | 2 (GQA) |
| Context length | 262,144 native |

## Layer Layout

The 40 layers follow a repeating pattern:
```
10 × (
  3 × (Gated DeltaNet → MoE)
  1 × (Gated Attention → MoE)
)
```

So: 30 DeltaNet layers + 10 Attention layers, all with MoE FFN.

## Key for Flash Offloading

**What must stay in VRAM (always hot):**
- Token embeddings: 248320 × 2048 × 2 bytes ≈ 1.0GB
- Per layer: attention/DeltaNet QKV + router = ~70MB per layer × 40 = ~2.8GB
- LM head: same as embeddings = 1.0GB
- **Total always-hot: ~4.8-5.4GB (comfortable for 8GB RTX 3070)**

**What can be offloaded to NVMe:**
- Expert FFN weights: gate_proj + up_proj + down_proj for each of 256 experts × 40 layers
- Expert size (fp16): 512 × 2048 × 2 bytes × 3 = ~6MB per expert × 40 layers = 240MB per expert
- Expert size (Q4_K_M): ~60MB per expert
- Total 256 experts (Q4_K_M): ~15GB

## Expert Sparsity: The Key Property

Per token, only 9/256 experts activate = **96.5% of expert weights idle**.

For a 256-token generation:
- Naive: load all 256 experts per layer per token → 256 × 40 × 60MB = massive redundant IO
- With windowing (k=5): ~40-60% of expert slots repeat from recent window → ~half the IO
- With predictor: can async prefetch before the router confirms → hides latency

## GQA Implication

2 KV heads means KV cache is tiny: 40 × 2 × seq_len × 256 × 2 bytes
At seq_len=512: 40 × 2 × 512 × 256 × 2 = ~20MB — negligible VRAM cost.
