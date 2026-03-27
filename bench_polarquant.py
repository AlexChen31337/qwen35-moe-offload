#!/usr/bin/env python3
"""
bench_polarquant.py — PolarQuant KV cache compression benchmark.

Measures:
1. Compression throughput (vectors/sec, GB/s)
2. Decompression throughput
3. Quality (cosine similarity) at various bit widths
4. End-to-end overhead estimate for Qwen3.5-35B-A3B

Architecture reference (Qwen3.5-35B-A3B):
  - n_layers = 64
  - n_kv_heads = 4 (GQA)
  - head_dim = 128
  - KV per token: 64 × 2 × 4 × 128 = 65,536 floats = 256 KB (f32)

Usage:
  uv run python bench_polarquant.py [--bits 4] [--head-dim 128] [--n-trials 100]
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

# Add scripts dir for polar_kv module
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from polar_kv import (
    polarquant_compress,
    polarquant_decompress,
    polarquant_round_trip,
    compression_ratio,
)

# ── Qwen3.5-35B-A3B architecture ────────────────────────────────────────────
N_LAYERS = 64
N_KV_HEADS = 4
HEAD_DIM = 128
TOTAL_HEADS_PER_TOKEN = N_LAYERS * 2 * N_KV_HEADS  # K+V


def bench_compression_throughput(bits: int, head_dim: int, n_trials: int):
    """Measure single-head compression + decompression throughput."""
    rng = np.random.RandomState(42)
    vectors = [rng.randn(head_dim).astype(np.float32) for _ in range(n_trials)]
    
    # Warmup
    for v in vectors[:10]:
        c = polarquant_compress(v, bits)
        _ = polarquant_decompress(c)
    
    # Compression
    t0 = time.perf_counter()
    compressed = []
    for v in vectors:
        compressed.append(polarquant_compress(v, bits))
    t_compress = time.perf_counter() - t0
    
    # Decompression
    t0 = time.perf_counter()
    decompressed = []
    for c in compressed:
        decompressed.append(polarquant_decompress(c))
    t_decompress = time.perf_counter() - t0
    
    # Quality
    sims = []
    for orig, recon in zip(vectors, decompressed):
        dot = np.dot(orig, recon)
        n_a = np.linalg.norm(orig)
        n_b = np.linalg.norm(recon)
        if n_a > 1e-12 and n_b > 1e-12:
            sims.append(float(np.clip(dot / (n_a * n_b), -1.0, 1.0)))
    
    compress_rate = n_trials / t_compress  # heads/sec
    decompress_rate = n_trials / t_decompress
    bytes_per_head = head_dim * 4  # f32
    compress_gbps = (n_trials * bytes_per_head) / t_compress / 1e9
    decompress_gbps = (n_trials * bytes_per_head) / t_decompress / 1e9
    
    return {
        'bits': bits,
        'head_dim': head_dim,
        'n_trials': n_trials,
        'compress_heads_per_sec': compress_rate,
        'decompress_heads_per_sec': decompress_rate,
        'compress_gbps': compress_gbps,
        'decompress_gbps': decompress_gbps,
        'avg_cosine_sim': float(np.mean(sims)),
        'min_cosine_sim': float(np.min(sims)),
        'p5_cosine_sim': float(np.percentile(sims, 5)),
        'compression_ratio': compression_ratio(head_dim, bits),
    }


def bench_full_token(bits: int, head_dim: int, n_tokens: int = 100):
    """Simulate compressing all KV heads for a full token decode step."""
    rng = np.random.RandomState(123)
    
    # Pre-generate KV data for n_tokens
    all_kv = rng.randn(n_tokens, TOTAL_HEADS_PER_TOKEN, head_dim).astype(np.float32)
    
    # Warmup
    for h in range(min(10, TOTAL_HEADS_PER_TOKEN)):
        c = polarquant_compress(all_kv[0, h], bits)
        _ = polarquant_decompress(c)
    
    # Measure: compress all heads for each token
    t0 = time.perf_counter()
    for tok in range(n_tokens):
        for h in range(TOTAL_HEADS_PER_TOKEN):
            c = polarquant_compress(all_kv[tok, h], bits)
            # In real usage, we'd store c and decompress on-demand
    t_total = time.perf_counter() - t0
    
    per_token_ms = (t_total / n_tokens) * 1000
    # If generation runs at 12 tok/s, each token budget is ~83ms
    # How much of that budget does PolarQuant consume?
    target_budget_ms = (1.0 / 12.0) * 1000
    overhead_pct = (per_token_ms / target_budget_ms) * 100
    
    return {
        'bits': bits,
        'n_tokens': n_tokens,
        'total_heads_per_token': TOTAL_HEADS_PER_TOKEN,
        'per_token_ms': per_token_ms,
        'target_budget_ms': target_budget_ms,
        'overhead_pct': overhead_pct,
        'max_tok_s_with_overhead': 1000.0 / (target_budget_ms + per_token_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="PolarQuant KV cache compression benchmark")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8],
                        help="Bit widths to test")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension (default: 128 for Qwen3.5-35B-A3B)")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of compression trials per config")
    parser.add_argument("--n-tokens", type=int, default=20,
                        help="Number of tokens for full-token simulation")
    args = parser.parse_args()
    
    print("=" * 78)
    print("PolarQuant KV Cache Compression Benchmark")
    print(f"Model: Qwen3.5-35B-A3B (n_layers={N_LAYERS}, n_kv_heads={N_KV_HEADS}, head_dim={args.head_dim})")
    print(f"Total KV heads per token: {TOTAL_HEADS_PER_TOKEN}")
    print("=" * 78)
    
    # ── Per-head throughput ───────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 1: Per-Head Compression Throughput")
    print(f"{'='*78}")
    print(f"{'bits':>4} {'ratio':>6} {'cos_sim':>8} {'min_sim':>8} {'comp/s':>10} {'decomp/s':>10} {'comp GB/s':>10}")
    print("-" * 66)
    
    results = []
    for bits in args.bits:
        r = bench_compression_throughput(bits, args.head_dim, args.n_trials)
        results.append(r)
        print(f"{r['bits']:>4} {r['compression_ratio']:>6.2f}x {r['avg_cosine_sim']:>8.4f} "
              f"{r['min_cosine_sim']:>8.4f} {r['compress_heads_per_sec']:>10.0f} "
              f"{r['decompress_heads_per_sec']:>10.0f} {r['compress_gbps']:>10.3f}")
    
    # ── Full-token simulation ────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 2: Full Token Overhead Simulation")
    print(f"  (Simulates compressing all {TOTAL_HEADS_PER_TOKEN} KV heads per decode step)")
    print(f"  (Baseline: 12.114 tok/s → budget = {1000/12.114:.1f} ms/token)")
    print(f"{'='*78}")
    print(f"{'bits':>4} {'per_tok_ms':>12} {'overhead%':>10} {'max_tok/s':>12} {'speedup_vs_12':>14}")
    print("-" * 56)
    
    for bits in args.bits:
        r = bench_full_token(bits, args.head_dim, args.n_tokens)
        speedup = r['max_tok_s_with_overhead'] / 12.114
        print(f"{bits:>4} {r['per_token_ms']:>12.2f} {r['overhead_pct']:>10.1f}% "
              f"{r['max_tok_s_with_overhead']:>12.3f} {speedup:>14.4f}x")
    
    # ── Quality analysis ─────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 3: Quality Analysis — Cosine Similarity Distribution")
    print(f"{'='*78}")
    rng = np.random.RandomState(999)
    for bits in args.bits:
        sims = []
        for i in range(1000):
            kv = rng.randn(args.head_dim).astype(np.float32)
            _, sim = polarquant_round_trip(kv, bits=bits)
            sims.append(sim)
        sims = np.array(sims)
        print(f"  {bits}-bit: mean={np.mean(sims):.4f} std={np.std(sims):.4f} "
              f"min={np.min(sims):.4f} p5={np.percentile(sims,5):.4f} "
              f"p50={np.median(sims):.4f} p95={np.percentile(sims,95):.4f}")
    
    # ── Comparison with llama.cpp built-in KV types ──────────────────────────
    print(f"\n{'='*78}")
    print("Part 4: Comparison with llama.cpp Built-in KV Compression")
    print(f"{'='*78}")
    print(f"{'Method':>20} {'Comp. Ratio':>12} {'Quality':>12} {'Notes'}")
    print("-" * 70)
    print(f"{'f16 (baseline)':>20} {'1.00x':>12} {'lossless':>12} llama.cpp default")
    print(f"{'q8_0':>20} {'2.00x':>12} {'~0.9999':>12} llama.cpp built-in, best Phase6 speed")
    print(f"{'q4_0':>20} {'4.00x':>12} {'~0.995':>12} llama.cpp built-in")
    print(f"{'iq4_nl':>20} {'4.00x':>12} {'~0.998':>12} llama.cpp built-in, Phase5 32K winner")
    for bits in [4, 6, 8]:
        r = [x for x in results if x['bits'] == bits][0]
        ratio_str = f"{r['compression_ratio']:.2f}x"
        sim_str = f"~{r['avg_cosine_sim']:.4f}"
        print(f"{'PolarQuant '+str(bits)+'b':>20} {ratio_str:>12} {sim_str:>12} Our impl, CPU post-process")
    
    print(f"\n{'='*78}")
    print("CONCLUSION")
    print(f"{'='*78}")
    print("""
PolarQuant is a POST-PROCESSING compression that runs on CPU AFTER llama.cpp
generates each token. It cannot replace llama.cpp's internal KV quantization
because llama.cpp manages its own KV buffer during attention computation.

Practical use case: PolarQuant could compress KV cache for STORAGE (disk
offloading) or TRANSFER (multi-GPU) but NOT for reducing VRAM during inference
(that's what llama.cpp's type_k/type_v already does).

The overhead analysis above shows how much CPU time PolarQuant would add if
used as a post-decode compression step (e.g., for KV cache offloading to RAM).
""")


if __name__ == "__main__":
    main()
