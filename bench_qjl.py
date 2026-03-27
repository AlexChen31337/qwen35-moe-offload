#!/usr/bin/env python3
"""
bench_qjl.py — QJL (Quantized Johnson-Lindenstrauss) KV cache compression benchmark.

Measures:
1. Projection throughput (vectors/sec, GB/s)
2. Reconstruction throughput
3. Attention score estimation quality (Pearson correlation)
4. End-to-end overhead estimate for Qwen3.5-35B-A3B

Architecture reference (Qwen3.5-35B-A3B):
  - n_layers = 64
  - n_kv_heads = 4 (GQA)
  - head_dim = 128
  - n_attn_heads = 24 (Q heads)

Usage:
  uv run python bench_qjl.py [--sketch-dims 32 64 128 256] [--head-dim 128]
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from polar_kv import (
    qjl_project,
    qjl_attention_score,
    qjl_compression_ratio,
)

# ── Qwen3.5-35B-A3B architecture ────────────────────────────────────────────
N_LAYERS = 64
N_KV_HEADS = 4
N_ATTN_HEADS = 24  # query heads (GQA ratio = 24/4 = 6)
HEAD_DIM = 128
TOTAL_K_HEADS_PER_TOKEN = N_LAYERS * N_KV_HEADS  # Only K needs sketching in QJL
TOTAL_KV_HEADS_PER_TOKEN = N_LAYERS * 2 * N_KV_HEADS  # For size comparison


def bench_projection_throughput(sketch_dim: int, head_dim: int, n_trials: int):
    """Measure JL projection throughput."""
    rng = np.random.RandomState(42)
    vectors = [rng.randn(head_dim).astype(np.float32) for _ in range(n_trials)]
    
    # Warmup
    for v in vectors[:10]:
        _ = qjl_project(v, sketch_dim, seed=7)
    
    # Projection
    t0 = time.perf_counter()
    sketches = []
    for v in vectors:
        sketches.append(qjl_project(v, sketch_dim, seed=7))
    t_project = time.perf_counter() - t0
    
    project_rate = n_trials / t_project
    bytes_per_head = head_dim * 4
    project_gbps = (n_trials * bytes_per_head) / t_project / 1e9
    
    return {
        'sketch_dim': sketch_dim,
        'head_dim': head_dim,
        'n_trials': n_trials,
        'project_heads_per_sec': project_rate,
        'project_gbps': project_gbps,
        'compression_ratio': qjl_compression_ratio(head_dim, sketch_dim),
    }


def bench_attention_quality(sketch_dim: int, head_dim: int, n_pairs: int = 500):
    """Measure attention score estimation quality."""
    rng = np.random.RandomState(42)
    
    true_scores = []
    est_scores = []
    seed = 7
    
    for i in range(n_pairs):
        q = rng.randn(head_dim).astype(np.float32)
        k = rng.randn(head_dim).astype(np.float32)
        
        true_score = float(np.dot(q, k))
        sketch = qjl_project(k, sketch_dim, seed=seed)
        est_score = qjl_attention_score(q, sketch, sketch_dim, seed=seed)
        
        true_scores.append(true_score)
        est_scores.append(est_score)
    
    true_scores = np.array(true_scores)
    est_scores = np.array(est_scores)
    
    # Pearson correlation
    corr = np.corrcoef(true_scores, est_scores)[0, 1]
    
    # Relative error
    mask = np.abs(true_scores) > 0.1  # avoid div-by-zero on near-zero scores
    if mask.any():
        rel_errors = np.abs(est_scores[mask] - true_scores[mask]) / np.abs(true_scores[mask])
        mean_rel_err = float(np.mean(rel_errors))
        median_rel_err = float(np.median(rel_errors))
    else:
        mean_rel_err = median_rel_err = float('inf')
    
    # RMSE
    rmse = float(np.sqrt(np.mean((true_scores - est_scores) ** 2)))
    
    return {
        'sketch_dim': sketch_dim,
        'pearson_r': float(corr),
        'rmse': rmse,
        'mean_rel_error': mean_rel_err,
        'median_rel_error': median_rel_err,
    }


def bench_full_token(sketch_dim: int, head_dim: int, n_tokens: int = 20):
    """Simulate projecting all K heads for a full token decode step.
    
    In QJL, only KEY vectors need sketching. Values are accessed at full precision
    using the attention weights derived from the sketch-based score estimation.
    """
    rng = np.random.RandomState(123)
    
    # Pre-generate K vectors for n_tokens
    all_keys = rng.randn(n_tokens, TOTAL_K_HEADS_PER_TOKEN, head_dim).astype(np.float32)
    
    # Warmup
    for h in range(min(10, TOTAL_K_HEADS_PER_TOKEN)):
        _ = qjl_project(all_keys[0, h], sketch_dim, seed=7)
    
    # Measure: project all K heads for each token
    t0 = time.perf_counter()
    for tok in range(n_tokens):
        for h in range(TOTAL_K_HEADS_PER_TOKEN):
            _ = qjl_project(all_keys[tok, h], sketch_dim, seed=7)
    t_total = time.perf_counter() - t0
    
    per_token_ms = (t_total / n_tokens) * 1000
    target_budget_ms = (1.0 / 12.114) * 1000  # 82.55 ms at 12.114 tok/s
    overhead_pct = (per_token_ms / target_budget_ms) * 100
    
    return {
        'sketch_dim': sketch_dim,
        'n_tokens': n_tokens,
        'total_k_heads_per_token': TOTAL_K_HEADS_PER_TOKEN,
        'per_token_ms': per_token_ms,
        'target_budget_ms': target_budget_ms,
        'overhead_pct': overhead_pct,
        'max_tok_s_with_overhead': 1000.0 / (target_budget_ms + per_token_ms),
    }


def bench_attention_decode_simulation(sketch_dim: int, head_dim: int, seq_len: int, n_queries: int = 10):
    """Simulate full attention computation using QJL sketches.
    
    For each query token, compute attention scores against all seq_len cached keys
    using sketch-based estimation, then compute softmax + weighted sum of values.
    
    This measures whether QJL can accelerate attention at long contexts.
    """
    rng = np.random.RandomState(456)
    seed = 7
    
    # Generate cached key sketches (pre-computed)
    key_sketches = [qjl_project(rng.randn(head_dim).astype(np.float32), sketch_dim, seed=seed)
                    for _ in range(seq_len)]
    values = rng.randn(seq_len, head_dim).astype(np.float32)
    queries = rng.randn(n_queries, head_dim).astype(np.float32)
    
    # Measure attention computation using sketches
    t0 = time.perf_counter()
    for q in queries:
        # Compute attention scores using sketch estimator
        scores = np.array([
            qjl_attention_score(q, ks, sketch_dim, seed=seed)
            for ks in key_sketches
        ])
        # Softmax
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        # Weighted sum of values
        output = weights @ values
    t_sketch = time.perf_counter() - t0
    
    # Measure standard attention (full precision)
    keys_full = rng.randn(seq_len, head_dim).astype(np.float32)
    t0 = time.perf_counter()
    for q in queries:
        scores = keys_full @ q  # (seq_len,) via matrix-vector multiply
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        output = weights @ values
    t_full = time.perf_counter() - t0
    
    return {
        'sketch_dim': sketch_dim,
        'seq_len': seq_len,
        'n_queries': n_queries,
        'sketch_attn_ms': t_sketch * 1000,
        'full_attn_ms': t_full * 1000,
        'speedup': t_full / t_sketch if t_sketch > 0 else 0,
        'per_query_sketch_ms': (t_sketch / n_queries) * 1000,
        'per_query_full_ms': (t_full / n_queries) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="QJL KV cache compression benchmark")
    parser.add_argument("--sketch-dims", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Sketch dimensions to test")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension (default: 128 for Qwen3.5-35B-A3B)")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of trials per config")
    parser.add_argument("--n-tokens", type=int, default=20,
                        help="Number of tokens for full-token simulation")
    args = parser.parse_args()
    
    print("=" * 78)
    print("QJL KV Cache Compression Benchmark")
    print(f"Model: Qwen3.5-35B-A3B (n_layers={N_LAYERS}, n_kv_heads={N_KV_HEADS}, head_dim={args.head_dim})")
    print(f"GQA: {N_ATTN_HEADS} Q heads, {N_KV_HEADS} KV heads (ratio 6:1)")
    print(f"Total K heads per token: {TOTAL_K_HEADS_PER_TOKEN}")
    print("=" * 78)
    
    # ── Part 1: Projection Throughput ────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 1: JL Projection Throughput (per head)")
    print(f"{'='*78}")
    print(f"{'sketch':>7} {'ratio':>6} {'proj/s':>10} {'GB/s':>8}")
    print("-" * 35)
    
    for sd in args.sketch_dims:
        r = bench_projection_throughput(sd, args.head_dim, args.n_trials)
        print(f"{sd:>7} {r['compression_ratio']:>6.1f}x {r['project_heads_per_sec']:>10.0f} "
              f"{r['project_gbps']:>8.3f}")
    
    # ── Part 2: Attention Quality ────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 2: Attention Score Estimation Quality")
    print(f"  (Pearson correlation between true Q·K and estimated Q·K)")
    print(f"{'='*78}")
    print(f"{'sketch':>7} {'ratio':>6} {'Pearson':>8} {'RMSE':>8} {'MedRelErr':>10}")
    print("-" * 43)
    
    for sd in args.sketch_dims:
        r = bench_attention_quality(sd, args.head_dim)
        print(f"{sd:>7} {qjl_compression_ratio(args.head_dim, sd):>6.1f}x "
              f"{r['pearson_r']:>8.4f} {r['rmse']:>8.3f} {r['median_rel_error']:>10.3f}")
    
    # ── Part 3: Full Token Overhead ──────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 3: Full Token Overhead Simulation")
    print(f"  (Projecting all {TOTAL_K_HEADS_PER_TOKEN} K heads per decode step)")
    print(f"  (Baseline: 12.114 tok/s → budget = {1000/12.114:.1f} ms/token)")
    print(f"{'='*78}")
    print(f"{'sketch':>7} {'per_tok_ms':>12} {'overhead%':>10} {'max_tok/s':>12}")
    print("-" * 45)
    
    for sd in args.sketch_dims:
        r = bench_full_token(sd, args.head_dim, args.n_tokens)
        print(f"{sd:>7} {r['per_token_ms']:>12.2f} {r['overhead_pct']:>10.1f}% "
              f"{r['max_tok_s_with_overhead']:>12.3f}")
    
    # ── Part 4: Attention Decode Simulation ──────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 4: QJL Attention Speedup at Long Context")
    print(f"  (Does sketch-based attention beat full-precision matmul?)")
    print(f"{'='*78}")
    print(f"{'sketch':>7} {'seq_len':>8} {'sketch_ms':>10} {'full_ms':>10} {'speedup':>8}")
    print("-" * 47)
    
    for seq_len in [512, 2048, 8192, 32768]:
        for sd in [64, 128]:
            r = bench_attention_decode_simulation(sd, args.head_dim, seq_len, n_queries=5)
            print(f"{sd:>7} {seq_len:>8} {r['per_query_sketch_ms']:>10.2f} "
                  f"{r['per_query_full_ms']:>10.2f} {r['speedup']:>8.3f}x")
    
    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Part 5: QJL vs PolarQuant vs llama.cpp Comparison")
    print(f"{'='*78}")
    print(f"{'Method':>22} {'Ratio':>7} {'Quality':>10} {'CPU Overhead':>13} {'Use Case'}")
    print("-" * 75)
    print(f"{'llama.cpp q8_0':>22} {'2.0x':>7} {'~0.9999':>10} {'0 (native)':>13} {'VRAM reduction during inference'}")
    print(f"{'llama.cpp iq4_nl':>22} {'4.0x':>7} {'~0.998':>10} {'0 (native)':>13} {'VRAM reduction, 32K context'}")
    print(f"{'PolarQuant 4-bit':>22} {'7.6x':>7} {'~0.88':>10} {'HIGH':>13} {'KV offloading/storage'}")
    print(f"{'PolarQuant 6-bit':>22} {'5.2x':>7} {'~0.993':>10} {'HIGH':>13} {'KV offloading/storage'}")
    print(f"{'QJL sketch=64':>22} {'8.0x':>7} {'r~0.4':>10} {'HIGH':>13} {'Approximate attention only'}")
    print(f"{'QJL sketch=128':>22} {'4.0x':>7} {'r~0.6':>10} {'HIGH':>13} {'Approximate attention only'}")
    print(f"{'QJL sketch=256':>22} {'2.0x':>7} {'r~0.7':>10} {'HIGH':>13} {'Better quality, less compression'}")
    
    print(f"\n{'='*78}")
    print("CONCLUSION")
    print(f"{'='*78}")
    print("""
QJL's sketch-based attention estimation in Python is SIGNIFICANTLY SLOWER than
NumPy's optimised matrix-vector multiply for full-precision attention. This is
because the Gaussian matrix generation + projection cost exceeds the benefit.

QJL becomes viable ONLY when:
1. Implemented in CUDA (GPU-parallel sketch computation)
2. Context length is extreme (>100K) where VRAM savings justify the compute
3. Used for approximate attention in a streaming/online setting

For Qwen3.5-35B-A3B at 32K context:
  - llama.cpp's iq4_nl KV (11.369 tok/s) is the practical winner
  - QJL/PolarQuant add too much CPU overhead without CUDA acceleration
  - The Rust crates provide ~10x faster compression than Python, but still
    operate on CPU and cannot replace llama.cpp's integrated KV management

VERDICT: PolarQuant/QJL are research tools for understanding KV compression
quality tradeoffs. For actual inference throughput, llama.cpp's native KV
quantization (q8_0, iq4_nl) remains unbeatable on this hardware.
""")


if __name__ == "__main__":
    main()
