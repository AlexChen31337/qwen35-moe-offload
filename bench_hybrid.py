#!/usr/bin/env python3
"""
bench_hybrid.py — Hybrid KV cache compression analysis.

Analyzes the feasibility of combining llama.cpp's native KV quantization
with PolarQuant/QJL for KV cache offloading to system RAM.

Key insight: llama.cpp manages KV in VRAM. We can't modify its internal
representation. But we CAN:
1. Run with smaller n_ctx (fits in VRAM with q8_0/iq4_nl)
2. Use PolarQuant/QJL to compress+offload OLD KV entries to RAM
3. Reload compressed KV when context window slides

This simulates the "infinite context via RAM offload" scenario.
"""

import sys
import json
from pathlib import Path

# Model architecture
N_LAYERS = 64
N_KV_HEADS = 4
HEAD_DIM = 128
VRAM_TOTAL_MB = 8192  # RTX 3070 8GB
VRAM_MODEL_MB = 7374  # Measured: model weights use ~7374 MB with n_gpu=16
VRAM_AVAILABLE_MB = VRAM_TOTAL_MB - VRAM_MODEL_MB  # ~818 MB for KV cache
RAM_AVAILABLE_MB = 256 * 1024  # 256 GB swapfile on GPU server

# KV sizes per token per format
def kv_bytes_per_token(format_name):
    """Calculate KV cache size per token for different formats."""
    heads = N_LAYERS * 2 * N_KV_HEADS  # 512 KV heads total
    if format_name == "f16":
        return heads * HEAD_DIM * 2  # 2 bytes per f16
    elif format_name == "q8_0":
        return heads * HEAD_DIM * 1  # ~1 byte per element in q8_0
    elif format_name == "iq4_nl":
        return heads * HEAD_DIM * 0.5  # ~0.5 byte per element
    elif format_name == "pq4":
        # PolarQuant 4-bit: radius(4 bytes) + 127*4 bits = 68 bytes per head
        return heads * 68
    elif format_name == "pq6":
        # PolarQuant 6-bit: radius(4 bytes) + 127*6 bits = 4 + 96 = 100 bytes per head
        return heads * 100
    elif format_name == "pq8":
        # PolarQuant 8-bit: radius(4 bytes) + 127*8 bits = 4 + 127 = 131 bytes per head  
        return heads * 131
    elif format_name == "qjl32":
        return heads * 32  # 32 i8 per head
    elif format_name == "qjl64":
        return heads * 64
    elif format_name == "qjl128":
        return heads * 128
    else:
        raise ValueError(f"Unknown format: {format_name}")


def main():
    print("=" * 78)
    print("Hybrid KV Cache Compression Analysis")
    print(f"GPU: RTX 3070 8GB | Model VRAM: ~{VRAM_MODEL_MB} MB | Free for KV: ~{VRAM_AVAILABLE_MB} MB")
    print(f"System RAM: {RAM_AVAILABLE_MB // 1024} GB (via swapfile)")
    print("=" * 78)
    
    # Part 1: Max context per format in VRAM
    print(f"\n{'='*78}")
    print("Part 1: Maximum Context Length in VRAM Only")
    print(f"{'='*78}")
    print(f"{'Format':>12} {'bytes/tok':>10} {'KB/tok':>8} {'Max n_ctx':>10} {'VRAM used':>10}")
    print("-" * 54)
    
    formats = ["f16", "q8_0", "iq4_nl", "pq4", "pq6", "pq8", "qjl32", "qjl64", "qjl128"]
    for fmt in formats:
        bpt = kv_bytes_per_token(fmt)
        kbpt = bpt / 1024
        max_ctx = int(VRAM_AVAILABLE_MB * 1024 * 1024 / bpt)
        vram_at_32k = bpt * 32768 / 1024 / 1024
        print(f"{fmt:>12} {bpt:>10.0f} {kbpt:>8.1f} {max_ctx:>10,} {vram_at_32k:>9.0f} MB")
    
    # Part 2: Hybrid VRAM + RAM offloading scenario
    print(f"\n{'='*78}")
    print("Part 2: Hybrid VRAM (active) + RAM (offloaded, PolarQuant compressed)")
    print(f"{'='*78}")
    print("Scenario: Keep last N tokens in VRAM (q8_0), offload older tokens to RAM (PQ-compressed)")
    print()
    
    # Active window in VRAM using iq4_nl
    vram_format = "iq4_nl"
    vram_bpt = kv_bytes_per_token(vram_format)
    active_windows = [512, 2048, 8192]
    offload_formats = ["pq4", "pq6", "qjl64"]
    
    print(f"{'Active':>8} {'VRAM fmt':>9} {'Offload':>8} {'VRAM MB':>8} {'Total ctx':>10} {'RAM MB':>8} {'Notes'}")
    print("-" * 75)
    
    for active_ctx in active_windows:
        vram_used_mb = active_ctx * vram_bpt / 1024 / 1024
        for off_fmt in offload_formats:
            off_bpt = kv_bytes_per_token(off_fmt)
            # How many tokens can we offload to 10 GB of RAM?
            ram_budget_mb = 10240  # 10 GB for KV offload
            offload_tokens = int(ram_budget_mb * 1024 * 1024 / off_bpt)
            total_ctx = active_ctx + offload_tokens
            ram_used_mb = offload_tokens * off_bpt / 1024 / 1024
            
            print(f"{active_ctx:>8} {vram_format:>9} {off_fmt:>8} {vram_used_mb:>8.1f} "
                  f"{total_ctx:>10,} {ram_used_mb:>8.0f} "
                  f"{'✓ fits' if vram_used_mb < VRAM_AVAILABLE_MB else '✗ OOM'}")
    
    # Part 3: Overhead analysis for hybrid approach
    print(f"\n{'='*78}")
    print("Part 3: CPU Overhead for Hybrid Offloading")
    print(f"{'='*78}")
    print("When a token exits the active window, compress with PolarQuant and store in RAM.")
    print("This happens once per generated token (amortised over the active window size).")
    print()
    
    # Rust PolarQuant overhead: ~3.1ms per full-token compression (512 heads)
    # But we only compress ONE token's KV at a time (the oldest in the window)
    # That's 512 heads × (compress time per head)
    # From Rust bench: ~150000 heads/sec = 6.67μs per head
    # 512 heads = 3.41ms per evicted token
    pq_per_head_us = 1e6 / 150000  # microseconds per head
    pq_per_token_ms = pq_per_head_us * (N_LAYERS * 2 * N_KV_HEADS) / 1000
    
    qjl_per_head_us_64 = 1e6 / 26986  # sketch_dim=64
    qjl_per_token_ms_64 = qjl_per_head_us_64 * (N_LAYERS * N_KV_HEADS) / 1000  # Only K heads
    
    budget_ms = 1000.0 / 12.114
    
    print(f"  PolarQuant 4-bit (Rust): {pq_per_token_ms:.2f} ms/eviction = {pq_per_token_ms/budget_ms*100:.1f}% of token budget")
    print(f"  QJL sketch=64 (Rust):    {qjl_per_token_ms_64:.2f} ms/eviction = {qjl_per_token_ms_64/budget_ms*100:.1f}% of token budget")
    print(f"  Token budget at 12.114 tok/s: {budget_ms:.1f} ms")
    print()
    print("  → PolarQuant adds ~4% overhead per token = ~11.65 tok/s effective")
    print("  → Acceptable! Enables 100K+ context at near-baseline speed")
    
    # Part 4: Quality implications
    print(f"\n{'='*78}")
    print("Part 4: Quality Implications")
    print(f"{'='*78}")
    print("""
For KV cache OFFLOADING (not inline replacement):
  - Offloaded tokens' KV cache goes through: q8_0 (in VRAM) → PQ4 (in RAM)
  - When needed, decompress: PQ4 (RAM) → approximate f32 → feed back to attention
  - Quality loss: cosine_sim ~0.90 (PQ4) or ~0.99 (PQ6) on the offloaded portion
  - Active window tokens: ZERO quality loss (still using native q8_0)
  
Key insight: For very long contexts, older tokens contribute LESS to attention
(attention scores decay). So approximate KV for old tokens is acceptable.

Recommendation:
  1. PolarQuant 6-bit for high-quality offloading (5.2x compression, cos_sim ~0.99)
  2. PolarQuant 4-bit for aggressive offloading (7.6x compression, cos_sim ~0.89)
  3. QJL is NOT suitable for KV offloading (it compresses K only, not the full KV)
""")
    
    # Part 5: Practical limitation
    print(f"{'='*78}")
    print("Part 5: Practical Limitation — Why This Can't Be Tested End-to-End")
    print(f"{'='*78}")
    print("""
llama.cpp's KV cache is managed internally in a contiguous VRAM buffer.
There is NO API to:
  1. Extract individual token's KV entries from the cache
  2. Re-inject modified/approximate KV entries back into the cache
  3. Selectively evict specific tokens from the KV cache

To actually implement hybrid offloading, we would need to:
  - Patch llama.cpp's kv_cache.cpp to expose KV extract/inject APIs
  - Or use the llama_kv_cache_seq_rm() + manual management approach
  - This is a SIGNIFICANT engineering effort beyond this benchmark phase

VERDICT: The COMPRESSION ALGORITHMS work and are fast enough in Rust.
The INTEGRATION with llama.cpp is the bottleneck, not the algorithms.
""")


if __name__ == "__main__":
    main()
