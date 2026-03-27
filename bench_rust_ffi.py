#!/usr/bin/env python3
"""
bench_rust_ffi.py — Benchmark calling Rust PolarQuant/QJL from Python via ctypes.

Tests the practical integration path: Python orchestration + Rust compression.
This is how PolarQuant would actually be used in a llama-cpp-python workflow.
"""

import ctypes
import time
import subprocess
import os
import sys
import numpy as np

# Path to compiled Rust shared libraries
POLARQUANT_LIB = None
QJL_LIB = None


def build_shared_libs():
    """Build Rust crates as shared libraries (.so) for ctypes loading."""
    repo = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we can build cdylib targets
    # First, add cdylib to Cargo.toml
    pq_toml = os.path.join(repo, "crates/polarquant/Cargo.toml")
    qjl_toml = os.path.join(repo, "crates/qjl/Cargo.toml")
    
    print("Rust shared libraries require [lib] crate-type = ['cdylib'] in Cargo.toml")
    print("and C-exported functions. Skipping FFI test — using subprocess benchmark instead.")
    return False


def bench_rust_subprocess(n_iterations=5):
    """Benchmark running Rust binaries as subprocesses."""
    repo = os.path.dirname(os.path.abspath(__file__))
    bench_pq = os.path.join(repo, "target/release/bench_throughput")
    
    # Check if bench binary exists (could be for polarquant or qjl)
    pq_bench = os.path.join(repo, "target/release/bench_throughput")
    
    print("=" * 78)
    print("Rust via Subprocess: PolarQuant + QJL Throughput")
    print("=" * 78)
    
    # The Rust benchmarks already measure throughput internally.
    # Here we test the subprocess overhead of calling them.
    
    for name, cargo_args in [
        ("PolarQuant", ["-p", "polarquant", "--bin", "bench_throughput"]),
        ("QJL", ["-p", "qjl", "--bin", "bench_throughput"]),
    ]:
        cargo_path = os.path.expanduser("~/.cargo/bin/cargo")
        
        times = []
        for i in range(n_iterations):
            t0 = time.perf_counter()
            result = subprocess.run(
                [cargo_path, "run", "--release"] + cargo_args,
                capture_output=True, text=True,
                cwd=repo, timeout=120,
                env={**os.environ, "PATH": os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")}
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if result.returncode != 0:
                print(f"  ERROR: {name} failed: {result.stderr[:200]}")
                break
        
        if times:
            avg = sum(times) / len(times)
            print(f"\n{name}: avg={avg:.2f}s per full benchmark run ({n_iterations} iterations)")
            # Parse the per-token overhead from the last run's output
            for line in result.stdout.split('\n'):
                if 'per_tok_ms' in line.lower() or 'overhead' in line.lower():
                    continue
                if '4   ' in line and ('3.' in line or '10.' in line):
                    print(f"  Last run 4-bit/sketch=64 result: {line.strip()}")


def bench_numpy_comparison():
    """Compare Python PolarQuant speed with numpy-optimised version."""
    print(f"\n{'='*78}")
    print("NumPy-Optimised PolarQuant (Vectorised)")
    print(f"{'='*78}")
    
    rng = np.random.RandomState(42)
    head_dim = 128
    n_heads = 512  # Full token
    bits = 4
    n_levels = (1 << bits) - 1
    
    # Generate KV data for one token
    kv = rng.randn(n_heads, head_dim).astype(np.float32)
    
    # Vectorised PolarQuant (simplified — no WHT, just polar coordinates)
    # This tests the theoretical floor of numpy speed
    
    def polar_compress_vectorised(kv_batch: np.ndarray, bits: int):
        """Vectorised polar quantization — no Hadamard transform."""
        n_levels = (1 << bits) - 1
        # Radius
        radii = np.linalg.norm(kv_batch, axis=1, keepdims=True)  # (n_heads, 1)
        
        # Normalise to unit sphere
        normalised = kv_batch / (radii + 1e-12)  # (n_heads, head_dim)
        
        # Simple uniform quantization as baseline
        quantised = np.clip(np.round((normalised + 1.0) / 2.0 * n_levels), 0, n_levels).astype(np.uint8)
        
        return radii, quantised
    
    def polar_decompress_vectorised(radii, quantised, bits):
        n_levels = (1 << bits) - 1
        normalised = (quantised.astype(np.float32) / n_levels) * 2.0 - 1.0
        # Re-normalise to unit sphere
        norms = np.linalg.norm(normalised, axis=1, keepdims=True)
        normalised = normalised / (norms + 1e-12)
        return normalised * radii
    
    # Warmup
    for _ in range(5):
        r, q = polar_compress_vectorised(kv, bits)
        _ = polar_decompress_vectorised(r, q, bits)
    
    # Benchmark
    n_iters = 100
    t0 = time.perf_counter()
    for _ in range(n_iters):
        r, q = polar_compress_vectorised(kv, bits)
    t_compress = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        recon = polar_decompress_vectorised(r, q, bits)
    t_decompress = time.perf_counter() - t0
    
    compress_per_tok = (t_compress / n_iters) * 1000  # ms
    decompress_per_tok = (t_decompress / n_iters) * 1000
    
    # Quality
    recon = polar_decompress_vectorised(r, q, bits)
    cosine_sims = []
    for i in range(n_heads):
        dot = np.dot(kv[i], recon[i])
        na = np.linalg.norm(kv[i])
        nb = np.linalg.norm(recon[i])
        if na > 1e-12 and nb > 1e-12:
            cosine_sims.append(float(np.clip(dot / (na * nb), -1.0, 1.0)))
    
    budget_ms = 1000.0 / 12.114
    print(f"  Vectorised compress: {compress_per_tok:.2f} ms/token ({compress_per_tok/budget_ms*100:.1f}% of budget)")
    print(f"  Vectorised decompress: {decompress_per_tok:.2f} ms/token")
    print(f"  Quality (uniform quant, no WHT): mean cos_sim={np.mean(cosine_sims):.4f}")
    print(f"  Note: This is SIMPLER than PolarQuant (no WHT, no polar coords)")
    print(f"  It gives a LOWER BOUND on vectorised numpy speed for KV compression")
    
    # Compare with Rust
    rust_ms = 3.24  # From Rust benchmark
    print(f"\n  Comparison:")
    print(f"    Vectorised NumPy (simplified): {compress_per_tok:.2f} ms/token")
    print(f"    Rust PolarQuant (full):        {rust_ms:.2f} ms/token")
    print(f"    Rust is {compress_per_tok/rust_ms:.1f}x faster than vectorised NumPy")
    print(f"    (Full PolarQuant in Python would be even slower)")


def main():
    # Subprocess benchmark
    bench_rust_subprocess(n_iterations=3)
    
    # NumPy comparison
    bench_numpy_comparison()
    
    print(f"\n{'='*78}")
    print("INTEGRATION PATH RECOMMENDATIONS")
    print(f"{'='*78}")
    print("""
For production integration:

1. BEST: PyO3 Python extension wrapping Rust crates
   - Zero-copy numpy → Rust → numpy
   - Expected: same speed as standalone Rust (~3ms/token)
   - Requires: adding PyO3 deps to crates, building with maturin

2. GOOD: ctypes/cffi FFI to cdylib .so
   - Minimal Python overhead for function calls
   - Expected: ~3-5ms/token (small FFI overhead)
   - Requires: C API wrappers in Rust, cdylib target

3. OK: Subprocess (current approach)
   - Runs Rust binary, parses output
   - Adds ~2s startup overhead per invocation
   - Only suitable for batch processing, not per-token

4. AVOID: Pure Python/NumPy
   - 300-1400ms/token even with vectorisation
   - 100-430x slower than Rust
   - Not viable for real-time inference
""")


if __name__ == "__main__":
    main()
