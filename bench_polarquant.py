"""
bench_polarquant.py — PolarQuant KV cache compression benchmark

Runs Qwen3.5-35B-A3B inference with Phase 4 best config and measures the
impact of PolarQuant KV cache compression on throughput, VRAM, and quality.

Experiment plan:
1. Baseline: q8_0 KV quant + flash attention (Phase 4 best: 10.2 tok/s)
2. PolarQuant 4-bit: compress/decompress KV cache via polar coordinate
   quantization on every attention step
3. PolarQuant 3-bit: more aggressive compression
4. Compare: tok/s, VRAM, compression ratio, output quality

Usage:
    uv run --with llama-cpp-python python bench_polarquant.py

Reference: arXiv:2502.02617 (AISTATS 2026)
"""

import os
import sys
import time
import json
import struct
import math
import numpy as np
from pathlib import Path
from datetime import datetime

# Add scripts to path for polar_kv
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from polar_kv import PolarKVCache, PolarKVCacheTorch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Phase 4 best config (baseline)
MODEL_DIR = Path("./models")
GGUF_PATTERN = "*.gguf"

# llama.cpp inference config
N_GPU_LAYERS = 10
N_BATCH = 32
N_UBATCH = 32
N_THREADS = 10
N_CTX = 512
FLASH_ATTN = True

# PolarQuant configs to test
POLAR_CONFIGS = [
    {"name": "baseline_q8_0", "kv_type": "q8_0", "polar": False},
    {"name": "polar_4bit",    "kv_type": "f16",  "polar": True, "n_bits": 4},
    {"name": "polar_3bit",    "kv_type": "f16",  "polar": True, "n_bits": 3},
    {"name": "polar_5bit",    "kv_type": "f16",  "polar": True, "n_bits": 5},
]

# Fixed prompt for reproducibility (same as harness)
PROMPT = """The following is a technical analysis of modern large language model architectures, focusing on the Mixture-of-Experts (MoE) approach and its implications for inference optimization on consumer hardware.

## Introduction

Large language models have grown dramatically in parameter count, with state-of-the-art models now exceeding hundreds of billions of parameters. However, the Mixture-of-Experts architecture offers a compelling solution: by activating only a subset of parameters per token, MoE models can maintain the quality of dense models while requiring significantly less compute per forward pass.

## Architecture Overview

The key innovation in sparse MoE models is the router network, which selects a subset of expert networks for each input token. In Qwen3.5-35B-A3B, the architecture uses"""

MAX_TOKENS = 128
SEED = 42


def find_model():
    """Find the GGUF model file."""
    for f in MODEL_DIR.glob(GGUF_PATTERN):
        if f.suffix == '.gguf':
            return str(f)
    # Check for common names
    candidates = [
        MODEL_DIR / "Qwen3.5-35B-A3B-Q3_K_M.gguf",
        MODEL_DIR / "qwen3.5-35b-a3b-q3_k_m.gguf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def get_vram_mb():
    """Get current GPU VRAM usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    
    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except (FileNotFoundError, ValueError):
        pass
    
    return 0.0


def run_baseline(model_path: str, config: dict) -> dict:
    """
    Run baseline inference with llama-cpp-python.
    Returns metrics dict.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed. Run: uv pip install llama-cpp-python")
        return {"error": "llama-cpp-python not installed"}
    
    kv_type = config.get("kv_type", "q8_0")
    
    print(f"\n{'='*60}")
    print(f"Config: {config['name']}")
    print(f"  KV type: {kv_type}, flash_attn: {FLASH_ATTN}")
    print(f"  n_gpu_layers: {N_GPU_LAYERS}, n_batch: {N_BATCH}")
    print(f"{'='*60}")
    
    vram_before = get_vram_mb()
    
    # Build llama.cpp kwargs
    kwargs = {
        "model_path": model_path,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
        "n_ubatch": N_UBATCH,
        "n_threads": N_THREADS,
        "n_ctx": N_CTX,
        "seed": SEED,
        "verbose": False,
    }
    
    # KV cache type — llama-cpp-python uses integer enum values
    # See llama_cpp.GGML_TYPE_* constants
    kv_type_map = {"q8_0": 8, "q4_0": 2, "q4_1": 3, "f16": 1, "f32": 0}
    if kv_type in kv_type_map:
        kwargs["type_k"] = kv_type_map[kv_type]
        kwargs["type_v"] = kv_type_map[kv_type]
    
    if FLASH_ATTN:
        kwargs["flash_attn"] = True
    
    load_start = time.perf_counter()
    model = Llama(**kwargs)
    load_time = time.perf_counter() - load_start
    
    vram_after_load = get_vram_mb()
    
    # Generate
    gen_start = time.perf_counter()
    output = model(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,  # deterministic
        top_p=1.0,
        seed=SEED,
    )
    gen_time = time.perf_counter() - gen_start
    
    vram_peak = get_vram_mb()
    
    # Extract metrics
    text = output['choices'][0]['text']
    n_tokens = output['usage']['completion_tokens']
    tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0
    
    # Simple quality metric: unique token ratio (diversity)
    words = text.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    result = {
        "config": config['name'],
        "tok_per_sec": round(tok_per_sec, 3),
        "n_tokens": n_tokens,
        "gen_time_s": round(gen_time, 2),
        "load_time_s": round(load_time, 2),
        "vram_before_mb": round(vram_before, 1),
        "vram_after_load_mb": round(vram_after_load, 1),
        "vram_peak_mb": round(vram_peak, 1),
        "text_preview": text[:200],
        "unique_word_ratio": round(unique_ratio, 3),
        "kv_type": kv_type,
    }
    
    print(f"  tok/s: {tok_per_sec:.3f}")
    print(f"  tokens: {n_tokens}")
    print(f"  gen_time: {gen_time:.2f}s")
    print(f"  VRAM peak: {vram_peak:.1f} MB")
    print(f"  Output preview: {text[:100]}...")
    
    del model
    
    return result


def run_polar_experiment(model_path: str, config: dict) -> dict:
    """
    Run inference with PolarQuant KV cache compression applied post-hoc.
    
    Strategy: We can't hook into llama.cpp's internal KV cache directly from
    Python. Instead, we:
    1. Run baseline inference to get the output (quality reference)
    2. Simulate PolarQuant compression on synthetic KV vectors matching
       the model's head dimensions
    3. Measure compression overhead (encode/decode time per step)
    4. Report projected tok/s impact
    
    For actual integration, PolarQuant would be implemented as a C++ layer
    inside llama.cpp's KV cache management.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed")
        return {"error": "llama-cpp-python not installed"}
    
    n_bits = config.get("n_bits", 4)
    
    print(f"\n{'='*60}")
    print(f"Config: {config['name']} (PolarQuant {n_bits}-bit)")
    print(f"{'='*60}")
    
    # Qwen3.5-35B-A3B architecture:
    # - 40 layers, 8 KV heads, head_dim = 128
    N_LAYERS = 40
    N_KV_HEADS = 8
    HEAD_DIM = 128
    
    # Initialize PolarQuant cache
    try:
        import torch
        polar = PolarKVCacheTorch(head_dim=HEAD_DIM, n_bits=n_bits, seed=SEED, device='cpu')
        use_torch = True
    except ImportError:
        polar = PolarKVCache(head_dim=HEAD_DIM, n_bits=n_bits, seed=SEED)
        use_torch = False
    
    # Step 1: Measure PolarQuant encode/decode latency on realistic KV shapes
    # Per token: compress 2 (K+V) × N_LAYERS × N_KV_HEADS × HEAD_DIM vectors
    n_vectors_per_token = 2 * N_LAYERS * N_KV_HEADS  # 640 vectors
    
    rng = np.random.RandomState(SEED)
    
    # Warm up
    if use_torch:
        import torch
        test_kv = torch.randn(n_vectors_per_token, HEAD_DIM)
        for _ in range(3):
            r, q = polar.compress_tensor(test_kv)
            _ = polar.decompress_tensor(r, q)
    else:
        test_kv = rng.randn(n_vectors_per_token, HEAD_DIM).astype(np.float32)
        for _ in range(3):
            packed = polar.compress_batch(test_kv)
            _ = polar.decompress_batch(packed)
    
    # Benchmark encode
    n_bench = 20
    encode_times = []
    decode_times = []
    
    for _ in range(n_bench):
        if use_torch:
            kv = torch.randn(n_vectors_per_token, HEAD_DIM)
            t0 = time.perf_counter()
            r, q = polar.compress_tensor(kv)
            encode_times.append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            _ = polar.decompress_tensor(r, q)
            decode_times.append(time.perf_counter() - t0)
        else:
            kv = rng.randn(n_vectors_per_token, HEAD_DIM).astype(np.float32)
            t0 = time.perf_counter()
            packed = polar.compress_batch(kv)
            encode_times.append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            _ = polar.decompress_batch(packed)
            decode_times.append(time.perf_counter() - t0)
    
    avg_encode_ms = np.mean(encode_times) * 1000
    avg_decode_ms = np.mean(decode_times) * 1000
    total_overhead_ms = avg_encode_ms + avg_decode_ms
    
    # Step 2: Run actual inference for quality comparison
    kwargs = {
        "model_path": model_path,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
        "n_ubatch": N_UBATCH,
        "n_threads": N_THREADS,
        "n_ctx": N_CTX,
        "seed": SEED,
        "verbose": False,
        "flash_attn": FLASH_ATTN,
    }
    
    vram_before = get_vram_mb()
    model = Llama(**kwargs)
    
    gen_start = time.perf_counter()
    output = model(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
    )
    gen_time = time.perf_counter() - gen_start
    vram_peak = get_vram_mb()
    
    text = output['choices'][0]['text']
    n_tokens = output['usage']['completion_tokens']
    baseline_tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0
    
    # Step 3: Project adjusted tok/s with PolarQuant overhead
    # Per-token time = 1/baseline_tok_per_sec + overhead_s
    baseline_per_token_ms = 1000.0 / baseline_tok_per_sec if baseline_tok_per_sec > 0 else float('inf')
    adjusted_per_token_ms = baseline_per_token_ms + total_overhead_ms
    projected_tok_per_sec = 1000.0 / adjusted_per_token_ms if adjusted_per_token_ms > 0 else 0
    
    # Step 4: Compression metrics
    if use_torch:
        np_cache = PolarKVCache(head_dim=HEAD_DIM, n_bits=n_bits, seed=SEED)
    else:
        np_cache = polar
    compression_ratio = np_cache.compression_ratio()
    
    # VRAM savings from KV cache compression
    # f16 KV cache at n_ctx=512: 2 × 40 × 8 × 512 × 128 × 2 bytes = 167MB
    kv_cache_f16_mb = 2 * N_LAYERS * N_KV_HEADS * N_CTX * HEAD_DIM * 2 / (1024 * 1024)
    kv_cache_polar_mb = kv_cache_f16_mb / compression_ratio
    vram_savings_mb = kv_cache_f16_mb - kv_cache_polar_mb
    
    words = text.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    result = {
        "config": config['name'],
        "n_bits": n_bits,
        "compression_ratio": round(compression_ratio, 2),
        "baseline_tok_per_sec": round(baseline_tok_per_sec, 3),
        "projected_tok_per_sec": round(projected_tok_per_sec, 3),
        "encode_ms_per_token": round(avg_encode_ms, 2),
        "decode_ms_per_token": round(avg_decode_ms, 2),
        "total_overhead_ms": round(total_overhead_ms, 2),
        "n_tokens": n_tokens,
        "gen_time_s": round(gen_time, 2),
        "vram_peak_mb": round(vram_peak, 1),
        "kv_cache_f16_mb": round(kv_cache_f16_mb, 1),
        "kv_cache_polar_mb": round(kv_cache_polar_mb, 1),
        "vram_savings_mb": round(vram_savings_mb, 1),
        "text_preview": text[:200],
        "unique_word_ratio": round(unique_ratio, 3),
        "use_torch": use_torch,
        "n_vectors_per_token": n_vectors_per_token,
    }
    
    print(f"  Compression ratio: {compression_ratio:.2f}×")
    print(f"  Encode latency: {avg_encode_ms:.2f} ms/token ({n_vectors_per_token} vectors)")
    print(f"  Decode latency: {avg_decode_ms:.2f} ms/token")
    print(f"  Total overhead: {total_overhead_ms:.2f} ms/token")
    print(f"  Baseline tok/s: {baseline_tok_per_sec:.3f}")
    print(f"  Projected tok/s: {projected_tok_per_sec:.3f}")
    print(f"  KV cache: {kv_cache_f16_mb:.1f} MB (f16) → {kv_cache_polar_mb:.1f} MB (polar)")
    print(f"  VRAM savings: {vram_savings_mb:.1f} MB")
    print(f"  Output preview: {text[:100]}...")
    
    del model
    
    return result


def run_polar_standalone_benchmark():
    """
    Standalone PolarQuant benchmark — no model required.
    Measures compression quality and throughput on synthetic KV vectors.
    """
    print("\n" + "=" * 60)
    print("PolarQuant Standalone Benchmark (no model required)")
    print("=" * 60)
    
    HEAD_DIM = 128
    N_LAYERS = 40
    N_KV_HEADS = 8
    N_VECTORS = 2 * N_LAYERS * N_KV_HEADS  # 640 per token
    
    results = []
    rng = np.random.RandomState(SEED)
    
    for n_bits in [3, 4, 5, 8]:
        cache = PolarKVCache(head_dim=HEAD_DIM, n_bits=n_bits, seed=SEED)
        
        # Generate realistic KV vectors (Gaussian, std ~1)
        vectors = rng.randn(N_VECTORS, HEAD_DIM).astype(np.float32)
        
        # Compression quality
        cos_sims = []
        for i in range(min(100, N_VECTORS)):
            packed = cache.compress(vectors[i])
            recon = cache.decompress(packed)
            cs = np.dot(vectors[i], recon) / (np.linalg.norm(vectors[i]) * np.linalg.norm(recon) + 1e-30)
            cos_sims.append(cs)
        
        mean_cos = np.mean(cos_sims)
        
        # Throughput
        t0 = time.perf_counter()
        packed_all = cache.compress_batch(vectors[:100])
        encode_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        _ = cache.decompress_batch(packed_all)
        decode_time = time.perf_counter() - t0
        
        ratio = cache.compression_ratio()
        bytes_per_vec = cache.compressed_bytes_per_vector()
        
        r = {
            "n_bits": n_bits,
            "compression_ratio": round(ratio, 2),
            "mean_cosine_sim": round(float(mean_cos), 4),
            "encode_ms_per_100vec": round(encode_time * 1000, 2),
            "decode_ms_per_100vec": round(decode_time * 1000, 2),
            "bytes_per_vector": bytes_per_vec,
            "fp16_bytes_per_vector": HEAD_DIM * 2,
        }
        results.append(r)
        
        print(f"\n  {n_bits}-bit angles:")
        print(f"    Compression ratio: {ratio:.2f}×")
        print(f"    Mean cosine similarity: {mean_cos:.4f}")
        print(f"    Encode: {encode_time*1000:.2f} ms / 100 vectors")
        print(f"    Decode: {decode_time*1000:.2f} ms / 100 vectors")
        print(f"    Size: {bytes_per_vec} bytes vs {HEAD_DIM*2} bytes (fp16)")
    
    # Angle distribution validation
    print(f"\n  Angle Distribution (PolarQuant insight validation):")
    cache_test = PolarKVCache(head_dim=HEAD_DIM, n_bits=4, seed=SEED)
    dist = cache_test.measure_angle_distribution(rng.randn(500, HEAD_DIM).astype(np.float32))
    print(f"    Mean: {dist['mean']:.4f} (expected π/2 = {dist['expected_mean']:.4f})")
    print(f"    Std:  {dist['std']:.4f} (expected 1/√d = {dist['expected_std']:.4f})")
    print(f"    Concentration quality: {dist['concentration_quality']:.2f}× theoretical")
    
    # Torch backend comparison
    try:
        import torch
        print(f"\n  Torch Backend Comparison:")
        tcache = PolarKVCacheTorch(head_dim=HEAD_DIM, n_bits=4, seed=SEED, device='cpu')
        tvec = torch.randn(N_VECTORS, HEAD_DIM)
        
        t0 = time.perf_counter()
        r, q = tcache.compress_tensor(tvec)
        torch_encode = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        _ = tcache.decompress_tensor(r, q)
        torch_decode = time.perf_counter() - t0
        
        print(f"    Encode: {torch_encode*1000:.2f} ms / {N_VECTORS} vectors")
        print(f"    Decode: {torch_decode*1000:.2f} ms / {N_VECTORS} vectors")
        print(f"    vs numpy: {encode_time*1000:.2f} / {decode_time*1000:.2f} ms per 100 vectors")
    except ImportError:
        print(f"\n  Torch not available, skipping GPU benchmark")
    
    return results


def main():
    print(f"PolarQuant KV Cache Compression Benchmark")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Reference: arXiv:2502.02617 (AISTATS 2026)")
    
    model_path = find_model()
    
    all_results = {}
    
    # Always run standalone benchmark (no model needed)
    standalone = run_polar_standalone_benchmark()
    all_results["standalone"] = standalone
    
    if model_path:
        print(f"\nModel found: {model_path}")
        
        # Run each config
        inference_results = []
        for config in POLAR_CONFIGS:
            try:
                if config.get("polar"):
                    result = run_polar_experiment(model_path, config)
                else:
                    result = run_baseline(model_path, config)
                inference_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                inference_results.append({"config": config["name"], "error": str(e)})
        
        all_results["inference"] = inference_results
        
        # Summary table
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"{'Config':<20} {'tok/s':<10} {'Ratio':<8} {'VRAM':<10}")
        print(f"{'-'*48}")
        for r in inference_results:
            if "error" in r:
                print(f"{r['config']:<20} {'ERROR':<10}")
                continue
            tok_s = r.get('projected_tok_per_sec') or r.get('tok_per_sec', 0)
            ratio = r.get('compression_ratio', '-')
            vram = r.get('vram_peak_mb', 0)
            print(f"{r['config']:<20} {tok_s:<10.3f} {str(ratio):<8} {vram:<10.1f}")
    else:
        print(f"\nNo GGUF model found in {MODEL_DIR}/")
        print(f"Run: uv run python scripts/download_model.py")
        print(f"Standalone benchmark completed above.")
    
    # Save results
    results_path = Path("results") / f"polarquant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
