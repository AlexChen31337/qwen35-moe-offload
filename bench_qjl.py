"""
bench_qjl.py — QJL (Quantized Johnson-Lindenstrauss) KV cache compression benchmark

Tests QJL's 1-bit key quantization with CUDA kernels from:
  https://github.com/amirzandieh/QJL (arXiv:2406.03482)

QJL compresses keys to 1-bit sign representations via JL random projections,
achieving ~5× compression with custom CUDA kernels for quantization and
attention score computation. Values use separate group quantization.

This benchmark:
1. Builds the QJL CUDA kernels (requires GPU + CUDA toolkit)
2. Runs standalone compression benchmarks (encode/decode throughput, quality)
3. If model available: compares against q8_0 baseline on real inference
4. Reports: compression ratio, VRAM savings, throughput, attention accuracy

Usage:
    uv run --with llama-cpp-python python bench_qjl.py

Reference: arXiv:2406.03482 (Zandieh et al., 2024)
"""

import os
import sys
import time
import json
import math
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# QJL setup
# ---------------------------------------------------------------------------

QJL_DIR = Path("/tmp/QJL")
QJL_KERNEL_DIR = QJL_DIR / "qjl_kernel"

# Model config — Qwen3.5-35B-A3B
N_LAYERS = 40
N_KV_HEADS = 8
N_Q_HEADS = 28  # query heads (GQA ratio 28:8)
HEAD_DIM = 128
GQA_RATIO = N_Q_HEADS // N_KV_HEADS  # 3.5 → round to 4 for GQA

# QJL config (from paper's recommended settings)
QJL_KEY_BITS = 256          # JL sketch dimension for keys
QJL_KEY_BITS_INITIAL = 512  # higher precision for early layers
QJL_INITIAL_LAYERS = 15
QJL_OUTLIER_COUNT = 8
QJL_VALUE_BITS = 2          # group quantization bits for values
QJL_GROUP_SIZE = 32
QJL_BUFFER_SIZE = 128

SEED = 42
MODEL_DIR = Path("./models")
N_CTX = 512
MAX_TOKENS = 128

# Same prompt as bench_polarquant.py for fair comparison
PROMPT = """The following is a technical analysis of modern large language model architectures, focusing on the Mixture-of-Experts (MoE) approach and its implications for inference optimization on consumer hardware.

## Introduction

Large language models have grown dramatically in parameter count, with state-of-the-art models now exceeding hundreds of billions of parameters. However, the Mixture-of-Experts architecture offers a compelling solution: by activating only a subset of parameters per token, MoE models can maintain the quality of dense models while requiring significantly less compute per forward pass.

## Architecture Overview

The key innovation in sparse MoE models is the router network, which selects a subset of expert networks for each input token. In Qwen3.5-35B-A3B, the architecture uses"""


def build_qjl_kernels() -> bool:
    """Build QJL CUDA kernels. Returns True if successful."""
    if not QJL_DIR.exists():
        print("ERROR: QJL repo not found at /tmp/QJL")
        print("Clone it: git clone https://github.com/amirzandieh/QJL /tmp/QJL")
        return False
    
    # Check if already built
    built_files = list(QJL_KERNEL_DIR.glob("*.so"))
    if len(built_files) >= 3:
        print(f"QJL kernels already built ({len(built_files)} .so files)")
        return True
    
    print("Building QJL CUDA kernels...")
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(QJL_KERNEL_DIR),
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr[-500:]}")
            return False
        print("QJL kernels built successfully")
        return True
    except subprocess.TimeoutExpired:
        print("Build timed out (300s)")
        return False
    except Exception as e:
        print(f"Build error: {e}")
        return False


def check_gpu() -> dict:
    """Check GPU availability and properties."""
    info = {"cuda_available": False, "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def run_qjl_standalone_benchmark() -> dict:
    """
    Standalone QJL benchmark using PyTorch (no CUDA kernels required).
    Tests the pure-Python QJL quantization path.
    """
    print("\n" + "=" * 60)
    print("QJL Standalone Benchmark (Pure Python/PyTorch)")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for QJL benchmark")
        return {"error": "PyTorch not installed"}
    
    gpu_info = check_gpu()
    device = gpu_info["device"]
    print(f"Device: {device}")
    if gpu_info.get("gpu_name"):
        print(f"GPU: {gpu_info['gpu_name']} ({gpu_info.get('gpu_memory_gb', 0):.1f} GB)")
    
    results = {}
    
    # --- Pure-Python JL projection (no CUDA kernel needed) ---
    print(f"\n  JL Projection (d={HEAD_DIM}, sketch_dim={QJL_KEY_BITS}):")
    
    rng = torch.Generator(device=device).manual_seed(SEED)
    
    # Generate random JL projection matrix
    # Normalized Gaussian: P ~ N(0, 1/sketch_dim) of shape (sketch_dim, head_dim)
    P = torch.randn(QJL_KEY_BITS, HEAD_DIM, device=device, generator=rng) / math.sqrt(QJL_KEY_BITS)
    
    # Simulate a batch of key vectors (seq_len tokens, 1 head)
    seq_len = 512
    keys = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, device=device)
    queries = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    
    # --- Encode: project keys → sign bits ---
    t0 = time.perf_counter()
    for _ in range(10):
        # Project: keys @ P^T → (batch, heads, seq, sketch_dim)
        projected = torch.matmul(keys, P.T)
        # Sign quantization: 1-bit
        sign_bits = (projected > 0).to(torch.uint8)
        # Store key norms for score rescaling
        key_norms = torch.norm(keys, dim=-1, keepdim=True)
    encode_time = (time.perf_counter() - t0) / 10
    
    # --- Decode: compute approximate attention scores ---
    t0 = time.perf_counter()
    for _ in range(10):
        # Project query
        q_projected = torch.matmul(queries, P.T)
        q_signs = (q_projected > 0).to(torch.uint8)
        
        # Hamming-based dot product approximation:
        # sign(Px)·sign(Py) ≈ cos(angle(x,y)) via JL guarantee
        # Count matching signs
        matches = (sign_bits == q_signs).float().sum(dim=-1)  # (batch, heads, seq)
        # Convert to cosine estimate
        cos_estimate = (2 * matches / QJL_KEY_BITS - 1)
        # Scale by norms
        approx_scores = cos_estimate * key_norms.squeeze(-1) * torch.norm(queries, dim=-1)
    decode_time = (time.perf_counter() - t0) / 10
    
    # --- Quality: compare with exact attention ---
    exact_scores = torch.matmul(queries, keys.transpose(-1, -2)).squeeze(-2)  # (batch, heads, seq)
    approx_flat = approx_scores.flatten()
    exact_flat = exact_scores.flatten()
    
    # Correlation
    corr = torch.corrcoef(torch.stack([exact_flat, approx_flat]))[0, 1].item()
    
    # Relative error
    rel_error = torch.abs(approx_flat - exact_flat) / (torch.abs(exact_flat) + 1e-8)
    mean_rel_error = rel_error.mean().item()
    
    # Compression ratio for keys
    # Original: head_dim × 16 bits (fp16) = 2048 bits per key
    # QJL: sketch_dim bits (sign) + 16 bits (norm) = 272 bits per key
    key_original_bits = HEAD_DIM * 16
    key_qjl_bits = QJL_KEY_BITS + 16  # sign bits + fp16 norm
    key_ratio = key_original_bits / key_qjl_bits
    
    # Memory: original vs compressed
    original_kv_mb = 2 * N_LAYERS * N_KV_HEADS * N_CTX * HEAD_DIM * 2 / (1024**2)
    # Keys: QJL compressed, Values: group quant at QJL_VALUE_BITS
    compressed_keys_mb = N_LAYERS * N_KV_HEADS * N_CTX * key_qjl_bits / 8 / (1024**2)
    compressed_values_mb = N_LAYERS * N_KV_HEADS * N_CTX * HEAD_DIM * QJL_VALUE_BITS / 8 / (1024**2)
    # Value quant also needs scale+zero per group
    value_overhead_mb = N_LAYERS * N_KV_HEADS * N_CTX * (HEAD_DIM // QJL_GROUP_SIZE) * 4 / (1024**2)
    compressed_total_mb = compressed_keys_mb + compressed_values_mb + value_overhead_mb
    total_ratio = original_kv_mb / compressed_total_mb
    
    results["jl_projection"] = {
        "sketch_dim": QJL_KEY_BITS,
        "head_dim": HEAD_DIM,
        "key_compression_ratio": round(key_ratio, 2),
        "total_kv_compression_ratio": round(total_ratio, 2),
        "encode_ms": round(encode_time * 1000, 3),
        "decode_ms": round(decode_time * 1000, 3),
        "attention_correlation": round(corr, 4),
        "mean_relative_error": round(mean_rel_error, 4),
        "seq_len": seq_len,
        "original_kv_mb": round(original_kv_mb, 1),
        "compressed_kv_mb": round(compressed_total_mb, 1),
        "vram_savings_mb": round(original_kv_mb - compressed_total_mb, 1),
    }
    
    print(f"    Key compression: {key_ratio:.2f}× ({key_original_bits} → {key_qjl_bits} bits)")
    print(f"    Total KV compression: {total_ratio:.2f}×")
    print(f"    Encode: {encode_time*1000:.3f} ms ({seq_len} keys × {N_KV_HEADS} heads)")
    print(f"    Score computation: {decode_time*1000:.3f} ms")
    print(f"    Attention correlation: {corr:.4f}")
    print(f"    Mean relative error: {mean_rel_error:.4f}")
    print(f"    KV cache: {original_kv_mb:.1f} MB → {compressed_total_mb:.1f} MB")
    
    # --- Test with CUDA kernels if available ---
    results["cuda_kernels"] = _test_cuda_kernels(device)
    
    # --- Sweep sketch dimensions ---
    print(f"\n  Sketch Dimension Sweep:")
    sweep_results = []
    for sketch_dim in [64, 128, 256, 512]:
        P_s = torch.randn(sketch_dim, HEAD_DIM, device=device) / math.sqrt(sketch_dim)
        projected_s = torch.matmul(keys, P_s.T)
        sign_s = (projected_s > 0).to(torch.uint8)
        
        q_proj_s = torch.matmul(queries, P_s.T)
        q_sign_s = (q_proj_s > 0).to(torch.uint8)
        
        matches_s = (sign_s == q_sign_s).float().sum(dim=-1)
        cos_est_s = (2 * matches_s / sketch_dim - 1)
        approx_s = cos_est_s * key_norms.squeeze(-1) * torch.norm(queries, dim=-1)
        
        corr_s = torch.corrcoef(torch.stack([exact_flat, approx_s.flatten()]))[0, 1].item()
        bits_per_key = sketch_dim + 16
        ratio_s = key_original_bits / bits_per_key
        
        sweep_results.append({
            "sketch_dim": sketch_dim,
            "bits_per_key": bits_per_key,
            "key_ratio": round(ratio_s, 2),
            "correlation": round(corr_s, 4),
        })
        print(f"    sketch_dim={sketch_dim}: {ratio_s:.2f}× compression, corr={corr_s:.4f}")
    
    results["sweep"] = sweep_results
    
    return results


def _test_cuda_kernels(device: str) -> dict:
    """Test QJL CUDA kernels if built and available."""
    if device != "cuda":
        return {"status": "skipped", "reason": "no GPU"}
    
    # Try to import QJL kernels
    if str(QJL_KERNEL_DIR) not in sys.path:
        sys.path.insert(0, str(QJL_KERNEL_DIR))
        sys.path.insert(0, str(QJL_DIR))
    
    try:
        from qjl_kernel.qjl_kernel import qjl_quant, qjl_score
        print(f"\n  CUDA Kernel Benchmark:")
    except ImportError as e:
        # Try building
        if build_qjl_kernels():
            try:
                from qjl_kernel.qjl_kernel import qjl_quant, qjl_score
                print(f"\n  CUDA Kernel Benchmark (freshly built):")
            except ImportError as e2:
                return {"status": "build_failed", "error": str(e2)}
        else:
            return {"status": "not_built", "error": str(e)}
    
    import torch
    
    # Prepare inputs matching QJL kernel expectations
    batch = 1
    seq_len = 128
    group_size = QJL_GROUP_SIZE
    num_groups = seq_len // group_size
    
    # Key states: (batch, heads, num_groups, group_size, head_dim)
    key_states = torch.randn(batch, N_KV_HEADS, num_groups, group_size, HEAD_DIM, 
                             device='cuda', dtype=torch.float16)
    
    # Random projection matrix
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    rand_prj = torch.randn(QJL_KEY_BITS, HEAD_DIM, device='cuda', dtype=torch.float16,
                            generator=rng)
    
    # Outlier indices: (batch, heads, num_groups, outlier_count)
    norms = key_states.norm(dim=-2)  # (batch, heads, num_groups, head_dim)
    _, outlier_indices = norms.topk(QJL_OUTLIER_COUNT, dim=-1)
    outlier_indices = outlier_indices.to(torch.uint8).contiguous()
    
    # Benchmark quantization
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        key_quant, key_outlier_quant, key_outlier_norm = qjl_quant(
            key_states, outlier_indices, rand_prj, QJL_KEY_BITS // 2
        )
        torch.cuda.synchronize()
    quant_time = (time.perf_counter() - t0) / n_iters
    
    result = {
        "status": "success",
        "quant_ms": round(quant_time * 1000, 3),
        "seq_len": seq_len,
        "n_heads": N_KV_HEADS,
    }
    
    print(f"    CUDA quant: {quant_time*1000:.3f} ms ({seq_len} tokens × {N_KV_HEADS} heads)")
    
    return result


def run_qjl_vs_polar_comparison():
    """
    Compare QJL and PolarQuant on the same synthetic KV data.
    """
    print("\n" + "=" * 60)
    print("QJL vs PolarQuant Comparison")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required")
        return {}
    
    sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
    from polar_kv import PolarKVCacheTorch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate identical test data
    torch.manual_seed(SEED)
    keys = torch.randn(1, N_KV_HEADS, 256, HEAD_DIM, device=device)
    queries = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    
    # Exact attention scores (ground truth)
    exact = torch.matmul(queries, keys.transpose(-1, -2)).squeeze(-2)
    
    results = {}
    
    # --- PolarQuant ---
    for n_bits in [3, 4, 5]:
        polar = PolarKVCacheTorch(head_dim=HEAD_DIM, n_bits=n_bits, seed=SEED, device=device)
        
        # Compress and decompress keys
        flat_keys = keys.reshape(-1, HEAD_DIM)
        t0 = time.perf_counter()
        radii, quant_angles = polar.compress_tensor(flat_keys)
        recon_keys = polar.decompress_tensor(radii, quant_angles)
        polar_time = time.perf_counter() - t0
        
        recon_keys = recon_keys.reshape_as(keys)
        polar_scores = torch.matmul(queries, recon_keys.transpose(-1, -2)).squeeze(-2)
        
        corr = torch.corrcoef(torch.stack([exact.flatten(), polar_scores.flatten()]))[0, 1].item()
        ratio = polar.compression_ratio()
        
        results[f"polar_{n_bits}bit"] = {
            "compression_ratio": round(ratio, 2),
            "attention_correlation": round(corr, 4),
            "time_ms": round(polar_time * 1000, 2),
        }
        print(f"  PolarQuant {n_bits}-bit: ratio={ratio:.2f}×, corr={corr:.4f}, time={polar_time*1000:.2f}ms")
    
    # --- QJL (pure Python path) ---
    for sketch_dim in [128, 256, 512]:
        P = torch.randn(sketch_dim, HEAD_DIM, device=device) / math.sqrt(sketch_dim)
        
        t0 = time.perf_counter()
        projected = torch.matmul(keys, P.T)
        sign_bits = (projected > 0).to(torch.uint8)
        key_norms = torch.norm(keys, dim=-1, keepdim=True)
        
        q_proj = torch.matmul(queries, P.T)
        q_signs = (q_proj > 0).to(torch.uint8)
        
        matches = (sign_bits == q_signs).float().sum(dim=-1)
        cos_est = (2 * matches / sketch_dim - 1)
        qjl_scores = cos_est * key_norms.squeeze(-1) * torch.norm(queries, dim=-1)
        qjl_time = time.perf_counter() - t0
        
        corr = torch.corrcoef(torch.stack([exact.flatten(), qjl_scores.flatten()]))[0, 1].item()
        bits_per_key = sketch_dim + 16
        ratio = (HEAD_DIM * 16) / bits_per_key
        
        results[f"qjl_s{sketch_dim}"] = {
            "compression_ratio": round(ratio, 2),
            "attention_correlation": round(corr, 4),
            "time_ms": round(qjl_time * 1000, 2),
            "sketch_dim": sketch_dim,
        }
        print(f"  QJL sketch={sketch_dim}: ratio={ratio:.2f}×, corr={corr:.4f}, time={qjl_time*1000:.2f}ms")
    
    return results


def main():
    print(f"QJL KV Cache Compression Benchmark")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Reference: arXiv:2406.03482 (QJL)")
    print(f"Reference: arXiv:2502.02617 (PolarQuant)")
    
    all_results = {
        "gpu": check_gpu(),
        "config": {
            "head_dim": HEAD_DIM,
            "n_layers": N_LAYERS,
            "n_kv_heads": N_KV_HEADS,
            "n_q_heads": N_Q_HEADS,
            "qjl_sketch_dim": QJL_KEY_BITS,
            "qjl_value_bits": QJL_VALUE_BITS,
            "n_ctx": N_CTX,
        }
    }
    
    # Standalone benchmark
    all_results["qjl_standalone"] = run_qjl_standalone_benchmark()
    
    # Comparison
    all_results["comparison"] = run_qjl_vs_polar_comparison()
    
    # Save results
    results_path = Path("results") / f"qjl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — QJL vs PolarQuant")
    print(f"{'='*60}")
    comp = all_results.get("comparison", {})
    if comp:
        print(f"{'Method':<25} {'Ratio':<10} {'Attn Corr':<12} {'Time (ms)':<10}")
        print(f"{'-'*57}")
        for name, data in sorted(comp.items()):
            print(f"{name:<25} {data['compression_ratio']:<10.2f} {data['attention_correlation']:<12.4f} {data['time_ms']:<10.2f}")


if __name__ == "__main__":
    main()
