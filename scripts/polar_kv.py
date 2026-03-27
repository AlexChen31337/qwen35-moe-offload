"""
polar_kv.py — PolarQuant KV cache compression primitives (Python reference).

Pure-Python/NumPy implementation of PolarQuant (arXiv:2502.02617) for
benchmarking and quality analysis. The Rust crate (crates/polarquant) is
the production implementation; this is for rapid prototyping and comparison.

Implements:
  1. Randomised Walsh-Hadamard Transform (WHT) preconditioner
  2. Cartesian → polar coordinate conversion
  3. Angle quantization (1–8 bits)
  4. Round-trip reconstruction + quality measurement
"""

import numpy as np
from typing import Tuple


# ── Hadamard Preconditioner ──────────────────────────────────────────────────

def _hadamard_matrix(n: int) -> np.ndarray:
    """Generate normalised Hadamard-like matrix for dimension n (power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    h = _hadamard_matrix(n // 2)
    return np.block([[h, h], [h, -h]]) / np.sqrt(2)


def _random_signs(d: int, seed: int) -> np.ndarray:
    """Random ±1 diagonal for randomised WHT."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=d).astype(np.float32)


def hadamard_precondition(kv: np.ndarray, seed: int = 0xDEADBEEF) -> np.ndarray:
    """Apply randomised Hadamard preconditioner to KV vector(s).
    
    Args:
        kv: shape (d,) or (n_heads, d) — KV cache vectors
        seed: deterministic seed for random signs
    
    Returns:
        Preconditioned vectors, same shape as input.
    """
    if kv.ndim == 1:
        d = kv.shape[0]
        # Pad to power of 2 if needed
        d_padded = 1 << (d - 1).bit_length()
        if d_padded != d:
            padded = np.zeros(d_padded, dtype=np.float32)
            padded[:d] = kv
        else:
            padded = kv.copy()
        signs = _random_signs(d_padded, seed)
        H = _hadamard_matrix(d_padded)
        result = H @ (signs * padded)
        return result[:d]
    else:
        # Batch mode
        return np.stack([
            hadamard_precondition(kv[i], seed ^ i)
            for i in range(kv.shape[0])
        ])


def hadamard_precondition_inv(kv: np.ndarray, seed: int = 0xDEADBEEF) -> np.ndarray:
    """Inverse Hadamard preconditioner."""
    if kv.ndim == 1:
        d = kv.shape[0]
        d_padded = 1 << (d - 1).bit_length()
        if d_padded != d:
            padded = np.zeros(d_padded, dtype=np.float32)
            padded[:d] = kv
        else:
            padded = kv.copy()
        signs = _random_signs(d_padded, seed)
        H = _hadamard_matrix(d_padded)
        # H is orthogonal and symmetric → H^-1 = H^T = H
        result = signs * (H @ padded)
        return result[:d]
    else:
        return np.stack([
            hadamard_precondition_inv(kv[i], seed ^ i)
            for i in range(kv.shape[0])
        ])


# ── Polar Coordinate Conversion ─────────────────────────────────────────────

def cartesian_to_polar(v: np.ndarray) -> Tuple[float, np.ndarray]:
    """Convert Cartesian vector to polar (spherical) coordinates.
    
    Returns:
        (radius, angles) where angles has length d-1 and each angle ∈ [0, π]
        except the last which is ∈ [0, 2π].
    """
    d = len(v)
    radius = np.linalg.norm(v).item()
    if radius < 1e-12:
        return 0.0, np.zeros(d - 1, dtype=np.float32)
    
    angles = np.zeros(d - 1, dtype=np.float32)
    for i in range(d - 1):
        # Sum of squares from this dimension onwards
        tail_norm = np.sqrt(np.sum(v[i:] ** 2))
        if tail_norm < 1e-12:
            break
        cos_angle = np.clip(v[i] / tail_norm, -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    
    # Last angle needs atan2 for full [0, 2π] range
    if d >= 2 and np.abs(v[-2]) + np.abs(v[-1]) > 1e-12:
        angles[-1] = np.arctan2(v[-1], v[-2])
        if angles[-1] < 0:
            angles[-1] += 2 * np.pi
    
    return radius, angles


def polar_to_cartesian(radius: float, angles: np.ndarray) -> np.ndarray:
    """Convert polar (spherical) coordinates back to Cartesian."""
    d = len(angles) + 1
    v = np.zeros(d, dtype=np.float32)
    
    if radius < 1e-12:
        return v
    
    # Standard n-sphere parametrisation
    for i in range(d):
        val = radius
        for j in range(i):
            val *= np.sin(angles[j])
        if i < d - 1:
            val *= np.cos(angles[i])
        v[i] = val
    
    return v


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_angles(angles: np.ndarray, bits: int) -> np.ndarray:
    """Quantize angles to `bits` bits each.
    
    All angles except the last are in [0, π].
    The last angle is in [0, 2π].
    """
    n_levels = (1 << bits) - 1  # e.g., 15 for 4-bit
    q = np.zeros(len(angles), dtype=np.uint8)
    
    for i in range(len(angles)):
        max_val = 2 * np.pi if i == len(angles) - 1 else np.pi
        # Normalise to [0, 1], then quantize
        normalised = np.clip(angles[i] / max_val, 0.0, 1.0)
        q[i] = int(round(normalised * n_levels))
    
    return q


def dequantize_angles(q: np.ndarray, bits: int) -> np.ndarray:
    """Dequantize angle codes back to angles."""
    n_levels = (1 << bits) - 1
    angles = np.zeros(len(q), dtype=np.float32)
    
    for i in range(len(q)):
        max_val = 2 * np.pi if i == len(q) - 1 else np.pi
        angles[i] = (q[i] / n_levels) * max_val
    
    return angles


# ── Full Pipeline ────────────────────────────────────────────────────────────

def polarquant_compress(kv: np.ndarray, bits: int = 4, seed: int = 0xDEADBEEF) -> dict:
    """Compress a KV head vector using PolarQuant.
    
    Args:
        kv: shape (d,) — single head's KV vector (float32)
        bits: quantization bits per angle (1–8)
        seed: Hadamard preconditioner seed
    
    Returns:
        dict with 'radius' (float32), 'angles' (uint8 codes), 'bits', 'head_dim'
    """
    # Step 1: Hadamard precondition
    pre = hadamard_precondition(kv, seed)
    
    # Step 2: Cartesian → Polar
    radius, angles = cartesian_to_polar(pre)
    
    # Step 3: Quantize angles
    q_angles = quantize_angles(angles, bits)
    
    return {
        'radius': np.float32(radius),
        'angles': q_angles,
        'bits': bits,
        'head_dim': len(kv),
    }


def polarquant_decompress(compressed: dict, seed: int = 0xDEADBEEF) -> np.ndarray:
    """Decompress a PolarQuant-compressed KV head vector.
    
    Args:
        compressed: dict from polarquant_compress
        seed: same seed used for compression
    
    Returns:
        Reconstructed float32 vector of shape (head_dim,)
    """
    # Step 1: Dequantize angles
    angles = dequantize_angles(compressed['angles'], compressed['bits'])
    
    # Step 2: Polar → Cartesian
    pre = polar_to_cartesian(compressed['radius'], angles)
    
    # Step 3: Inverse Hadamard
    result = hadamard_precondition_inv(pre, seed)
    
    return result


def polarquant_round_trip(kv: np.ndarray, bits: int = 4, seed: int = 0xDEADBEEF) -> Tuple[np.ndarray, float]:
    """Compress and decompress, returning (reconstructed, cosine_similarity)."""
    compressed = polarquant_compress(kv, bits, seed)
    reconstructed = polarquant_decompress(compressed, seed)
    
    # Cosine similarity
    dot = np.dot(kv, reconstructed)
    norm_a = np.linalg.norm(kv)
    norm_b = np.linalg.norm(reconstructed)
    if norm_a < 1e-12 or norm_b < 1e-12:
        sim = 0.0
    else:
        sim = float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))
    
    return reconstructed, sim


def compression_ratio(head_dim: int, bits: int, baseline_bits: int = 32) -> float:
    """Theoretical compression ratio.
    
    Baseline: head_dim × baseline_bits
    Compressed: 32 (f32 radius) + (head_dim-1) × bits
    """
    original = head_dim * baseline_bits
    compressed = 32 + (head_dim - 1) * bits
    return original / compressed


# ── QJL Implementation ───────────────────────────────────────────────────────

def qjl_project(kv: np.ndarray, sketch_dim: int, seed: int = 42) -> np.ndarray:
    """JL projection + sign quantization.
    
    Args:
        kv: shape (d,) — key vector
        sketch_dim: output dimension k
        seed: deterministic seed for Gaussian matrix
    
    Returns:
        shape (sketch_dim,) i8 array with values in {-1, +1}
    """
    d = len(kv)
    rng = np.random.RandomState(seed)
    # Gaussian random matrix scaled by 1/sqrt(k)
    A = rng.randn(sketch_dim, d).astype(np.float32) / np.sqrt(sketch_dim)
    projected = A @ kv
    return np.sign(projected).astype(np.int8)


def qjl_attention_score(query: np.ndarray, key_sketch: np.ndarray,
                         sketch_dim: int, seed: int = 42) -> float:
    """Asymmetric attention estimator: estimate Q·K from sketch of K.
    
    Args:
        query: full-precision query vector, shape (d,)
        key_sketch: 1-bit sketch of key, shape (sketch_dim,)
        sketch_dim: sketch dimension
        seed: same seed used for sketching
    
    Returns:
        Estimated dot product Q·K
    """
    d = len(query)
    rng = np.random.RandomState(seed)
    A = rng.randn(sketch_dim, d).astype(np.float32) / np.sqrt(sketch_dim)
    # score = (1/k) * (A^T q) · sketch
    Aq = A @ query
    return float(np.dot(Aq, key_sketch.astype(np.float32)))


def qjl_compression_ratio(original_dim: int, sketch_dim: int) -> float:
    """QJL compression ratio: f32 → i8 sign bits."""
    return (original_dim * 32) / (sketch_dim * 8)


if __name__ == "__main__":
    # Quick self-test
    np.random.seed(42)
    
    print("=== PolarQuant Self-Test ===")
    for bits in [2, 4, 6, 8]:
        sims = []
        for i in range(20):
            kv = np.random.randn(128).astype(np.float32)
            _, sim = polarquant_round_trip(kv, bits=bits)
            sims.append(sim)
        avg_sim = np.mean(sims)
        ratio = compression_ratio(128, bits)
        print(f"  {bits}-bit: avg cosine_sim={avg_sim:.4f}, compression={ratio:.2f}x")
    
    print("\n=== QJL Self-Test ===")
    for sketch_dim in [32, 64, 128, 256]:
        sims = []
        for i in range(50):
            q = np.random.randn(128).astype(np.float32)
            k = np.random.randn(128).astype(np.float32)
            true_score = float(np.dot(q, k))
            sketch = qjl_project(k, sketch_dim, seed=7)
            est_score = qjl_attention_score(q, sketch, sketch_dim, seed=7)
            # Correlation proxy: same-sign agreement
            sims.append((true_score, est_score))
        
        true_scores = [s[0] for s in sims]
        est_scores = [s[1] for s in sims]
        corr = np.corrcoef(true_scores, est_scores)[0, 1]
        ratio = qjl_compression_ratio(128, sketch_dim)
        print(f"  sketch_dim={sketch_dim}: Pearson r={corr:.4f}, compression={ratio:.2f}x")
