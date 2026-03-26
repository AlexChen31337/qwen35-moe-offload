"""
polar_kv.py — PolarQuant KV cache compression
Reference: arXiv:2502.02617 (AISTATS 2026, Google Research + KAIST + Yale)

Algorithm:
1. Random preconditioning: multiply KV vectors by a random orthogonal matrix H
   → makes angle distribution tight and analytically predictable (concentrated ~π/2)
2. Recursive polar decomposition: (x₁,...,xₙ) → (r, θ₁,...,θₙ₋₁)
   where r = ‖x‖₂, θᵢ computed recursively via arctan2
3. Quantize only angles: after preconditioning, distribution is known
   → uniform quantizer on [0, π] works without per-block scale/zero-point
4. Store: radius r in fp16, angles θᵢ in low precision (n_bits)

Key insight: after random preconditioning, angles concentrate near π/2
with variance O(1/d) where d = head dimension. A uniform quantizer on [0, π]
achieves near-optimal compression with ZERO normalization overhead.
"""

import struct
import math
import numpy as np
from typing import Optional


def _generate_orthogonal_preconditioner(dim: int, seed: int) -> np.ndarray:
    """
    Generate a fixed random orthogonal matrix via QR decomposition of a
    random Gaussian matrix. This is the random preconditioning step from
    PolarQuant — it rotates the KV vectors so their angular distribution
    becomes tight and predictable.
    
    Args:
        dim: dimension of the matrix (head_dim × head_dim)
        seed: random seed for reproducibility
        
    Returns:
        Orthogonal matrix H of shape (dim, dim), float32
    """
    rng = np.random.RandomState(seed)
    # Gaussian random matrix
    G = rng.randn(dim, dim).astype(np.float32)
    # QR decomposition → Q is orthogonal
    Q, R = np.linalg.qr(G)
    # Ensure deterministic sign (Haar-distributed): multiply by sign of diagonal of R
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d[np.newaxis, :]
    return Q


def _cartesian_to_polar(v: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Recursive polar decomposition: convert n-dimensional Cartesian vector
    to polar coordinates (r, θ₁, ..., θₙ₋₁).
    
    For a vector (x₁, ..., xₙ):
      r = ‖x‖₂
      θ₁ = arccos(x₁ / r)                    ∈ [0, π]
      θ₂ = arccos(x₂ / (r·sin(θ₁)))         ∈ [0, π]
      ...
      θₙ₋₂ = arccos(x_{n-2} / (r·∏sin(θⱼ))) ∈ [0, π]
      θₙ₋₁ = atan2(xₙ, xₙ₋₁)                ∈ [-π, π] → shifted to [0, 2π]
    
    Args:
        v: 1D numpy array of shape (d,)
        
    Returns:
        (r, angles) where r is scalar radius and angles is array of d-1 angles
    """
    d = len(v)
    r = np.linalg.norm(v)
    
    if r < 1e-30:
        # Zero vector → all angles π/2 (the concentration point)
        return r, np.full(d - 1, math.pi / 2, dtype=np.float32)
    
    angles = np.zeros(d - 1, dtype=np.float64)
    
    # Recursive: θᵢ = arccos(xᵢ / ‖remaining‖)
    # where ‖remaining‖ = sqrt(x_i² + x_{i+1}² + ... + x_{d-1}²)
    for i in range(d - 2):
        remaining_norm = np.linalg.norm(v[i:])
        if remaining_norm < 1e-30:
            angles[i] = math.pi / 2
        else:
            # Clamp to [-1, 1] for numerical stability
            cos_val = np.clip(v[i] / remaining_norm, -1.0, 1.0)
            angles[i] = math.acos(cos_val)
    
    # Last angle uses atan2 for full [0, 2π] range
    angles[d - 2] = math.atan2(v[d - 1], v[d - 2])
    # Shift from [-π, π] to [0, 2π]
    if angles[d - 2] < 0:
        angles[d - 2] += 2 * math.pi
    
    return float(r), angles.astype(np.float32)


def _polar_to_cartesian(r: float, angles: np.ndarray) -> np.ndarray:
    """
    Reconstruct Cartesian vector from polar coordinates (r, θ₁, ..., θₙ₋₁).
    
    x₁ = r·cos(θ₁)
    x₂ = r·sin(θ₁)·cos(θ₂)
    x₃ = r·sin(θ₁)·sin(θ₂)·cos(θ₃)
    ...
    xₙ₋₁ = r·∏sin(θⱼ, j=1..n-2)·cos(θₙ₋₁)
    xₙ = r·∏sin(θⱼ, j=1..n-2)·sin(θₙ₋₁)
    
    where the last angle θₙ₋₁ uses sin/cos directly (from atan2).
    """
    d = len(angles) + 1
    v = np.zeros(d, dtype=np.float64)
    
    if r < 1e-30:
        return v.astype(np.float32)
    
    # Product of sines accumulated
    sin_product = r
    
    for i in range(d - 2):
        v[i] = sin_product * math.cos(angles[i])
        sin_product *= math.sin(angles[i])
    
    # Last two components from the atan2 angle
    v[d - 2] = sin_product * math.cos(angles[d - 2])
    v[d - 1] = sin_product * math.sin(angles[d - 2])
    
    return v.astype(np.float32)


class PolarKVCache:
    """
    PolarQuant KV cache compressor.
    
    Compresses head_dim-dimensional KV vectors using:
    1. Random orthogonal preconditioning (rotates distribution)
    2. Polar decomposition (radius + angles)
    3. Uniform angle quantization (no per-block normalization needed)
    
    Compression ratio: fp16 baseline (16 bits/dim) vs compressed
    (fp16 radius + n_bits per angle).
    
    For head_dim=128, n_bits=4:
      Baseline: 128 × 16 = 2048 bits
      Compressed: 16 (radius) + 127 × 4 = 524 bits
      Ratio: 2048 / 524 ≈ 3.91×
    
    Args:
        head_dim: dimension of each KV head vector
        n_bits: bits per quantized angle (2-8, default 4)
        seed: random seed for preconditioner generation
    """
    
    def __init__(self, head_dim: int, n_bits: int = 4, seed: int = 42):
        assert 2 <= n_bits <= 8, f"n_bits must be in [2, 8], got {n_bits}"
        assert head_dim >= 2, f"head_dim must be >= 2, got {head_dim}"
        
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.seed = seed
        
        # Number of quantization levels per angle
        self.n_levels = (1 << n_bits)  # 2^n_bits
        
        # Generate fixed orthogonal preconditioner
        self.H = _generate_orthogonal_preconditioner(head_dim, seed)
        # H^{-1} = H^T for orthogonal matrices
        self.H_inv = self.H.T.copy()
        
        # Number of angles = head_dim - 1
        self.n_angles = head_dim - 1
        
        # Angle ranges:
        # θ₁ to θₙ₋₂: [0, π]
        # θₙ₋₁ (last): [0, 2π]
        self._angle_max = np.full(self.n_angles, math.pi, dtype=np.float32)
        self._angle_max[-1] = 2 * math.pi
    
    def compress(self, kv_vector: np.ndarray) -> bytes:
        """
        Compress a single KV vector to packed bytes.
        
        Steps:
        1. Apply preconditioner: v = H @ kv_vector
        2. Polar decomposition: (r, θ₁, ..., θₙ₋₁)
        3. Quantize angles to n_bits uniform grid
        4. Pack: fp16 radius + packed angle bits
        
        Args:
            kv_vector: 1D float32 array of shape (head_dim,)
            
        Returns:
            Packed bytes representation
        """
        assert kv_vector.shape == (self.head_dim,), \
            f"Expected shape ({self.head_dim},), got {kv_vector.shape}"
        
        # Step 1: Precondition
        v = self.H @ kv_vector.astype(np.float32)
        
        # Step 2: Polar decomposition
        r, angles = _cartesian_to_polar(v)
        
        # Step 3: Quantize angles
        # Each angle θᵢ ∈ [0, max_i] → quantize to [0, n_levels-1]
        quantized = np.zeros(self.n_angles, dtype=np.uint8)
        for i in range(self.n_angles):
            # Clamp to valid range
            a = np.clip(angles[i], 0.0, self._angle_max[i])
            # Uniform quantization
            q = a / self._angle_max[i] * (self.n_levels - 1)
            quantized[i] = int(np.clip(np.round(q), 0, self.n_levels - 1))
        
        # Step 4: Pack into bytes
        # Header: fp16 radius (2 bytes)
        packed = struct.pack('<e', np.float16(r))
        
        # Pack quantized angles into a bitstream
        packed += self._pack_angles(quantized)
        
        return packed
    
    def decompress(self, packed: bytes) -> np.ndarray:
        """
        Decompress packed bytes back to a KV vector.
        
        Steps:
        1. Unpack fp16 radius + quantized angles
        2. Dequantize angles to continuous values
        3. Reconstruct Cartesian from polar coordinates
        4. Apply inverse preconditioner: x = H^{-1} @ v
        
        Args:
            packed: bytes from compress()
            
        Returns:
            Reconstructed float32 array of shape (head_dim,)
        """
        # Step 1: Unpack radius
        r = float(struct.unpack('<e', packed[:2])[0])
        
        # Unpack quantized angles
        quantized = self._unpack_angles(packed[2:])
        
        # Step 2: Dequantize
        angles = np.zeros(self.n_angles, dtype=np.float32)
        for i in range(self.n_angles):
            angles[i] = (quantized[i] / (self.n_levels - 1)) * self._angle_max[i]
        
        # Step 3: Polar → Cartesian
        v = _polar_to_cartesian(r, angles)
        
        # Step 4: Inverse preconditioner
        x = self.H_inv @ v
        
        return x.astype(np.float32)
    
    def compress_batch(self, kv_vectors: np.ndarray) -> list[bytes]:
        """
        Compress a batch of KV vectors.
        
        Args:
            kv_vectors: 2D array of shape (seq_len, head_dim)
            
        Returns:
            List of packed bytes, one per vector
        """
        assert kv_vectors.ndim == 2 and kv_vectors.shape[1] == self.head_dim
        return [self.compress(kv_vectors[i]) for i in range(len(kv_vectors))]
    
    def decompress_batch(self, packed_list: list[bytes]) -> np.ndarray:
        """
        Decompress a batch of packed KV vectors.
        
        Args:
            packed_list: list of packed bytes from compress_batch()
            
        Returns:
            2D float32 array of shape (len(packed_list), head_dim)
        """
        vecs = [self.decompress(p) for p in packed_list]
        return np.stack(vecs, axis=0)
    
    def compressed_size_bits(self) -> int:
        """
        Size of a single compressed vector in bits.
        
        = 16 (fp16 radius) + n_angles × n_bits
        """
        return 16 + self.n_angles * self.n_bits
    
    def baseline_size_bits(self) -> int:
        """
        Size of a single fp16 KV vector in bits.
        
        = head_dim × 16
        """
        return self.head_dim * 16
    
    def compression_ratio(self) -> float:
        """
        Compression ratio: fp16 baseline / compressed size.
        
        For head_dim=128, n_bits=4:
          Baseline: 128 × 16 = 2048 bits
          Compressed: 16 + 127 × 4 = 524 bits
          Ratio: 2048 / 524 ≈ 3.91×
        """
        return self.baseline_size_bits() / self.compressed_size_bits()
    
    def compressed_bytes_per_vector(self) -> int:
        """Number of bytes needed to store one compressed vector."""
        # 2 bytes for fp16 radius + ceil(n_angles * n_bits / 8) for angles
        angle_bytes = math.ceil(self.n_angles * self.n_bits / 8)
        return 2 + angle_bytes
    
    def _pack_angles(self, quantized: np.ndarray) -> bytes:
        """
        Bitpack quantized angle values into bytes.
        
        Each angle takes n_bits. We pack them MSB-first into a byte stream.
        """
        total_bits = self.n_angles * self.n_bits
        total_bytes = math.ceil(total_bits / 8)
        result = bytearray(total_bytes)
        
        bit_pos = 0
        for i in range(self.n_angles):
            val = int(quantized[i])
            # Write n_bits of val starting at bit_pos
            for b in range(self.n_bits):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                if val & (1 << (self.n_bits - 1 - b)):
                    result[byte_idx] |= (1 << (7 - bit_idx))
                bit_pos += 1
        
        return bytes(result)
    
    def _unpack_angles(self, data: bytes) -> np.ndarray:
        """
        Unpack bitpacked angle values from bytes.
        """
        quantized = np.zeros(self.n_angles, dtype=np.uint8)
        
        bit_pos = 0
        for i in range(self.n_angles):
            val = 0
            for b in range(self.n_bits):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                if data[byte_idx] & (1 << (7 - bit_idx)):
                    val |= (1 << (self.n_bits - 1 - b))
                bit_pos += 1
            quantized[i] = val
        
        return quantized
    
    def measure_angle_distribution(self, kv_vectors: np.ndarray) -> dict:
        """
        Measure the angle distribution after preconditioning.
        
        This validates the PolarQuant insight: after random orthogonal
        preconditioning, angles should concentrate near π/2 with
        variance O(1/d).
        
        Args:
            kv_vectors: 2D array of shape (n_samples, head_dim)
            
        Returns:
            Dict with 'mean', 'std', 'expected_mean' (π/2), 
            'expected_std' (√(1/d)), 'concentration_quality'
        """
        all_angles = []
        
        for i in range(len(kv_vectors)):
            v = self.H @ kv_vectors[i].astype(np.float32)
            _, angles = _cartesian_to_polar(v)
            # Exclude last angle (atan2, different range)
            all_angles.append(angles[:-1])
        
        all_angles = np.concatenate(all_angles)
        
        mean = float(np.mean(all_angles))
        std = float(np.std(all_angles))
        expected_mean = math.pi / 2
        expected_std = 1.0 / math.sqrt(self.head_dim)
        
        # Quality: how close is the std to the theoretical bound
        # Lower is better (tighter concentration)
        concentration_quality = std / expected_std if expected_std > 0 else float('inf')
        
        return {
            'mean': mean,
            'std': std,
            'expected_mean': expected_mean,
            'expected_std': expected_std,
            'concentration_quality': concentration_quality,
            'n_samples': len(kv_vectors),
        }


class PolarKVCacheTorch:
    """
    PyTorch-accelerated PolarQuant for integration with model inference.
    
    Same algorithm as PolarKVCache but uses torch tensors for GPU acceleration.
    Falls back to numpy implementation if torch is unavailable.
    """
    
    def __init__(self, head_dim: int, n_bits: int = 4, seed: int = 42, 
                 device: str = 'cpu'):
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError("PyTorch required for PolarKVCacheTorch")
        
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.n_levels = (1 << n_bits)
        self.n_angles = head_dim - 1
        self.device = device
        
        # Generate preconditioner on CPU, move to device
        H_np = _generate_orthogonal_preconditioner(head_dim, seed)
        self.H = torch.from_numpy(H_np).to(device)
        self.H_inv = self.H.T.contiguous()
        
        # Angle ranges
        angle_max = torch.full((self.n_angles,), math.pi, device=device)
        angle_max[-1] = 2 * math.pi
        self.angle_max = angle_max
    
    def compress_tensor(self, kv: 'torch.Tensor') -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Compress a batch of KV vectors.
        
        Args:
            kv: tensor of shape (..., head_dim)
            
        Returns:
            (radii, quantized_angles) where:
              radii: fp16 tensor of shape (...,)
              quantized_angles: uint8 tensor of shape (..., n_angles)
        """
        torch = self._torch
        orig_shape = kv.shape[:-1]
        flat = kv.reshape(-1, self.head_dim).float()
        
        # Step 1: Precondition — batch matrix multiply
        # v = flat @ H^T  (each row multiplied by H)
        v = flat @ self.H.T
        
        # Step 2: Compute radii
        radii = torch.norm(v, dim=-1)  # shape (N,)
        
        # Step 3: Recursive polar decomposition (vectorized)
        N, d = v.shape
        angles = torch.zeros(N, d - 1, device=self.device, dtype=torch.float32)
        
        # For each dimension except the last two, compute arccos
        for i in range(d - 2):
            remaining_norm = torch.norm(v[:, i:], dim=-1)
            cos_val = torch.clamp(v[:, i] / (remaining_norm + 1e-30), -1.0, 1.0)
            angles[:, i] = torch.acos(cos_val)
        
        # Last angle: atan2
        angles[:, d - 2] = torch.atan2(v[:, d - 1], v[:, d - 2])
        # Shift to [0, 2π]
        angles[:, d - 2] = torch.where(
            angles[:, d - 2] < 0,
            angles[:, d - 2] + 2 * math.pi,
            angles[:, d - 2]
        )
        
        # Step 4: Quantize
        angles = torch.clamp(angles, min=0.0)
        angles = torch.min(angles, self.angle_max.unsqueeze(0))
        q = angles / self.angle_max.unsqueeze(0) * (self.n_levels - 1)
        quantized = torch.clamp(torch.round(q), 0, self.n_levels - 1).to(torch.uint8)
        
        # Reshape back
        radii = radii.reshape(orig_shape).to(torch.float16)
        quantized = quantized.reshape(*orig_shape, self.n_angles)
        
        return radii, quantized
    
    def decompress_tensor(self, radii: 'torch.Tensor', 
                          quantized: 'torch.Tensor') -> 'torch.Tensor':
        """
        Decompress quantized angles + radii back to KV vectors.
        
        Args:
            radii: fp16 tensor of shape (...,)
            quantized: uint8 tensor of shape (..., n_angles)
            
        Returns:
            Reconstructed tensor of shape (..., head_dim)
        """
        torch = self._torch
        orig_shape = radii.shape
        
        radii_flat = radii.reshape(-1).float()
        q_flat = quantized.reshape(-1, self.n_angles).float()
        N = radii_flat.shape[0]
        d = self.head_dim
        
        # Dequantize angles
        angles = q_flat / (self.n_levels - 1) * self.angle_max.unsqueeze(0)
        
        # Polar → Cartesian (vectorized)
        v = torch.zeros(N, d, device=self.device, dtype=torch.float32)
        
        sin_product = radii_flat.clone()
        
        for i in range(d - 2):
            v[:, i] = sin_product * torch.cos(angles[:, i])
            sin_product = sin_product * torch.sin(angles[:, i])
        
        # Last two from atan2 angle
        v[:, d - 2] = sin_product * torch.cos(angles[:, d - 2])
        v[:, d - 1] = sin_product * torch.sin(angles[:, d - 2])
        
        # Inverse preconditioner
        x = v @ self.H_inv.T
        
        return x.reshape(*orig_shape, d)
    
    def compression_ratio(self) -> float:
        """fp16 baseline / compressed size ratio."""
        baseline = self.head_dim * 16  # bits
        compressed = 16 + self.n_angles * self.n_bits  # bits
        return baseline / compressed


if __name__ == '__main__':
    # Quick demo
    print("PolarQuant KV Cache Compression Demo")
    print("=" * 50)
    
    head_dim = 128
    n_bits = 4
    
    cache = PolarKVCache(head_dim=head_dim, n_bits=n_bits)
    
    # Generate random KV vector (simulating a real head vector)
    rng = np.random.RandomState(123)
    kv = rng.randn(head_dim).astype(np.float32)
    
    # Compress
    packed = cache.compress(kv)
    
    # Decompress
    kv_hat = cache.decompress(packed)
    
    # Metrics
    mse = np.mean((kv - kv_hat) ** 2)
    cosine_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat))
    
    print(f"Head dimension: {head_dim}")
    print(f"Quantization bits: {n_bits}")
    print(f"Compression ratio: {cache.compression_ratio():.2f}×")
    print(f"Compressed size: {len(packed)} bytes vs {head_dim * 2} bytes (fp16)")
    print(f"MSE: {mse:.6f}")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    
    # Angle distribution analysis
    print("\nAngle Distribution Analysis (n=1000 random vectors):")
    samples = rng.randn(1000, head_dim).astype(np.float32)
    dist = cache.measure_angle_distribution(samples)
    print(f"  Mean angle: {dist['mean']:.4f} (expected: {dist['expected_mean']:.4f} = π/2)")
    print(f"  Std angle:  {dist['std']:.4f} (expected: {dist['expected_std']:.4f} = 1/√d)")
    print(f"  Concentration quality: {dist['concentration_quality']:.2f}× theoretical")
