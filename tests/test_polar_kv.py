"""
Tests for PolarQuant KV cache compression.

Tests cover:
1. Compress/decompress roundtrip fidelity
2. Compression ratio ≥ 3× for typical configs
3. Angle distribution after preconditioning (PolarQuant insight)
4. Edge cases (zero vector, unit vector, large/small magnitudes)
5. Batch operations
6. Bit-packing correctness
7. Different n_bits configurations
8. Torch backend (if available)
"""

import sys
import math
import numpy as np
import pytest

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'scripts'))
from polar_kv import (
    PolarKVCache,
    PolarKVCacheTorch,
    _cartesian_to_polar,
    _polar_to_cartesian,
    _generate_orthogonal_preconditioner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_128_4():
    """Standard config: head_dim=128, 4-bit angles."""
    return PolarKVCache(head_dim=128, n_bits=4, seed=42)


@pytest.fixture
def cache_64_3():
    """Smaller config: head_dim=64, 3-bit angles."""
    return PolarKVCache(head_dim=64, n_bits=3, seed=42)


@pytest.fixture
def cache_128_8():
    """High-precision config: head_dim=128, 8-bit angles."""
    return PolarKVCache(head_dim=128, n_bits=8, seed=42)


@pytest.fixture
def rng():
    return np.random.RandomState(12345)


# ---------------------------------------------------------------------------
# 1. Compress/decompress roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_basic_roundtrip(self, cache_128_4, rng):
        """Compress + decompress should produce a vector close to the original."""
        kv = rng.randn(128).astype(np.float32)
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        # Cosine similarity should be high (>0.85 for 4-bit on 128-dim)
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
        assert cos_sim > 0.85, f"Cosine similarity too low: {cos_sim}"
    
    def test_roundtrip_many_vectors(self, cache_128_4, rng):
        """Roundtrip fidelity over 100 random vectors."""
        cos_sims = []
        for _ in range(100):
            kv = rng.randn(128).astype(np.float32)
            packed = cache_128_4.compress(kv)
            kv_hat = cache_128_4.decompress(packed)
            cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
            cos_sims.append(cos_sim)
        
        mean_cos = np.mean(cos_sims)
        assert mean_cos > 0.85, f"Mean cosine similarity too low: {mean_cos}"
    
    def test_roundtrip_high_precision(self, cache_128_8, rng):
        """8-bit angles should give very high fidelity."""
        kv = rng.randn(128).astype(np.float32)
        packed = cache_128_8.compress(kv)
        kv_hat = cache_128_8.decompress(packed)
        
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
        assert cos_sim > 0.999, f"8-bit cosine similarity too low: {cos_sim}"
    
    def test_roundtrip_3bit(self, cache_64_3, rng):
        """3-bit angles, smaller head dim — should still work reasonably."""
        kv = rng.randn(64).astype(np.float32)
        packed = cache_64_3.compress(kv)
        kv_hat = cache_64_3.decompress(packed)
        
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
        assert cos_sim > 0.70, f"3-bit cosine similarity too low: {cos_sim}"
    
    def test_norm_preservation(self, cache_128_4, rng):
        """Radius stored in fp16 — norm should be approximately preserved."""
        kv = rng.randn(128).astype(np.float32) * 5.0  # scaled
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        orig_norm = np.linalg.norm(kv)
        recon_norm = np.linalg.norm(kv_hat)
        # fp16 gives ~3 decimal digits of precision
        assert abs(orig_norm - recon_norm) / orig_norm < 0.01, \
            f"Norm not preserved: {orig_norm} vs {recon_norm}"


# ---------------------------------------------------------------------------
# 2. Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_ratio_128_4bit(self, cache_128_4):
        """head_dim=128, 4-bit: ratio should be ≈ 3.91×."""
        ratio = cache_128_4.compression_ratio()
        assert ratio >= 3.0, f"Compression ratio {ratio} < 3.0"
        # Theoretical: 2048 / (16 + 127*4) = 2048/524 ≈ 3.91
        assert abs(ratio - 3.91) < 0.1, f"Unexpected ratio: {ratio}"
    
    def test_ratio_64_3bit(self, cache_64_3):
        """head_dim=64, 3-bit: ratio should be ≥ 3×."""
        ratio = cache_64_3.compression_ratio()
        assert ratio >= 3.0, f"Compression ratio {ratio} < 3.0"
    
    def test_ratio_128_8bit(self, cache_128_8):
        """head_dim=128, 8-bit: ratio should be ≈ 1.98×."""
        ratio = cache_128_8.compression_ratio()
        # 2048 / (16 + 127*8) = 2048/1032 ≈ 1.98
        assert ratio > 1.9, f"Ratio too low for 8-bit: {ratio}"
    
    def test_compressed_bytes(self, cache_128_4, rng):
        """Actual compressed output should match expected byte count."""
        kv = rng.randn(128).astype(np.float32)
        packed = cache_128_4.compress(kv)
        
        expected_bytes = cache_128_4.compressed_bytes_per_vector()
        assert len(packed) == expected_bytes, \
            f"Packed size {len(packed)} != expected {expected_bytes}"
    
    def test_actual_vs_fp16_bytes(self, cache_128_4, rng):
        """Verify actual byte savings."""
        kv = rng.randn(128).astype(np.float32)
        packed = cache_128_4.compress(kv)
        
        fp16_size = 128 * 2  # 256 bytes
        compressed_size = len(packed)
        actual_ratio = fp16_size / compressed_size
        
        assert actual_ratio >= 3.0, f"Actual byte ratio {actual_ratio} < 3.0"


# ---------------------------------------------------------------------------
# 3. Angle distribution after preconditioning
# ---------------------------------------------------------------------------

class TestAngleDistribution:
    def test_mean_near_pi_over_2(self, cache_128_4, rng):
        """
        After preconditioning, angles should concentrate near π/2.
        This is the key PolarQuant insight.
        """
        samples = rng.randn(500, 128).astype(np.float32)
        dist = cache_128_4.measure_angle_distribution(samples)
        
        # Mean should be close to π/2
        assert abs(dist['mean'] - math.pi / 2) < 0.1, \
            f"Mean angle {dist['mean']} too far from π/2 = {math.pi/2}"
    
    def test_concentration_improves_with_dimension(self):
        """Higher dimension → tighter angle concentration (variance ∝ 1/d)."""
        rng = np.random.RandomState(42)
        
        stds = {}
        for dim in [16, 64, 256]:
            cache = PolarKVCache(head_dim=dim, n_bits=4, seed=42)
            samples = rng.randn(300, dim).astype(np.float32)
            dist = cache.measure_angle_distribution(samples)
            stds[dim] = dist['std']
        
        # std should decrease with increasing dimension
        assert stds[256] < stds[64] < stds[16], \
            f"Concentration not improving: {stds}"
    
    def test_preconditioner_is_orthogonal(self, cache_128_4):
        """H should satisfy H @ H^T ≈ I (orthogonal matrix)."""
        HHT = cache_128_4.H @ cache_128_4.H.T
        I = np.eye(128, dtype=np.float32)
        assert np.allclose(HHT, I, atol=1e-5), "H is not orthogonal"
    
    def test_inverse_preconditioner(self, cache_128_4):
        """H_inv should be H^T for orthogonal H."""
        assert np.allclose(cache_128_4.H_inv, cache_128_4.H.T, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_vector(self, cache_128_4):
        """Zero vector should compress and decompress without errors."""
        kv = np.zeros(128, dtype=np.float32)
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        assert np.allclose(kv_hat, 0.0, atol=1e-3), "Zero vector not reconstructed"
    
    def test_unit_vector(self, cache_128_4):
        """Standard basis vector e_0 should roundtrip well."""
        kv = np.zeros(128, dtype=np.float32)
        kv[0] = 1.0
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
        assert cos_sim > 0.85, f"Unit vector cosine sim too low: {cos_sim}"
    
    def test_large_magnitude(self, cache_128_4, rng):
        """Large vectors (fp16 can store up to 65504)."""
        kv = rng.randn(128).astype(np.float32) * 100.0
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat) + 1e-30)
        assert cos_sim > 0.85, f"Large vector cosine sim too low: {cos_sim}"
    
    def test_small_magnitude(self, cache_128_4, rng):
        """Small vectors should still decompress without NaN/Inf."""
        kv = rng.randn(128).astype(np.float32) * 1e-4
        packed = cache_128_4.compress(kv)
        kv_hat = cache_128_4.decompress(packed)
        
        assert not np.any(np.isnan(kv_hat)), "NaN in decompressed small vector"
        assert not np.any(np.isinf(kv_hat)), "Inf in decompressed small vector"
    
    def test_min_head_dim(self):
        """Minimum head_dim=2 should work."""
        cache = PolarKVCache(head_dim=2, n_bits=4, seed=42)
        kv = np.array([1.0, 2.0], dtype=np.float32)
        packed = cache.compress(kv)
        kv_hat = cache.decompress(packed)
        
        cos_sim = np.dot(kv, kv_hat) / (np.linalg.norm(kv) * np.linalg.norm(kv_hat))
        assert cos_sim > 0.95, f"2D cosine sim too low: {cos_sim}"
    
    def test_deterministic(self, cache_128_4, rng):
        """Same input → same output (deterministic preconditioner)."""
        kv = rng.randn(128).astype(np.float32)
        packed1 = cache_128_4.compress(kv)
        packed2 = cache_128_4.compress(kv)
        assert packed1 == packed2, "Compression not deterministic"


# ---------------------------------------------------------------------------
# 5. Batch operations
# ---------------------------------------------------------------------------

class TestBatch:
    def test_compress_batch(self, cache_128_4, rng):
        """Batch compress should produce correct number of packed outputs."""
        batch = rng.randn(10, 128).astype(np.float32)
        packed_list = cache_128_4.compress_batch(batch)
        
        assert len(packed_list) == 10
        for p in packed_list:
            assert len(p) == cache_128_4.compressed_bytes_per_vector()
    
    def test_decompress_batch(self, cache_128_4, rng):
        """Batch decompress should produce correct shape."""
        batch = rng.randn(10, 128).astype(np.float32)
        packed_list = cache_128_4.compress_batch(batch)
        recon = cache_128_4.decompress_batch(packed_list)
        
        assert recon.shape == (10, 128)
    
    def test_batch_roundtrip_matches_individual(self, cache_128_4, rng):
        """Batch roundtrip should match individual roundtrips."""
        batch = rng.randn(5, 128).astype(np.float32)
        
        packed_list = cache_128_4.compress_batch(batch)
        recon_batch = cache_128_4.decompress_batch(packed_list)
        
        for i in range(5):
            packed_i = cache_128_4.compress(batch[i])
            recon_i = cache_128_4.decompress(packed_i)
            assert np.allclose(recon_batch[i], recon_i, atol=1e-6), \
                f"Batch vs individual mismatch at index {i}"


# ---------------------------------------------------------------------------
# 6. Bit-packing correctness
# ---------------------------------------------------------------------------

class TestBitPacking:
    def test_pack_unpack_roundtrip(self, cache_128_4):
        """Pack → unpack should recover exact quantized values."""
        rng = np.random.RandomState(42)
        quantized = rng.randint(0, 16, size=127).astype(np.uint8)  # 4-bit values
        
        packed = cache_128_4._pack_angles(quantized)
        unpacked = cache_128_4._unpack_angles(packed)
        
        np.testing.assert_array_equal(quantized, unpacked)
    
    def test_pack_unpack_3bit(self, cache_64_3):
        """3-bit packing roundtrip."""
        rng = np.random.RandomState(42)
        quantized = rng.randint(0, 8, size=63).astype(np.uint8)  # 3-bit values
        
        packed = cache_64_3._pack_angles(quantized)
        unpacked = cache_64_3._unpack_angles(packed)
        
        np.testing.assert_array_equal(quantized, unpacked)
    
    def test_pack_unpack_8bit(self, cache_128_8):
        """8-bit packing roundtrip (each angle = 1 byte, essentially)."""
        rng = np.random.RandomState(42)
        quantized = rng.randint(0, 256, size=127).astype(np.uint8)
        
        packed = cache_128_8._pack_angles(quantized)
        unpacked = cache_128_8._unpack_angles(packed)
        
        np.testing.assert_array_equal(quantized, unpacked)
    
    def test_all_zeros(self, cache_128_4):
        """All-zero quantized angles should pack/unpack correctly."""
        quantized = np.zeros(127, dtype=np.uint8)
        packed = cache_128_4._pack_angles(quantized)
        unpacked = cache_128_4._unpack_angles(packed)
        np.testing.assert_array_equal(quantized, unpacked)
    
    def test_all_max(self, cache_128_4):
        """All-max quantized angles should pack/unpack correctly."""
        quantized = np.full(127, 15, dtype=np.uint8)  # max for 4-bit
        packed = cache_128_4._pack_angles(quantized)
        unpacked = cache_128_4._unpack_angles(packed)
        np.testing.assert_array_equal(quantized, unpacked)


# ---------------------------------------------------------------------------
# 7. Polar coordinate conversion
# ---------------------------------------------------------------------------

class TestPolarConversion:
    def test_cartesian_polar_roundtrip_2d(self):
        """2D vector roundtrip through polar coords."""
        v = np.array([3.0, 4.0], dtype=np.float32)
        r, angles = _cartesian_to_polar(v)
        v_hat = _polar_to_cartesian(r, angles)
        
        assert abs(r - 5.0) < 1e-5, f"Radius wrong: {r}"
        np.testing.assert_allclose(v, v_hat, atol=1e-5)
    
    def test_cartesian_polar_roundtrip_high_dim(self):
        """128D vector roundtrip through polar coords."""
        rng = np.random.RandomState(42)
        v = rng.randn(128).astype(np.float32)
        
        r, angles = _cartesian_to_polar(v)
        v_hat = _polar_to_cartesian(r, angles)
        
        np.testing.assert_allclose(v, v_hat, atol=1e-4)
    
    def test_radius_is_norm(self):
        """Radius should equal L2 norm."""
        rng = np.random.RandomState(42)
        v = rng.randn(64).astype(np.float32)
        r, _ = _cartesian_to_polar(v)
        
        assert abs(r - np.linalg.norm(v)) < 1e-5
    
    def test_negative_components(self):
        """Vectors with negative components should roundtrip correctly."""
        v = np.array([-1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        r, angles = _cartesian_to_polar(v)
        v_hat = _polar_to_cartesian(r, angles)
        
        np.testing.assert_allclose(v, v_hat, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Preconditioner quality
# ---------------------------------------------------------------------------

class TestPreconditioner:
    def test_orthogonal(self):
        """Generated matrix should be orthogonal."""
        H = _generate_orthogonal_preconditioner(64, seed=42)
        I = np.eye(64, dtype=np.float32)
        np.testing.assert_allclose(H @ H.T, I, atol=1e-5)
        np.testing.assert_allclose(H.T @ H, I, atol=1e-5)
    
    def test_deterministic(self):
        """Same seed → same matrix."""
        H1 = _generate_orthogonal_preconditioner(64, seed=42)
        H2 = _generate_orthogonal_preconditioner(64, seed=42)
        np.testing.assert_array_equal(H1, H2)
    
    def test_different_seeds(self):
        """Different seeds → different matrices."""
        H1 = _generate_orthogonal_preconditioner(64, seed=42)
        H2 = _generate_orthogonal_preconditioner(64, seed=99)
        assert not np.allclose(H1, H2)


# ---------------------------------------------------------------------------
# 9. PyTorch backend
# ---------------------------------------------------------------------------

class TestTorchBackend:
    @pytest.fixture
    def torch_cache(self):
        try:
            import torch
            return PolarKVCacheTorch(head_dim=128, n_bits=4, seed=42, device='cpu')
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_compress_decompress(self, torch_cache):
        import torch
        kv = torch.randn(128)
        radii, quantized = torch_cache.compress_tensor(kv.unsqueeze(0))
        recon = torch_cache.decompress_tensor(radii, quantized)
        
        cos_sim = torch.nn.functional.cosine_similarity(
            kv.unsqueeze(0), recon, dim=-1
        ).item()
        assert cos_sim > 0.85, f"Torch cosine sim too low: {cos_sim}"
    
    def test_batch_compress(self, torch_cache):
        import torch
        kv = torch.randn(32, 128)
        radii, quantized = torch_cache.compress_tensor(kv)
        
        assert radii.shape == (32,)
        assert quantized.shape == (32, 127)
        assert radii.dtype == torch.float16
        assert quantized.dtype == torch.uint8
    
    def test_batch_roundtrip(self, torch_cache):
        import torch
        kv = torch.randn(16, 128)
        radii, quantized = torch_cache.compress_tensor(kv)
        recon = torch_cache.decompress_tensor(radii, quantized)
        
        cos_sim = torch.nn.functional.cosine_similarity(kv, recon, dim=-1)
        assert cos_sim.mean().item() > 0.85
    
    def test_multidim_batch(self, torch_cache):
        """Test with multi-dimensional batch (batch, heads, seq, head_dim)."""
        import torch
        kv = torch.randn(2, 4, 8, 128)  # batch=2, heads=4, seq=8
        radii, quantized = torch_cache.compress_tensor(kv)
        recon = torch_cache.decompress_tensor(radii, quantized)
        
        assert recon.shape == kv.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            kv.reshape(-1, 128), recon.reshape(-1, 128), dim=-1
        )
        assert cos_sim.mean().item() > 0.88
    
    def test_compression_ratio(self, torch_cache):
        ratio = torch_cache.compression_ratio()
        assert ratio >= 3.0


# ---------------------------------------------------------------------------
# 10. Configuration validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_invalid_n_bits_low(self):
        with pytest.raises(AssertionError):
            PolarKVCache(head_dim=128, n_bits=1)
    
    def test_invalid_n_bits_high(self):
        with pytest.raises(AssertionError):
            PolarKVCache(head_dim=128, n_bits=9)
    
    def test_invalid_head_dim(self):
        with pytest.raises(AssertionError):
            PolarKVCache(head_dim=1, n_bits=4)
    
    def test_valid_configs(self):
        """Various valid configs should instantiate without error."""
        for dim in [2, 16, 64, 128, 256]:
            for bits in [2, 3, 4, 5, 6, 7, 8]:
                c = PolarKVCache(head_dim=dim, n_bits=bits)
                assert c.compression_ratio() > 1.0
