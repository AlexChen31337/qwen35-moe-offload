//! # qjl
//!
//! CPU implementation of **QJL** (Quantized Johnson-Lindenstrauss) KV cache
//! compression from arXiv:2406.03482.
//!
//! ## Algorithm summary
//! QJL compresses key (K) vectors in transformer attention to 1-bit sketches:
//!
//! 1. **Sketch**: project K ∈ ℝ^d via a random Gaussian matrix A ∈ ℝ^{k×d}
//!    (seed-deterministic), then store `sign(AK) ∈ {−1,+1}^k` as `i8`.
//! 2. **Reconstruction** (approximate): apply Aᵀ to the sign vector.
//! 3. **Attention estimation**: given query Q at full precision, compute an
//!    *asymmetric* estimator `score(Q, sign(AK))` that approximates `Q·K`
//!    without ever materialising K at full precision.
//!
//! ## Compression ratio
//! `original_dim` f32 elements → `sketch_dim` i8 elements:
//!   ratio = (original_dim × 32) / (sketch_dim × 8) = 4 × (original_dim / sketch_dim)
//!
//! For head_dim=128, sketch_dim=64: ratio = 8×.
//! For head_dim=128, sketch_dim=128: ratio = 4×.
//!
//! ## CPU vs CUDA
//! All functions in this crate run on CPU.  The `cuda` feature flag compiles
//! in cudarc bindings for future GPU acceleration, but does not change any
//! function signatures or behaviour on CPU-only builds.

pub mod estimator;
pub mod transform;

use estimator::estimate_attention_score;
use transform::{gaussian_matrix, project, reconstruct};

// ─── Public API ──────────────────────────────────────────────────────────────

/// Johnson-Lindenstrauss projection + sign-bit quantization.
///
/// Projects `kv ∈ ℝ^d` to `sign(A·kv) ∈ {−1,+1}^sketch_dim` using a
/// deterministic Gaussian matrix seeded by `seed`.
///
/// # Arguments
/// * `kv` — key or value vector of dimension d
/// * `sketch_dim` — output dimension (k in the paper); should be ≥ 32
/// * `seed` — seed for the random Gaussian matrix; must match at decode time
///
/// # Returns
/// A `Vec<i8>` of length `sketch_dim` with values in {−1, +1}.
pub fn jl_project(kv: &[f32], sketch_dim: usize, seed: u64) -> Vec<i8> {
    let d = kv.len();
    let matrix = gaussian_matrix(sketch_dim, d, seed);
    let projected = project(kv, &matrix, sketch_dim);
    projected
        .into_iter()
        .map(|v| if v >= 0.0 { 1i8 } else { -1i8 })
        .collect()
}

/// Approximate reconstruction of the original vector from a 1-bit JL sketch.
///
/// This is a lossy reconstruction: it preserves direction approximately but
/// cannot recover magnitude exactly.  Useful for sanity-checking and
/// soft-attention emulation.
///
/// # Arguments
/// * `sketch` — 1-bit sketch from [`jl_project`] (values in {−1, +1})
/// * `original_dim` — original vector dimension d
/// * `sketch_dim` — sketch dimension k (must equal `sketch.len()`)
/// * `seed` — same seed used in [`jl_project`]
pub fn jl_reconstruct(sketch: &[i8], original_dim: usize, sketch_dim: usize, seed: u64) -> Vec<f32> {
    assert_eq!(
        sketch.len(),
        sketch_dim,
        "sketch length must equal sketch_dim"
    );
    let matrix = gaussian_matrix(sketch_dim, original_dim, seed);
    let y: Vec<f32> = sketch.iter().map(|&s| s as f32).collect();
    reconstruct(&y, &matrix, original_dim)
}

/// Asymmetric attention score estimator: estimate `query · key` from the
/// 1-bit sketch of the key.
///
/// # Arguments
/// * `query` — full-precision query vector
/// * `key_sketch` — 1-bit sketch of the key from [`jl_project`]
/// * `sketch_dim` — sketch dimension k
/// * `seed` — same seed used when sketching the key
///
/// # Returns
/// Estimated dot product `query · key`.
pub fn attention_score(query: &[f32], key_sketch: &[i8], sketch_dim: usize, seed: u64) -> f32 {
    estimate_attention_score(query, key_sketch, sketch_dim, seed)
}

/// Theoretical compression ratio of the JL sketch.
///
/// Compares storing `original_dim` f32 values (32 bits each) against
/// `sketch_dim` i8 sign values (8 bits each, 1 bit effective).
///
/// ratio = (original_dim × 32) / (sketch_dim × 8)
///
/// # Examples
/// ```
/// use qjl::compression_ratio;
/// let r = compression_ratio(128, 64);
/// assert!((r - 8.0).abs() < 0.01);
/// ```
pub fn compression_ratio(original_dim: usize, sketch_dim: usize) -> f64 {
    (original_dim as f64 * 32.0) / (sketch_dim as f64 * 8.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
    }

    // ── jl_project ──────────────────────────────────────────────────────────

    #[test]
    fn jl_project_output_length() {
        let kv = rand_vec(128, 1);
        let sketch = jl_project(&kv, 64, 42);
        assert_eq!(sketch.len(), 64);
    }

    #[test]
    fn jl_project_values_are_plus_minus_one() {
        let kv = rand_vec(128, 2);
        let sketch = jl_project(&kv, 64, 42);
        for &s in &sketch {
            assert!(s == 1i8 || s == -1i8, "sketch value {s} is not ±1");
        }
    }

    #[test]
    fn jl_project_deterministic() {
        let kv = rand_vec(64, 3);
        let s1 = jl_project(&kv, 32, 99);
        let s2 = jl_project(&kv, 32, 99);
        assert_eq!(s1, s2, "same seed must give same sketch");
    }

    #[test]
    fn jl_project_different_seeds_give_different_sketches() {
        let kv = rand_vec(64, 4);
        let s1 = jl_project(&kv, 32, 1);
        let s2 = jl_project(&kv, 32, 2);
        assert_ne!(s1, s2);
    }

    // ── jl_reconstruct ──────────────────────────────────────────────────────

    #[test]
    fn jl_reconstruct_output_length() {
        let kv = rand_vec(128, 5);
        let sketch = jl_project(&kv, 64, 42);
        let reconstructed = jl_reconstruct(&sketch, 128, 64, 42);
        assert_eq!(reconstructed.len(), 128);
    }

    #[test]
    fn jl_reconstruct_preserves_sign_consistency() {
        // The reconstructed vector should agree in sign with the original
        // for at least 50% of dimensions (better than random).
        let d = 128;
        let sketch_dim = 64;
        let kv = rand_vec(d, 6);
        let sketch = jl_project(&kv, sketch_dim, 42);
        let recon = jl_reconstruct(&sketch, d, sketch_dim, 42);
        let agrees: usize = kv.iter().zip(recon.iter())
            .filter(|(a, b)| (a.signum() as i8) == (b.signum() as i8))
            .count();
        let rate = agrees as f32 / d as f32;
        assert!(rate > 0.5, "sign consistency rate = {rate:.2}");
    }

    // ── attention_score ──────────────────────────────────────────────────────

    #[test]
    fn attention_score_returns_finite() {
        let q = rand_vec(64, 10);
        let k = rand_vec(64, 11);
        let sketch = jl_project(&k, 64, 42);
        let score = attention_score(&q, &sketch, 64, 42);
        assert!(score.is_finite(), "attention score must be finite, got {score}");
    }

    // ── compression_ratio ───────────────────────────────────────────────────

    #[test]
    fn compression_ratio_128_64() {
        let r = compression_ratio(128, 64);
        // (128 * 32) / (64 * 8) = 4096 / 512 = 8.0
        assert!((r - 8.0).abs() < 0.001, "ratio = {r}");
    }

    #[test]
    fn compression_ratio_128_128() {
        let r = compression_ratio(128, 128);
        // (128 * 32) / (128 * 8) = 4096 / 1024 = 4.0
        assert!((r - 4.0).abs() < 0.001, "ratio = {r}");
    }

    #[test]
    fn compression_ratio_higher_sketch_dim_less_compression() {
        let r_small = compression_ratio(128, 32);
        let r_large = compression_ratio(128, 128);
        assert!(r_small > r_large, "smaller sketch_dim should give higher ratio");
    }
}
