//! Johnson-Lindenstrauss random projection (CPU version).
//!
//! This module implements the JL transform: project a high-dimensional
//! vector `x ∈ ℝ^d` down to `x' ∈ ℝ^k` (k << d) via a random Gaussian
//! matrix `A ∈ ℝ^{k×d}` scaled by `1/√k`.
//!
//! The JL lemma guarantees that pairwise dot-products (and hence attention
//! scores) are approximately preserved with high probability when k is
//! large enough (typically k = 64–256).
//!
//! Reference: arXiv:2406.03482 (QJL paper), Section 2.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Generate a deterministic k×d Gaussian projection matrix.
///
/// Each entry is drawn from N(0, 1) using the given seed.  The matrix is
/// stored in row-major order: `matrix[i * d + j]` is the (i, j) entry.
pub fn gaussian_matrix(k: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..k * d)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect()
}

/// Project `x ∈ ℝ^d` to `y ∈ ℝ^k` using `A` (k×d Gaussian matrix).
///
/// The output is scaled by `1/√k` so that `E[‖y‖²] = ‖x‖²`.
///
/// # Panics
/// Panics if `matrix.len() != k * d` or `x.len() != d`.
pub fn project(x: &[f32], matrix: &[f32], k: usize) -> Vec<f32> {
    let d = x.len();
    assert_eq!(matrix.len(), k * d, "matrix must have k*d elements");

    let scale = 1.0 / (k as f32).sqrt();
    let mut out = vec![0.0_f32; k];

    for (i, y_i) in out.iter_mut().enumerate() {
        let row = &matrix[i * d..(i + 1) * d];
        *y_i = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
    out
}

/// Approximate inverse: project `y ∈ ℝ^k` back to `x̂ ∈ ℝ^d` using Aᵀ.
///
/// This is *not* the true inverse (it loses information), but `Aᵀy` is an
/// unbiased estimator of the original vector in expectation.  Useful for
/// reconstruction quality tests.
///
/// Output is scaled by `√k / k = 1/√k` to approximately undo the forward
/// scaling — actually we scale by `√k` to invert the `1/√k` forward scale
/// then divide by `d/k` — the correct factor is `√k` on the reconstruction.
pub fn reconstruct(y: &[f32], matrix: &[f32], d: usize) -> Vec<f32> {
    let k = y.len();
    assert_eq!(matrix.len(), k * d, "matrix must have k*d elements");

    let scale = (k as f32).sqrt();
    let mut out = vec![0.0_f32; d];

    for (i, &y_i) in y.iter().enumerate() {
        let row = &matrix[i * d..(i + 1) * d];
        for (x_j, &a_ij) in out.iter_mut().zip(row.iter()) {
            *x_j += a_ij * y_i * scale;
        }
    }
    // Normalise by k to correct for the k-sum.
    let inv_k = 1.0 / k as f32;
    for v in out.iter_mut() {
        *v *= inv_k;
    }
    out
}

/// Draw a seeded random Gaussian float.  Used in tests and the estimator.
pub fn seeded_gaussian(seed: u64) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    StandardNormal.sample(&mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
    }

    #[test]
    fn project_output_length() {
        let x = rand_vec(128, 1);
        let mat = gaussian_matrix(64, 128, 42);
        let y = project(&x, &mat, 64);
        assert_eq!(y.len(), 64);
    }

    #[test]
    fn jl_preserves_dot_product_approximately() {
        // The JL lemma guarantees that ‖Ax - Ay‖² ≈ ‖x - y‖².
        // For a single random matrix, projected dot-products E[<Ax,Ay>] = <x,y>,
        // but variance is high at finite k.  We verify the absolute error is
        // bounded on average across many pairs.
        let d = 64;
        let k = 256; // generous sketch dim for tight error
        let mat = gaussian_matrix(k, d, 99);

        let n = 50;
        let mut total_err = 0.0_f32;
        let mut total_scale = 0.0_f32;

        for i in 0..n {
            let x = rand_vec(d, i as u64 * 2 + 10);
            let y = rand_vec(d, i as u64 * 2 + 11);
            // Normalise for a fair comparison.
            let nx: f32 = x.iter().map(|v| v*v).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|v| v*v).sum::<f32>().sqrt();
            let scale = (nx * ny).max(1e-6);
            let dot_xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>() / scale;
            let px = project(&x, &mat, k);
            let py = project(&y, &mat, k);
            let npx: f32 = px.iter().map(|v| v*v).sum::<f32>().sqrt();
            let npy: f32 = py.iter().map(|v| v*v).sum::<f32>().sqrt();
            let pscale = (npx * npy).max(1e-6);
            let dot_pxpy: f32 = px.iter().zip(py.iter()).map(|(a, b)| a * b).sum::<f32>() / pscale;
            total_err += (dot_pxpy - dot_xy).abs();
            total_scale += 1.0;
        }

        let avg_err = total_err / total_scale;
        // Average absolute error in normalised dot-product should be < 0.3.
        assert!(
            avg_err < 0.3,
            "JL average normalised dot-product error = {avg_err:.3} (expected < 0.3)"
        );
    }

    #[test]
    fn reconstruct_preserves_direction() {
        let d = 64;
        let k = 32;
        let x = rand_vec(d, 5);
        let mat = gaussian_matrix(k, d, 7);
        let y = project(&x, &mat, k);
        let xhat = reconstruct(&y, &mat, d);

        let dot: f32 = x.iter().zip(xhat.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = xhat.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = if na > 1e-12 && nb > 1e-12 { dot / (na * nb) } else { 0.0 };
        // Direction should be roughly preserved (not perfect, just directionally correct).
        assert!(cos > 0.5, "reconstruct cosine similarity = {cos}");
    }

    #[test]
    fn gaussian_matrix_deterministic() {
        let m1 = gaussian_matrix(4, 4, 42);
        let m2 = gaussian_matrix(4, 4, 42);
        assert_eq!(m1, m2);
    }

    #[test]
    fn different_seeds_give_different_matrices() {
        let m1 = gaussian_matrix(4, 4, 1);
        let m2 = gaussian_matrix(4, 4, 2);
        assert_ne!(m1, m2);
    }
}
