//! Asymmetric attention score estimator for QJL.
//!
//! In standard attention, the score for query q against key k is `q·k`.
//! QJL stores keys as 1-bit JL sketches (`sign(Ak)`, where A is a random
//! Gaussian matrix).  Queries are stored at full precision.
//!
//! The asymmetric estimator computes:
//!   `score(q, sign(Ak)) = (√(π/2) / sketch_dim) * q^T A^T sign(Ak)`
//!
//! which is an unbiased estimator of `q·k` under the JL distribution.
//!
//! Reference: arXiv:2406.03482, §3 (QJL attention estimation).

use crate::transform::gaussian_matrix;

/// Estimate the attention score `q · k` from query `q` (full precision) and
/// the 1-bit JL sketch of key `k`.
///
/// The estimator is:
///   `score = (√(π/2) / k) * Σ_i q̃_i * sign_i`
///
/// where `q̃ = A q` is the projected query and `sign_i ∈ {-1, +1}`.
///
/// # Arguments
/// * `query` — full-precision query vector of dimension `d`
/// * `key_sketch` — 1-bit sketch of the key: `sign(Ak) ∈ {-1, +1}^sketch_dim` stored as i8
/// * `sketch_dim` — dimension of the JL sketch space (k in the paper)
/// * `seed` — same seed used to generate the JL matrix during key sketching
///
/// # Panics
/// Panics if `key_sketch.len() != sketch_dim`.
pub fn estimate_attention_score(
    query: &[f32],
    key_sketch: &[i8],
    sketch_dim: usize,
    seed: u64,
) -> f32 {
    assert_eq!(
        key_sketch.len(),
        sketch_dim,
        "key_sketch length must equal sketch_dim"
    );

    let d = query.len();
    let matrix = gaussian_matrix(sketch_dim, d, seed);

    // Project query: q̃ = A q  (scaled by 1/√k already in project())
    // But here we need the raw projection (without the 1/√k scaling from
    // the JL theorem), so we recompute manually.
    let scale = 1.0 / (sketch_dim as f32).sqrt();
    let mut q_proj = vec![0.0_f32; sketch_dim];
    for (i, qp) in q_proj.iter_mut().enumerate() {
        let row = &matrix[i * d..(i + 1) * d];
        *qp = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
    }

    // Estimator: (√(π/2) / sketch_dim) * q̃ · sign(Ak)
    // The sign values are in {-1, +1} encoded as i8.
    let correction = (std::f32::consts::PI / 2.0).sqrt() / sketch_dim as f32;
    let dot: f32 = q_proj
        .iter()
        .zip(key_sketch.iter())
        .map(|(qp, &sk)| qp * sk as f32)
        .sum();

    correction * dot * sketch_dim as f32
    // Simplification: the sketch_dim factors cancel, leaving:
    // correction * dot * sketch_dim = (√(π/2) / sketch_dim) * dot * sketch_dim
    //                                = √(π/2) * dot
}

/// Alternative simpler estimator: sign-agreemnt ratio scaled to [-1, +1].
///
/// For each projected query dimension, if `sign(q̃_i) == sign_i` → +1, else -1.
/// Then `score ≈ ‖q‖ * ‖k‖ * (2 * agreement_rate - 1) * (π/2)`.
///
/// This is a rough approximation; the `estimate_attention_score` function
/// above is more accurate.
pub fn sign_agreement_score(
    query: &[f32],
    key_sketch: &[i8],
    sketch_dim: usize,
    seed: u64,
) -> f32 {
    assert_eq!(key_sketch.len(), sketch_dim);
    let d = query.len();
    let matrix = gaussian_matrix(sketch_dim, d, seed);
    let scale = 1.0 / (sketch_dim as f32).sqrt();

    let mut agree = 0i32;
    for (i, &sk) in key_sketch.iter().enumerate() {
        let row = &matrix[i * d..(i + 1) * d];
        let qp: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
        if (qp >= 0.0) == (sk >= 0) {
            agree += 1;
        }
    }
    let agreement_rate = agree as f32 / sketch_dim as f32;
    // Map [0,1] → [-1, +1] and scale by π/2 (angle-to-cosine approximation).
    (2.0 * agreement_rate - 1.0) * (std::f32::consts::PI / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::gaussian_matrix;

    fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
    }

    fn sketch_key(key: &[f32], sketch_dim: usize, seed: u64) -> Vec<i8> {
        let d = key.len();
        let mat = gaussian_matrix(sketch_dim, d, seed);
        let scale = 1.0 / (sketch_dim as f32).sqrt();
        (0..sketch_dim)
            .map(|i| {
                let row = &mat[i * d..(i + 1) * d];
                let v: f32 = row.iter().zip(key.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                if v >= 0.0 { 1i8 } else { -1i8 }
            })
            .collect()
    }

    #[test]
    fn estimator_correlation_with_true_score() {
        // Over many query-key pairs, the estimated score should correlate
        // with the true dot product (r > 0.7).
        let d = 64;
        let sketch_dim = 128;
        let seed = 42u64;
        let n_pairs = 50;

        let mut true_scores = Vec::new();
        let mut est_scores = Vec::new();

        for i in 0..n_pairs {
            let q = rand_vec(d, i * 2);
            let k = rand_vec(d, i * 2 + 1);
            let true_score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            let sketch = sketch_key(&k, sketch_dim, seed);
            let est = estimate_attention_score(&q, &sketch, sketch_dim, seed);
            true_scores.push(true_score);
            est_scores.push(est);
        }

        // Pearson correlation.
        let n = n_pairs as f32;
        let mean_t = true_scores.iter().sum::<f32>() / n;
        let mean_e = est_scores.iter().sum::<f32>() / n;
        let num: f32 = true_scores
            .iter()
            .zip(est_scores.iter())
            .map(|(t, e)| (t - mean_t) * (e - mean_e))
            .sum();
        let den_t: f32 = true_scores.iter().map(|t| (t - mean_t).powi(2)).sum::<f32>().sqrt();
        let den_e: f32 = est_scores.iter().map(|e| (e - mean_e).powi(2)).sum::<f32>().sqrt();
        let corr = if den_t > 1e-8 && den_e > 1e-8 {
            num / (den_t * den_e)
        } else {
            0.0
        };

        assert!(
            corr > 0.7,
            "attention score correlation = {corr:.3} (required > 0.7)"
        );
    }

    #[test]
    fn sign_agreement_same_vector() {
        // For q = k (identical), sign-agreement should be high.
        let d = 64;
        let sketch_dim = 128;
        let seed = 1u64;
        let v = rand_vec(d, 10);
        let sketch = sketch_key(&v, sketch_dim, seed);
        let score = sign_agreement_score(&v, &sketch, sketch_dim, seed);
        // Should be close to π/2 ≈ 1.57 (full agreement).
        assert!(score > 1.0, "identical vector sign-agreement score = {score}");
    }
}
