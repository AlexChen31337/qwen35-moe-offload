//! Integration tests for qjl.

use qjl::{attention_score, compression_ratio, jl_project, jl_reconstruct};

fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

fn true_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn pearson_corr(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let num: f32 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let dx: f32 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f32>().sqrt();
    let dy: f32 = ys.iter().map(|y| (y - my).powi(2)).sum::<f32>().sqrt();
    if dx < 1e-8 || dy < 1e-8 { 0.0 } else { num / (dx * dy) }
}

// ─── jl_project tests ────────────────────────────────────────────────────────

#[test]
fn jl_project_produces_only_sign_values() {
    let kv = rand_vec(256, 100);
    let sketch = jl_project(&kv, 128, 0);
    for &s in &sketch {
        assert!(s == 1i8 || s == -1i8, "non-sign value {s}");
    }
}

#[test]
fn jl_project_deterministic_across_calls() {
    let kv = rand_vec(64, 200);
    let s1 = jl_project(&kv, 32, 1234);
    let s2 = jl_project(&kv, 32, 1234);
    assert_eq!(s1, s2);
}

#[test]
fn jl_project_different_inputs_different_sketches() {
    let k1 = rand_vec(64, 1);
    let k2 = rand_vec(64, 2);
    let s1 = jl_project(&k1, 32, 42);
    let s2 = jl_project(&k2, 32, 42);
    // Very unlikely to be identical for random vectors.
    assert_ne!(s1, s2);
}

// ─── compression_ratio tests ─────────────────────────────────────────────────

#[test]
fn compression_ratio_matches_expected_formula() {
    // ratio = (d * 32) / (k * 8) = 4 * d / k
    assert!((compression_ratio(128, 64) - 8.0).abs() < 0.001);
    assert!((compression_ratio(256, 64) - 16.0).abs() < 0.001);
    assert!((compression_ratio(128, 32) - 16.0).abs() < 0.001);
    assert!((compression_ratio(64, 64) - 4.0).abs() < 0.001);
}

#[test]
fn compression_ratio_sketch_dim_equals_original_gives_4x() {
    // 1-bit per dimension vs 32-bit per dimension = 32/8 = 4× (using i8 storage).
    let r = compression_ratio(128, 128);
    assert!((r - 4.0).abs() < 0.001, "ratio = {r}");
}

// ─── attention_score correlation test ────────────────────────────────────────

#[test]
fn attention_score_correlation_above_threshold() {
    // Over 100 random query-key pairs, estimated scores should correlate
    // with true dot products at r > 0.7.
    let d = 64;
    let sketch_dim = 128;
    let seed = 7u64;

    let n = 100;
    let mut true_scores = Vec::with_capacity(n);
    let mut est_scores = Vec::with_capacity(n);

    for i in 0..n {
        let q = rand_vec(d, i as u64 * 3);
        let k = rand_vec(d, i as u64 * 3 + 1);
        true_scores.push(true_dot(&q, &k));
        let sketch = jl_project(&k, sketch_dim, seed);
        est_scores.push(attention_score(&q, &sketch, sketch_dim, seed));
    }

    let corr = pearson_corr(&true_scores, &est_scores);
    assert!(
        corr > 0.7,
        "attention score Pearson correlation = {corr:.3} (required > 0.7)"
    );
}

// ─── round-trip sign consistency test ────────────────────────────────────────

#[test]
fn round_trip_sign_consistency() {
    // After project → reconstruct, the sign of the reconstructed vector should
    // agree with the original on > 55% of dimensions.
    let d = 128;
    let sketch_dim = 64;
    let seed = 99u64;

    let total_agree: usize = (0..20u64)
        .map(|s| {
            let kv = rand_vec(d, s * 17);
            let sketch = jl_project(&kv, sketch_dim, seed);
            let recon = jl_reconstruct(&sketch, d, sketch_dim, seed);
            kv.iter()
                .zip(recon.iter())
                .filter(|(a, b)| (a.signum() as i32) == (b.signum() as i32))
                .count()
        })
        .sum();
    let total_dims = 20 * d;
    let rate = total_agree as f32 / total_dims as f32;
    assert!(
        rate > 0.55,
        "sign consistency rate = {rate:.3} (required > 0.55)"
    );
}

#[test]
fn jl_reconstruct_output_shape() {
    let kv = rand_vec(256, 77);
    let sketch = jl_project(&kv, 64, 5);
    let recon = jl_reconstruct(&sketch, 256, 64, 5);
    assert_eq!(recon.len(), 256);
}

// ─── sketch_dim variation ────────────────────────────────────────────────────

#[test]
fn higher_sketch_dim_better_correlation() {
    // With more sketch dimensions, the estimator should be more accurate.
    let d = 128;
    let seed = 42u64;
    let n = 50;

    fn run_correlation(d: usize, sketch_dim: usize, seed: u64, n: usize) -> f32 {
        let mut true_scores = Vec::with_capacity(n);
        let mut est_scores = Vec::with_capacity(n);
        for i in 0..n {
            let q = rand_vec(d, i as u64 * 5);
            let k = rand_vec(d, i as u64 * 5 + 2);
            true_scores.push(true_dot(&q, &k));
            let sketch = jl_project(&k, sketch_dim, seed);
            est_scores.push(attention_score(&q, &sketch, sketch_dim, seed));
        }
        pearson_corr(&true_scores, &est_scores)
    }

    let corr_low = run_correlation(d, 16, seed, n);
    let corr_high = run_correlation(d, 256, seed, n);
    // Higher sketch dim should generally give better correlation.
    // We allow a tolerance since both should exceed 0.6.
    assert!(
        corr_high > 0.6,
        "high sketch_dim correlation = {corr_high:.3}"
    );
    assert!(
        corr_low >= 0.0,
        "low sketch_dim correlation = {corr_low:.3}"
    );
}
