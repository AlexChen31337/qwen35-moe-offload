//! Walsh-Hadamard Transform (WHT) for KV vector preconditiong.
//!
//! The Hadamard preconditioner randomises the magnitude distribution of
//! KV elements before quantization.  Multiplying by a random ±1 diagonal
//! (the "randomised" part) and then applying the Walsh-Hadamard transform
//! spreads energy uniformly, which tightens the angular quantization grid
//! used by PolarQuant.
//!
//! Reference: arXiv:2502.02617, §3.1.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Apply an in-place normalised Walsh-Hadamard transform to `data`.
///
/// `data.len()` must be a power of two.  The transform is unitary:
/// dividing every output element by `sqrt(n)` preserves vector norms.
///
/// # Panics
/// Panics if `data.len()` is not a power of two.
pub fn wht_inplace(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "WHT length must be a power of two, got {n}");

    let mut h = 1usize;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    // Normalise so the transform is unitary (preserves dot-products).
    let scale = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Build a deterministic ±1 sign vector from `seed` with length `n`.
pub fn random_signs(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| if rng.gen::<bool>() { 1.0_f32 } else { -1.0_f32 })
        .collect()
}

/// Apply a randomised Hadamard preconditioner in-place.
///
/// If `n` (the used slice length) is smaller than the next power of two,
/// the vector is zero-padded, transformed, then truncated.  In practice
/// the caller should pad before calling if they need a full-length output.
///
/// Steps:
/// 1. Element-wise multiply by random ±1 diagonal `D`.
/// 2. Apply normalised WHT.
///
/// Both steps are seeded from `seed` for reproducibility.
pub fn hadamard_precondition_seeded(data: &mut Vec<f32>, seed: u64) {
    let n = data.len();
    // Pad to next power of two.
    let padded = n.next_power_of_two();
    data.resize(padded, 0.0);

    // Step 1: random sign flip.
    let signs = random_signs(padded, seed);
    for (v, s) in data.iter_mut().zip(signs.iter()) {
        *v *= s;
    }

    // Step 2: WHT.
    wht_inplace(data);

    // Trim back to original length.
    data.truncate(n);
}

/// Inverse of `hadamard_precondition_seeded`.
///
/// WHT is its own inverse (up to scaling), so we just re-apply it and
/// undo the sign flip.
pub fn hadamard_precondition_inv_seeded(data: &mut Vec<f32>, seed: u64) {
    let n = data.len();
    let padded = n.next_power_of_two();
    data.resize(padded, 0.0);

    // Step 1: inverse WHT (same as forward WHT — it's symmetric).
    wht_inplace(data);

    // Step 2: undo the sign flip (multiply by the same ±1 vector).
    let signs = random_signs(padded, seed);
    for (v, s) in data.iter_mut().zip(signs.iter()) {
        *v *= s;
    }

    data.truncate(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn wht_round_trip() {
        let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        wht_inplace(&mut data);
        wht_inplace(&mut data);
        // Two WHTs = identity (each is unitary, so two = scaling by 1 after
        // normalisation).
        for (a, b) in original.iter().zip(data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn wht_preserves_norm() {
        let data = vec![1.0_f32, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut transformed = data.clone();
        wht_inplace(&mut transformed);
        let norm_after: f32 = transformed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-4);
    }

    #[test]
    fn hadamard_precondition_round_trip() {
        let original = vec![0.5_f32, -0.3, 1.2, -0.7, 0.1, 0.9, -0.4, 0.6];
        let mut data = original.clone();
        hadamard_precondition_seeded(&mut data, 42);
        hadamard_precondition_inv_seeded(&mut data, 42);
        for (a, b) in original.iter().zip(data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }
}
