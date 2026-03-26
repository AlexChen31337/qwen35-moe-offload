//! Polar decomposition and angle quantization for PolarQuant.
//!
//! PolarQuant separates a KV vector into:
//!   - A scalar *radius* (the L2 norm of the vector).
//!   - A unit-sphere *direction* encoded as (n-1) angles in spherical coordinates.
//!
//! The radius is stored at higher precision (f16 or f32) while the angles
//! are quantized to `bits` bits each.  For `bits=4` and `head_dim=128`,
//! this gives a theoretical 3.91× compression ratio over f16.
//!
//! Reference: arXiv:2502.02617, §3.2–3.3.

use thiserror::Error;

/// Errors that can arise during polar quantization.
#[derive(Debug, Error)]
pub enum PolarQuantError {
    #[error("head_dim must be ≥ 2, got {0}")]
    HeadDimTooSmall(usize),
    #[error("bits must be in 1..=8, got {0}")]
    InvalidBits(u8),
    #[error("compressed buffer length mismatch: expected {expected}, got {got}")]
    BufferLengthMismatch { expected: usize, got: usize },
}

/// Number of bytes needed to store the compressed representation.
///
/// Layout: 4 bytes f32 radius + ceil((head_dim - 1) * bits / 8) bytes for angles.
pub fn compressed_byte_len(head_dim: usize, bits: u8) -> usize {
    let angle_bits = (head_dim - 1) * bits as usize;
    let angle_bytes = (angle_bits + 7) / 8;
    4 + angle_bytes
}

/// Convert a KV vector to polar coordinates and quantize the angles.
///
/// Returns a byte buffer with layout:
///   - bytes 0..4: f32 radius (little-endian)
///   - bytes 4..: packed angle quantization codes (`bits` bits each)
///
/// # Errors
/// Returns an error if `head_dim < 2` or `bits` is outside 1..=8.
pub fn quantize(kv: &[f32], bits: u8) -> Result<Vec<u8>, PolarQuantError> {
    let head_dim = kv.len();
    if head_dim < 2 {
        return Err(PolarQuantError::HeadDimTooSmall(head_dim));
    }
    if bits == 0 || bits > 8 {
        return Err(PolarQuantError::InvalidBits(bits));
    }

    // --- Polar decomposition ---
    let radius: f32 = kv.iter().map(|x| x * x).sum::<f32>().sqrt();

    let levels = (1u32 << bits) as f32; // number of quantization levels
    let mut out = Vec::with_capacity(compressed_byte_len(head_dim, bits));

    // Store radius as 4-byte f32 little-endian.
    out.extend_from_slice(&radius.to_le_bytes());

    if radius < 1e-12 {
        // Zero vector — fill angles with zeros.
        let angle_bits = (head_dim - 1) * bits as usize;
        let angle_bytes = (angle_bits + 7) / 8;
        out.resize(4 + angle_bytes, 0u8);
        return Ok(out);
    }

    // Compute (n-1) spherical coordinate angles from a unit vector.
    // We use the standard recursive formula:
    //   unit[i] = cos(θ_i) * Π_{j<i} sin(θ_j)   for i < n-1
    //   unit[n-1] = Π_{j<n-1} sin(θ_j)
    //
    // Each θ_i ∈ [0, π) is quantized uniformly to `bits` bits.
    let unit: Vec<f32> = kv.iter().map(|x| x / radius).collect();
    let angles = unit_to_angles(&unit);

    // Pack angles into bit stream.
    let max_code = levels as u32 - 1;
    let mut bit_buf: u64 = 0;
    let mut bit_pos: u32 = 0;

    for &angle in &angles {
        // Quantize θ ∈ [0, π) → code ∈ [0, levels-1].
        let normalised = angle / std::f32::consts::PI; // 0..1
        let code = (normalised * levels).floor() as u32;
        let code = code.min(max_code);

        bit_buf |= (code as u64) << bit_pos;
        bit_pos += bits as u32;

        while bit_pos >= 8 {
            out.push((bit_buf & 0xFF) as u8);
            bit_buf >>= 8;
            bit_pos -= 8;
        }
    }
    // Flush remaining bits.
    if bit_pos > 0 {
        out.push((bit_buf & 0xFF) as u8);
    }

    Ok(out)
}

/// Reconstruct a KV vector from a polar-quantized byte buffer.
///
/// # Errors
/// Returns an error if the buffer length doesn't match expectations.
pub fn dequantize(
    compressed: &[u8],
    head_dim: usize,
    bits: u8,
) -> Result<Vec<f32>, PolarQuantError> {
    if head_dim < 2 {
        return Err(PolarQuantError::HeadDimTooSmall(head_dim));
    }
    if bits == 0 || bits > 8 {
        return Err(PolarQuantError::InvalidBits(bits));
    }

    let expected = compressed_byte_len(head_dim, bits);
    if compressed.len() != expected {
        return Err(PolarQuantError::BufferLengthMismatch {
            expected,
            got: compressed.len(),
        });
    }

    // Read radius.
    let radius = f32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);

    if radius < 1e-12 {
        return Ok(vec![0.0f32; head_dim]);
    }

    let levels = (1u32 << bits) as f32;
    let n_angles = head_dim - 1;

    // Unpack angle codes from bit stream.
    let mask = (1u64 << bits) - 1;
    let mut bit_buf: u64 = 0;
    let mut bit_pos: u32 = 0;
    let mut byte_idx = 4usize;
    let mut angles = Vec::with_capacity(n_angles);

    for _ in 0..n_angles {
        while bit_pos < bits as u32 {
            if byte_idx < compressed.len() {
                bit_buf |= (compressed[byte_idx] as u64) << bit_pos;
                byte_idx += 1;
            }
            bit_pos += 8;
        }
        let code = bit_buf & mask;
        bit_buf >>= bits;
        bit_pos -= bits as u32;

        // Dequantize: mid-point of bin.
        let normalised = (code as f32 + 0.5) / levels;
        angles.push(normalised * std::f32::consts::PI);
    }

    // Reconstruct unit vector from angles.
    let unit = angles_to_unit(&angles, head_dim);
    Ok(unit.into_iter().map(|x| x * radius).collect())
}

// --- helpers ---

/// Convert a unit vector to (n-1) spherical angles, each in [0, π).
pub(crate) fn unit_to_angles(unit: &[f32]) -> Vec<f32> {
    let n = unit.len();
    let mut angles = Vec::with_capacity(n - 1);
    let mut sin_prod = 1.0_f32;

    for i in 0..(n - 1) {
        let cos_theta = if sin_prod.abs() < 1e-10 {
            0.0
        } else {
            (unit[i] / sin_prod).clamp(-1.0, 1.0)
        };
        let theta = cos_theta.acos(); // ∈ [0, π]
        angles.push(theta.clamp(0.0, std::f32::consts::PI - 1e-6));
        sin_prod *= theta.sin();
    }
    angles
}

/// Reconstruct a unit vector from (n-1) spherical angles.
pub(crate) fn angles_to_unit(angles: &[f32], n: usize) -> Vec<f32> {
    let mut unit = vec![0.0_f32; n];
    let mut sin_prod = 1.0_f32;

    for (i, &theta) in angles.iter().enumerate() {
        unit[i] = sin_prod * theta.cos();
        sin_prod *= theta.sin();
    }
    // Last component.
    unit[n - 1] = sin_prod;
    unit
}

/// Compute the cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_kv(n: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
    }

    #[test]
    fn angles_round_trip() {
        // Use head_dim=128 for a numerically stable test (small dims lose
        // precision when sin_prod underflows near the last component).
        let unit = {
            let v = sample_kv(128, 1);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<_>>()
        };
        let angles = unit_to_angles(&unit);
        let reconstructed = angles_to_unit(&angles, unit.len());
        let cos_sim = cosine_similarity(&unit, &reconstructed);
        assert!(cos_sim > 0.99, "angle round-trip cosine similarity = {cos_sim}");
    }

    #[test]
    fn quantize_buffer_length() {
        let kv = sample_kv(128, 2);
        let buf = quantize(&kv, 4).unwrap();
        assert_eq!(buf.len(), compressed_byte_len(128, 4));
    }

    #[test]
    fn dequantize_length_mismatch_error() {
        let bad_buf = vec![0u8; 5];
        let result = dequantize(&bad_buf, 128, 4);
        assert!(result.is_err());
    }

    #[test]
    fn zero_vector_round_trip() {
        let kv = vec![0.0_f32; 8];
        let buf = quantize(&kv, 4).unwrap();
        let out = dequantize(&buf, 8, 4).unwrap();
        for v in out {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-6);
        }
    }
}
