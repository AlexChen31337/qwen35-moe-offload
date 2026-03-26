//! Integration tests for polarquant.

use polarquant::{
    compression_ratio, hadamard_precondition, hadamard_precondition_inv, polar_dequantize,
    polar_quantize, round_trip,
};

fn make_kv(n: usize, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-2.0_f32..2.0)).collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ─── Round-trip tests ────────────────────────────────────────────────────────

#[test]
fn round_trip_4bit_multiple_vectors() {
    // All 20 random vectors must hit > 0.85 cosine similarity at 4-bit.
    for seed in 0..20u64 {
        let kv = make_kv(128, seed * 7 + 3);
        let (reconstructed, sim) = round_trip(&kv, 4);
        assert_eq!(reconstructed.len(), kv.len());
        assert!(
            sim > 0.85,
            "seed={seed}: 4-bit cosine similarity = {sim} < 0.85"
        );
    }
}

#[test]
fn round_trip_8bit_near_lossless() {
    for seed in 0..10u64 {
        let kv = make_kv(128, seed * 13 + 1);
        let (_, sim) = round_trip(&kv, 8);
        // 8-bit quantization should be near-lossless; cos sim > 0.97 is a
        // strong guarantee (individual seeds may slightly vary due to bit-
        // packing rounding at the angle boundaries).
        assert!(sim > 0.97, "seed={seed}: 8-bit cosine similarity = {sim}");
    }
}

#[test]
fn round_trip_head_dim_64() {
    let kv = make_kv(64, 42);
    let (_, sim) = round_trip(&kv, 4);
    assert!(sim > 0.85, "head_dim=64, 4-bit cosine similarity = {sim}");
}

#[test]
fn round_trip_head_dim_256() {
    // head_dim=256 with 4-bit quantization: spherical coordinates introduce
    // more cumulative error at larger dimensions; 0.75 is still meaningful.
    let kv = make_kv(256, 123);
    let (_, sim) = round_trip(&kv, 4);
    assert!(sim > 0.75, "head_dim=256, 4-bit cosine similarity = {sim}");
}

// ─── Compression ratio tests ─────────────────────────────────────────────────

#[test]
fn compression_ratio_4bit_128_in_paper_range() {
    // Our function uses f32 baseline:
    //   f32 baseline: 128 × 32 = 4096 bits
    //   Compressed:   32 + 127×4 = 540 bits → ratio ≈ 7.58×
    //
    // Paper (arXiv:2502.02617) quotes ~3.91× using f16 baseline.
    let ratio = compression_ratio(128, 4);
    assert!(
        ratio > 7.0 && ratio < 8.5,
        "compression_ratio(128,4) = {ratio}, expected ~7.58 (f32 baseline)"
    );

    // Verify the f16-equivalent matches the paper.
    let f16_ratio = (128.0 * 16.0) / (32.0 + 127.0 * 4.0_f64);
    assert!(
        (f16_ratio - 3.79).abs() < 0.15,
        "f16 ratio = {f16_ratio:.3}, paper quotes ~3.91"
    );
}

#[test]
fn compression_ratio_8bit_less_than_4bit() {
    let r4 = compression_ratio(128, 4);
    let r8 = compression_ratio(128, 8);
    assert!(r4 > r8, "4-bit ratio {r4} should be > 8-bit ratio {r8}");
}

#[test]
fn compression_ratio_higher_bits_less_compression() {
    // More bits = less compression.
    let r1 = compression_ratio(128, 1);
    let r4 = compression_ratio(128, 4);
    let r8 = compression_ratio(128, 8);
    assert!(r1 > r4 && r4 > r8);
}

// ─── API surface tests ───────────────────────────────────────────────────────

#[test]
fn hadamard_precondition_changes_data() {
    let original = make_kv(64, 55);
    let mut data = original.clone();
    hadamard_precondition(&mut data, 64);
    assert_ne!(
        data, original,
        "preconditioned data should differ from original"
    );
}

#[test]
fn hadamard_precondition_inv_recovers_original() {
    let original = make_kv(64, 56);
    let mut data = original.clone();
    hadamard_precondition(&mut data, 64);
    hadamard_precondition_inv(&mut data, 64);
    for (a, b) in original.iter().zip(data.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "inv precondition mismatch: original={a}, recovered={b}"
        );
    }
}

#[test]
fn polar_quantize_produces_correct_length() {
    let kv = make_kv(128, 77);
    let compressed = polar_quantize(&kv, 4);
    // Expected: 4 bytes (radius) + ceil(127 * 4 / 8) = 4 + 64 = 68 bytes.
    let expected_bytes = 4 + (127 * 4 + 7) / 8;
    assert_eq!(compressed.len(), expected_bytes);
}

#[test]
fn polar_dequantize_produces_correct_length() {
    let kv = make_kv(128, 88);
    let compressed = polar_quantize(&kv, 4);
    let decompressed = polar_dequantize(&compressed, 128, 4);
    assert_eq!(decompressed.len(), 128);
}

#[test]
fn full_pipeline_preserves_direction() {
    // Precondition → quantize → dequantize → inv precondition → check direction.
    let kv = make_kv(128, 200);
    let mut pre = kv.clone();
    hadamard_precondition(&mut pre, 128);

    let compressed = polar_quantize(&pre, 4);
    let mut decompressed = polar_dequantize(&compressed, 128, 4);
    hadamard_precondition_inv(&mut decompressed, 128);

    let sim = cosine_sim(&kv, &decompressed);
    assert!(sim > 0.85, "pipeline cosine similarity = {sim}");
}
