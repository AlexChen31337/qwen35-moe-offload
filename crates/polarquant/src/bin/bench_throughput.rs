// Standalone Rust throughput benchmark for PolarQuant compression
// Run: cargo run --release -p polarquant --bin bench_throughput

use polarquant::{hadamard_precondition, hadamard_precondition_inv, polar_quantize, polar_dequantize, compression_ratio};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const HEAD_DIM: usize = 128;
const N_LAYERS: usize = 64;
const N_KV_HEADS: usize = 4;
const TOTAL_HEADS: usize = N_LAYERS * 2 * N_KV_HEADS; // 512

fn make_kv(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-2.0_f32..2.0)).collect()
}

fn bench_single_head(bits: u8, n_trials: usize) -> (f64, f64, f64) {
    let vectors: Vec<Vec<f32>> = (0..n_trials)
        .map(|i| make_kv(HEAD_DIM, i as u64 * 7 + 3))
        .collect();

    // Warmup
    for v in vectors.iter().take(10) {
        let mut pre = v.clone();
        hadamard_precondition(&mut pre, HEAD_DIM);
        let c = polar_quantize(&pre, bits);
        let mut dec = polar_dequantize(&c, HEAD_DIM, bits);
        hadamard_precondition_inv(&mut dec, HEAD_DIM);
    }

    // Compress
    let t0 = Instant::now();
    let mut compressed = Vec::with_capacity(n_trials);
    for v in &vectors {
        let mut pre = v.clone();
        hadamard_precondition(&mut pre, HEAD_DIM);
        compressed.push(polar_quantize(&pre, bits));
    }
    let t_compress = t0.elapsed().as_secs_f64();

    // Decompress + quality
    let t0 = Instant::now();
    let mut cosine_sims = Vec::with_capacity(n_trials);
    for (c, orig) in compressed.iter().zip(vectors.iter()) {
        let mut dec = polar_dequantize(c, HEAD_DIM, bits);
        hadamard_precondition_inv(&mut dec, HEAD_DIM);

        let dot: f32 = orig.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 1e-12 && nb > 1e-12 {
            cosine_sims.push((dot / (na * nb)).clamp(-1.0, 1.0));
        }
    }
    let t_decompress = t0.elapsed().as_secs_f64();

    let avg_sim = cosine_sims.iter().sum::<f32>() / cosine_sims.len() as f32;
    (
        n_trials as f64 / t_compress,
        n_trials as f64 / t_decompress,
        avg_sim as f64,
    )
}

fn bench_full_token(bits: u8, n_tokens: usize) -> f64 {
    let vectors: Vec<Vec<f32>> = (0..TOTAL_HEADS * n_tokens)
        .map(|i| make_kv(HEAD_DIM, i as u64))
        .collect();

    let t0 = Instant::now();
    for v in &vectors {
        let mut pre = v.clone();
        hadamard_precondition(&mut pre, HEAD_DIM);
        let _c = polar_quantize(&pre, bits);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed / n_tokens as f64) * 1000.0 // ms per token
}

fn main() {
    println!("========================================================================");
    println!("Rust PolarQuant Throughput Benchmark (--release)");
    println!("Model: Qwen3.5-35B-A3B (head_dim={}, n_layers={}, n_kv_heads={})", HEAD_DIM, N_LAYERS, N_KV_HEADS);
    println!("Total KV heads per token: {}", TOTAL_HEADS);
    println!("========================================================================");
    println!();

    let n_trials = 10000;
    println!("Part 1: Per-Head Throughput ({} trials)", n_trials);
    println!("{:>4} {:>7} {:>12} {:>14} {:>8}", "bits", "ratio", "compress/s", "decompress/s", "cos_sim");
    println!("{}", "-".repeat(49));

    for bits in [2u8, 3, 4, 5, 6, 8] {
        let ratio = compression_ratio(HEAD_DIM, bits);
        let (cr, dr, sim) = bench_single_head(bits, n_trials);
        println!("{:>4} {:>6.1}x {:>12.0} {:>14.0} {:>8.4}", bits, ratio, cr, dr, sim);
    }

    println!();
    println!("Part 2: Full Token Overhead (compressing all {} heads)", TOTAL_HEADS);
    println!("  Baseline: 12.114 tok/s → budget = {:.1} ms/token", 1000.0_f64 / 12.114);
    println!("{:>4} {:>12} {:>10} {:>12}", "bits", "per_tok_ms", "overhead%", "max_tok/s");
    println!("{}", "-".repeat(42));

    let n_tokens = 100;
    let budget_ms = 1000.0_f64 / 12.114;
    for bits in [2u8, 3, 4, 5, 6, 8] {
        let per_tok_ms = bench_full_token(bits, n_tokens);
        let overhead_pct = (per_tok_ms / budget_ms) * 100.0;
        let max_toks = 1000.0 / (budget_ms + per_tok_ms);
        println!("{:>4} {:>12.2} {:>9.1}% {:>12.3}", bits, per_tok_ms, overhead_pct, max_toks);
    }

    // Python comparison
    println!();
    println!("Part 3: Python vs Rust Speedup");
    println!("  Python per-token (4-bit): ~1380 ms");
    println!("  Rust per-token (4-bit):   see above");
    println!("  Expected speedup: 50-200x");
}
