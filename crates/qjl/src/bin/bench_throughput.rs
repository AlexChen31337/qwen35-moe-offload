// Standalone Rust throughput benchmark for QJL compression
// Run: cargo run --release -p qjl --bin bench_throughput

use qjl::{jl_project, jl_reconstruct, attention_score, compression_ratio};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const HEAD_DIM: usize = 128;
const N_LAYERS: usize = 64;
const N_KV_HEADS: usize = 4;
const TOTAL_K_HEADS: usize = N_LAYERS * N_KV_HEADS; // 256 (only K needs sketching)

fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

fn bench_projection(sketch_dim: usize, n_trials: usize) -> (f64, f64) {
    let vectors: Vec<Vec<f32>> = (0..n_trials)
        .map(|i| rand_vec(HEAD_DIM, i as u64 * 3 + 1))
        .collect();

    // Warmup
    for v in vectors.iter().take(10) {
        let _ = jl_project(v, sketch_dim, 42);
    }

    // Project
    let t0 = Instant::now();
    let mut sketches = Vec::with_capacity(n_trials);
    for v in &vectors {
        sketches.push(jl_project(v, sketch_dim, 42));
    }
    let t_project = t0.elapsed().as_secs_f64();

    // Reconstruct
    let t0 = Instant::now();
    for s in &sketches {
        let _ = jl_reconstruct(s, HEAD_DIM, sketch_dim, 42);
    }
    let t_reconstruct = t0.elapsed().as_secs_f64();

    (
        n_trials as f64 / t_project,
        n_trials as f64 / t_reconstruct,
    )
}

fn bench_attention_quality(sketch_dim: usize, n_pairs: usize) -> f64 {
    let seed = 7u64;
    let mut true_scores = Vec::with_capacity(n_pairs);
    let mut est_scores = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let q = rand_vec(HEAD_DIM, i as u64 * 5);
        let k = rand_vec(HEAD_DIM, i as u64 * 5 + 2);
        let true_score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        true_scores.push(true_score);

        let sketch = jl_project(&k, sketch_dim, seed);
        est_scores.push(attention_score(&q, &sketch, sketch_dim, seed));
    }

    // Pearson correlation
    let n = true_scores.len() as f32;
    let mx: f32 = true_scores.iter().sum::<f32>() / n;
    let my: f32 = est_scores.iter().sum::<f32>() / n;
    let num: f32 = true_scores.iter().zip(est_scores.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let dx: f32 = true_scores.iter().map(|x| (x - mx).powi(2)).sum::<f32>().sqrt();
    let dy: f32 = est_scores.iter().map(|y| (y - my).powi(2)).sum::<f32>().sqrt();
    if dx < 1e-8 || dy < 1e-8 { 0.0 } else { (num / (dx * dy)) as f64 }
}

fn bench_full_token(sketch_dim: usize, n_tokens: usize) -> f64 {
    let vectors: Vec<Vec<f32>> = (0..TOTAL_K_HEADS * n_tokens)
        .map(|i| rand_vec(HEAD_DIM, i as u64))
        .collect();

    let t0 = Instant::now();
    for v in &vectors {
        let _ = jl_project(v, sketch_dim, 42);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    (elapsed / n_tokens as f64) * 1000.0 // ms per token
}

fn bench_sketch_attention(sketch_dim: usize, seq_len: usize, n_queries: usize) -> (f64, f64) {
    let seed = 7u64;
    
    // Pre-compute key sketches
    let key_sketches: Vec<Vec<i8>> = (0..seq_len)
        .map(|i| {
            let k = rand_vec(HEAD_DIM, i as u64 * 11);
            jl_project(&k, sketch_dim, seed)
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| rand_vec(HEAD_DIM, i as u64 * 13 + 1000))
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| rand_vec(HEAD_DIM, i as u64 * 17 + 5000))
        .collect();

    // Sketch-based attention
    let t0 = Instant::now();
    for q in &queries {
        let scores: Vec<f32> = key_sketches.iter()
            .map(|ks| attention_score(q, ks, sketch_dim, seed))
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();
        // Weighted sum of values
        let mut _output = vec![0.0f32; HEAD_DIM];
        for (w, v) in weights.iter().zip(values.iter()) {
            for (o, vi) in _output.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
    }
    let t_sketch = t0.elapsed().as_secs_f64();

    // Full-precision attention
    let keys_full: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| rand_vec(HEAD_DIM, i as u64 * 11))
        .collect();
    let t0 = Instant::now();
    for q in &queries {
        let scores: Vec<f32> = keys_full.iter()
            .map(|k| q.iter().zip(k.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();
        let mut _output = vec![0.0f32; HEAD_DIM];
        for (w, v) in weights.iter().zip(values.iter()) {
            for (o, vi) in _output.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
    }
    let t_full = t0.elapsed().as_secs_f64();

    (
        (t_sketch / n_queries as f64) * 1000.0, // ms per query
        (t_full / n_queries as f64) * 1000.0,
    )
}

fn main() {
    println!("========================================================================");
    println!("Rust QJL Throughput Benchmark (--release)");
    println!("Model: Qwen3.5-35B-A3B (head_dim={}, n_layers={}, n_kv_heads={})", HEAD_DIM, N_LAYERS, N_KV_HEADS);
    println!("Total K heads per token: {}", TOTAL_K_HEADS);
    println!("========================================================================");
    println!();

    let n_trials = 10000;
    println!("Part 1: Per-Head Throughput ({} trials)", n_trials);
    println!("{:>7} {:>7} {:>12} {:>14} {:>9}", "sketch", "ratio", "project/s", "reconstruct/s", "Pearson r");
    println!("{}", "-".repeat(53));

    for sd in [32usize, 64, 128, 256] {
        let ratio = compression_ratio(HEAD_DIM, sd);
        let (pr, rr) = bench_projection(sd, n_trials);
        let corr = bench_attention_quality(sd, 500);
        println!("{:>7} {:>6.1}x {:>12.0} {:>14.0} {:>9.4}", sd, ratio, pr, rr, corr);
    }

    println!();
    println!("Part 2: Full Token Overhead (projecting all {} K heads)", TOTAL_K_HEADS);
    println!("  Baseline: 12.114 tok/s → budget = {:.1} ms/token", 1000.0_f64 / 12.114);
    println!("{:>7} {:>12} {:>10} {:>12}", "sketch", "per_tok_ms", "overhead%", "max_tok/s");
    println!("{}", "-".repeat(45));

    let n_tokens = 100;
    let budget_ms = 1000.0_f64 / 12.114;
    for sd in [32usize, 64, 128, 256] {
        let per_tok_ms = bench_full_token(sd, n_tokens);
        let overhead_pct = (per_tok_ms / budget_ms) * 100.0;
        let max_toks = 1000.0 / (budget_ms + per_tok_ms);
        println!("{:>7} {:>12.2} {:>9.1}% {:>12.3}", sd, per_tok_ms, overhead_pct, max_toks);
    }

    println!();
    println!("Part 3: Sketch Attention vs Full Attention Speed");
    println!("{:>7} {:>8} {:>12} {:>12} {:>8}", "sketch", "seq_len", "sketch_ms", "full_ms", "speedup");
    println!("{}", "-".repeat(51));

    for seq_len in [512, 2048, 8192, 32768] {
        for sd in [64, 128] {
            let (sk_ms, full_ms) = bench_sketch_attention(sd, seq_len, 5);
            let speedup = full_ms / sk_ms;
            println!("{:>7} {:>8} {:>12.2} {:>12.2} {:>7.3}x", sd, seq_len, sk_ms, full_ms, speedup);
        }
    }
}
