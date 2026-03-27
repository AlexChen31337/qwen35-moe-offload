// Comprehensive PolarQuant quality analysis across bit widths and head dimensions
// Run: cargo run --release -p polarquant --bin quality_sweep

use polarquant::{
    hadamard_precondition, hadamard_precondition_inv,
    polar_quantize, polar_dequantize, compression_ratio, round_trip,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N_SAMPLES: usize = 10000;

fn make_kv(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-2.0_f32..2.0)).collect()
}

fn statistics(values: &[f32]) -> (f32, f32, f32, f32, f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let p5 = sorted[(0.05 * sorted.len() as f64) as usize];
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(0.95 * sorted.len() as f64) as usize];
    (mean, std, min, p5, median, p95)
}

fn main() {
    println!("========================================================================");
    println!("PolarQuant Quality Sweep ({} samples per config)", N_SAMPLES);
    println!("========================================================================\n");

    // Sweep bit widths for head_dim=128 (Qwen3.5-35B-A3B)
    println!("--- head_dim=128 (Qwen3.5-35B-A3B) ---");
    println!("{:>4} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
             "bits", "ratio", "mean", "std", "min", "p5", "median", "p95");
    println!("{}", "-".repeat(60));

    for bits in 1..=8u8 {
        let mut sims = Vec::with_capacity(N_SAMPLES);
        for i in 0..N_SAMPLES {
            let kv = make_kv(128, i as u64 * 7 + 11);
            let (_, sim) = round_trip(&kv, bits);
            sims.push(sim);
        }
        let (mean, std, min, p5, med, p95) = statistics(&sims);
        let ratio = compression_ratio(128, bits);
        println!("{:>4} {:>6.1}x {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                 bits, ratio, mean, std, min, p5, med, p95);
    }

    // Different head dimensions
    println!("\n--- 4-bit PolarQuant across head dimensions ---");
    println!("{:>8} {:>7} {:>8} {:>8} {:>8} {:>8}",
             "head_dim", "ratio", "mean", "std", "min", "p5");
    println!("{}", "-".repeat(47));

    for head_dim in [32, 64, 128, 192, 256, 384, 512] {
        let mut sims = Vec::with_capacity(N_SAMPLES);
        for i in 0..N_SAMPLES {
            let kv = make_kv(head_dim, i as u64 * 13 + 5);
            let (_, sim) = round_trip(&kv, 4);
            sims.push(sim);
        }
        let (mean, std, min, p5, _, _) = statistics(&sims);
        let ratio = compression_ratio(head_dim, 4);
        println!("{:>8} {:>6.1}x {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                 head_dim, ratio, mean, std, min, p5);
    }

    // Simulated q8_0 input (values rounded to 256 levels)
    println!("\n--- PolarQuant on q8_0 dequantized input (simulated) ---");
    println!("  (What if KV cache is already in q8_0 and we PolarQuant compress further?)");
    println!("{:>4} {:>7} {:>12} {:>12} {:>12}",
             "bits", "ratio", "direct_sim", "via_q8_sim", "quality_loss");
    println!("{}", "-".repeat(55));

    for bits in [3u8, 4, 5, 6, 8] {
        let mut direct_sims = Vec::new();
        let mut q8_sims = Vec::new();

        for i in 0..5000usize {
            let kv = make_kv(128, i as u64 * 19 + 7);

            // Direct PolarQuant
            let (_, sim_direct) = round_trip(&kv, bits);
            direct_sims.push(sim_direct);

            // Simulate q8_0: quantize to 256 levels then dequantize
            let q8_kv: Vec<f32> = kv.iter().map(|&v| {
                let scale = 2.0_f32; // Typical q8_0 scale for randn values
                let q = (v / scale * 127.0).round().clamp(-127.0, 127.0);
                q * scale / 127.0
            }).collect();

            let (_, sim_q8) = round_trip(&q8_kv, bits);
            q8_sims.push(sim_q8);
        }

        let mean_d = direct_sims.iter().sum::<f32>() / direct_sims.len() as f32;
        let mean_q = q8_sims.iter().sum::<f32>() / q8_sims.len() as f32;
        let loss = mean_d - mean_q;
        println!("{:>4} {:>6.1}x {:>12.4} {:>12.4} {:>12.4}",
                 bits, compression_ratio(128, bits), mean_d, mean_q, loss);
    }

    // Error distribution analysis
    println!("\n--- Per-element error analysis (head_dim=128, 4-bit) ---");
    let mut all_errors = Vec::new();
    let mut all_rel_errors = Vec::new();
    for i in 0..1000usize {
        let kv = make_kv(128, i as u64 * 31);
        let (recon, _) = round_trip(&kv, 4);
        for (orig, rec) in kv.iter().zip(recon.iter()) {
            let err = (orig - rec).abs();
            all_errors.push(err);
            if orig.abs() > 0.01 {
                all_rel_errors.push(err / orig.abs());
            }
        }
    }
    all_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = all_errors.len();
    println!("  Absolute error: mean={:.4} median={:.4} p95={:.4} p99={:.4} max={:.4}",
             all_errors.iter().sum::<f32>() / n as f32,
             all_errors[n / 2],
             all_errors[(0.95 * n as f64) as usize],
             all_errors[(0.99 * n as f64) as usize],
             all_errors[n - 1]);
    
    let nr = all_rel_errors.len();
    println!("  Relative error: mean={:.4} median={:.4} p95={:.4} p99={:.4} max={:.4}",
             all_rel_errors.iter().sum::<f32>() / nr as f32,
             all_rel_errors[nr / 2],
             all_rel_errors[(0.95 * nr as f64) as usize],
             all_rel_errors[(0.99 * nr as f64) as usize],
             all_rel_errors[nr - 1]);
}
