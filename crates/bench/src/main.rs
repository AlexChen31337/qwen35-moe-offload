// Manual FFI bindings for llama.cpp — no bindgen, no Python overhead
#![allow(non_camel_case_types)]

use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use clap::Parser;

// ============================================================================
// FFI types and functions — minimal subset for benchmarking
// ============================================================================

pub type llama_token = i32;

// Opaque pointers
#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

// ggml_type enum values we care about
pub const GGML_TYPE_F32: i32 = 0;
pub const GGML_TYPE_F16: i32 = 1;
pub const GGML_TYPE_Q4_0: i32 = 2;
pub const GGML_TYPE_Q5_0: i32 = 6;
pub const GGML_TYPE_Q5_1: i32 = 7;
pub const GGML_TYPE_Q8_0: i32 = 8;
pub const GGML_TYPE_IQ4_NL: i32 = 20;

// flash_attn_type
pub const LLAMA_FLASH_ATTN_DISABLED: i32 = 0;
pub const LLAMA_FLASH_ATTN_ENABLED: i32 = 1;

// Model params — must match C struct layout exactly
// We use a big enough buffer and access known-offset fields
#[repr(C)]
pub struct llama_model_params {
    // This struct has pointers and ints. We'll use the default_params function
    // and only override what we need via raw pointer arithmetic.
    _data: [u8; 256], // oversized buffer
}

// Context params — similarly
#[repr(C)]
pub struct llama_context_params {
    _data: [u8; 512], // oversized buffer
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

#[repr(C)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut i32,      // llama_pos = i32
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,  // llama_seq_id = i32
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct llama_perf_context_data {
    pub t_start_ms: f64,
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
    pub n_reused: i32,
}

extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    pub fn llama_model_load_from_file(
        path_model: *const libc::c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);

    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;

    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const libc::c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_batch_get_one(
        tokens: *mut llama_token,
        n_tokens: i32,
    ) -> llama_batch;

    pub fn llama_decode(
        ctx: *mut llama_context,
        batch: llama_batch,
    ) -> i32;

    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_free(smpl: *mut llama_sampler);

    pub fn llama_perf_context(ctx: *const llama_context) -> llama_perf_context_data;
    pub fn llama_perf_context_print(ctx: *const llama_context);
    pub fn llama_perf_context_reset(ctx: *mut llama_context);

    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
}

// ============================================================================
// We need to set fields in the opaque structs. Since we know the struct layout
// from the header, we'll do it via a helper that pokes the right offsets.
// But this is fragile. Better approach: use a C shim.
// ============================================================================

// Instead of raw offset poking, we'll write a tiny C wrapper.
// Actually — simplest approach: write a Python script that calls the C API
// through ctypes but with zero overhead in the hot loop. 
//
// But the whole point is NO PYTHON. So let's use a different approach:
// Write a small C file that creates the params structs for us.

// Actually — the easiest reliable approach is to just write a C shim file
// that we compile and link.

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug, Clone)]
#[command(name = "bench", about = "Phase 6 Rust benchmark runner for llama.cpp")]
struct Args {
    #[arg(long, default_value = "/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf")]
    model: String,

    #[arg(long, default_value_t = 16)]
    n_gpu: i32,

    #[arg(long, default_value_t = 512)]
    n_ctx: u32,

    #[arg(long, default_value_t = 64)]
    n_batch: u32,

    #[arg(long, default_value_t = 64)]
    n_ubatch: u32,

    /// KV type: 0=f32, 1=f16, 8=q8_0, 2=q4_0, 20=iq4_nl, 6=q5_0
    #[arg(long, default_value_t = 8)]
    type_k: i32,

    #[arg(long, default_value_t = 8)]
    type_v: i32,

    /// Flash attention: 0=disabled, 1=enabled
    #[arg(long, default_value_t = 1)]
    flash_attn: i32,

    #[arg(long, default_value_t = 128)]
    n_gen: i32,

    #[arg(long, default_value_t = 1)]
    exp_id: i32,

    #[arg(long, default_value = "rust_bench")]
    label: String,

    #[arg(long, default_value = "/tmp/qwen35-moe-offload/results_phase6.tsv")]
    results_file: String,

    #[arg(long, default_value_t = 4)]
    n_threads: i32,

    #[arg(long, default_value = "The meaning of life is")]
    prompt: String,
}

fn ggml_type_name(t: i32) -> &'static str {
    match t {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        20 => "iq4_nl",
        _ => "unknown",
    }
}

fn get_vram_mb() -> u64 {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<u64>().unwrap_or(0)
        }
        Err(_) => 0,
    }
}

fn append_result(args: &Args, tok_s: f64, vram_mb: u64, status: &str, notes: &str) {
    let kv_name = ggml_type_name(args.type_k);
    let line = format!(
        "{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        args.exp_id, tok_s, vram_mb, args.n_ctx,
        kv_name, kv_name,
        if args.flash_attn == 1 { "True" } else { "False" },
        args.n_gpu, args.n_batch, args.n_ubatch,
        args.label, status, notes
    );

    let needs_header = !std::path::Path::new(&args.results_file).exists()
        || std::fs::metadata(&args.results_file).map(|m| m.len() == 0).unwrap_or(true);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.results_file)
        .expect("Failed to open results file");

    if needs_header {
        writeln!(file, "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tlabel\tstatus\tnotes").unwrap();
    }

    file.write_all(line.as_bytes()).expect("Failed to write result");

    println!("\n=== RESULT ===");
    println!("exp_id={} tok/s={:.3} vram={}MB status={}", args.exp_id, tok_s, vram_mb, status);
    println!("notes: {}", notes);
}

fn main() {
    let args = Args::parse();

    println!("=== Phase 6 Rust Benchmark ===");
    println!("Config: n_gpu={}, n_ctx={}, batch={}/{}, kv={}/{}, flash={}",
             args.n_gpu, args.n_ctx, args.n_batch, args.n_ubatch,
             ggml_type_name(args.type_k), ggml_type_name(args.type_v), args.flash_attn);

    // We can't safely set struct fields without knowing exact offsets.
    // Use a C shim instead. The shim is compiled by build.rs.
    // For now, use the shim approach below.
    
    match run_bench_via_shim(&args) {
        Ok((tok_s, vram_mb, notes)) => {
            append_result(&args, tok_s, vram_mb, "ok", &notes);
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            let status = if e.contains("OOM") || e.contains("oom") { "oom" } else { "crash" };
            append_result(&args, 0.0, 0, status, &e);
        }
    }
}

/// Run benchmark via C shim that handles struct field access
fn run_bench_via_shim(args: &Args) -> Result<(f64, u64, String), String> {
    unsafe {
        // Load backends from the venv lib directory
        let backend_dir = CString::new("/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/llama_cpp/lib").unwrap();
        bench_shim_load_backends(backend_dir.as_ptr());

        llama_backend_init();

        // Get default params and patch them via our C shim
        let mparams = bench_shim_model_params(args.n_gpu);

        let model_path = CString::new(args.model.as_str()).unwrap();
        let model = llama_model_load_from_file(model_path.as_ptr(), mparams);
        if model.is_null() {
            llama_backend_free();
            return Err("Failed to load model".to_string());
        }

        let cparams = bench_shim_context_params(
            args.n_ctx, args.n_batch, args.n_ubatch,
            args.n_threads, args.flash_attn,
            args.type_k, args.type_v,
        );

        let ctx = llama_init_from_model(model, cparams);
        if ctx.is_null() {
            llama_model_free(model);
            llama_backend_free();
            return Err("Failed to create context (OOM?)".to_string());
        }

        // Verify context settings
        let actual_ctx = llama_n_ctx(ctx);
        println!("Context created: n_ctx={}", actual_ctx);

        // Tokenize
        let vocab = llama_model_get_vocab(model);
        let prompt_c = CString::new(args.prompt.as_str()).unwrap();
        let max_tokens = 512i32;
        let mut tokens: Vec<llama_token> = vec![0; max_tokens as usize];

        let n_prompt = llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            args.prompt.len() as i32,
            tokens.as_mut_ptr(),
            max_tokens,
            true, true,
        );

        if n_prompt < 0 {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return Err(format!("Tokenization failed: {}", n_prompt));
        }
        tokens.truncate(n_prompt as usize);
        println!("Prompt tokens: {}", n_prompt);

        // Eval prompt
        let batch = llama_batch_get_one(tokens.as_mut_ptr(), n_prompt);
        let ret = llama_decode(ctx, batch);
        if ret != 0 {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return Err(format!("Prompt eval failed: {}", ret));
        }
        println!("Prompt evaluated");

        // Reset perf after prompt
        llama_perf_context_reset(ctx);

        // Create greedy sampler
        let sparams = llama_sampler_chain_default_params();
        let smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // Generate
        let eos = llama_vocab_eos(vocab);
        let gen_start = Instant::now();
        let mut n_generated = 0i32;

        for _ in 0..args.n_gen {
            let new_token = llama_sampler_sample(smpl, ctx, -1);
            if new_token == eos {
                break;
            }
            n_generated += 1;

            let mut next_tokens: [llama_token; 1] = [new_token];
            let next_batch = llama_batch_get_one(next_tokens.as_mut_ptr(), 1);
            let ret = llama_decode(ctx, next_batch);
            if ret != 0 {
                eprintln!("decode failed at token {}: {}", n_generated, ret);
                break;
            }
        }

        let gen_elapsed = gen_start.elapsed();
        let vram = get_vram_mb();

        // Get internal perf
        let perf = llama_perf_context(ctx);
        let internal_tok_s = if perf.t_eval_ms > 0.0 {
            (perf.n_eval as f64 / perf.t_eval_ms) * 1000.0
        } else {
            0.0
        };
        let wall_tok_s = if gen_elapsed.as_secs_f64() > 0.0 {
            n_generated as f64 / gen_elapsed.as_secs_f64()
        } else {
            0.0
        };

        llama_perf_context_print(ctx);

        let notes = format!(
            "n_gpu={}, batch={}/{}, {} KV, n_ctx={}, flash={} — wall={:.3}tok/s internal={:.3}tok/s, gen={}, t_eval={:.1}ms",
            args.n_gpu, args.n_batch, args.n_ubatch,
            ggml_type_name(args.type_k), args.n_ctx, args.flash_attn,
            wall_tok_s, internal_tok_s,
            n_generated, perf.t_eval_ms
        );

        // Cleanup
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        Ok((internal_tok_s, vram, notes))
    }
}

// C shim functions — defined in shim.c
extern "C" {
    fn bench_shim_load_backends(dir_path: *const libc::c_char);
    fn bench_shim_model_params(n_gpu_layers: i32) -> llama_model_params;
    fn bench_shim_context_params(
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
        n_threads: i32,
        flash_attn: i32,
        type_k: i32,
        type_v: i32,
    ) -> llama_context_params;
}
