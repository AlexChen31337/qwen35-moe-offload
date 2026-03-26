"""
bench_kv.py — Phase 4: KV cache compression experiments.

Tests different KV cache quantization types and context lengths
on Qwen3.5-35B-A3B-Q3_K_M via llama-cpp-python.

Usage: uv run --with llama-cpp-python python bench_kv.py
"""
import os, time, sys, subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# EXPERIMENT KNOBS — modify these per experiment
# ---------------------------------------------------------------------------

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"

# Phase 3 best config
N_GPU_LAYERS = 10
N_CTX = 512
N_THREADS = 10
N_BATCH = 32
N_UBATCH = 32

# KV cache quantization types (GGML integer constants)
# None = default (f16=1), q8_0=8, q4_0=2, q4_1=3, q5_0=6, q5_1=7, iq4_nl=20, q4_k=12, q5_k=13
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_IQ4_NL = 20
GGML_Q4_K = 12
GGML_Q5_K = 13

TYPE_K = GGML_Q8_0  # Best KV quant
TYPE_V = GGML_Q8_0

# Flash attention (REQUIRED for V cache quantization in llama.cpp)
FLASH_ATTN = True

# Op offload — offload host tensor operations to GPU
OP_OFFLOAD = False

# Sliding window attention — use full KV for SWA layers
SWA_FULL = False

# Number of tokens to generate
MAX_TOKENS = 256

# Test prompt — long version to stress KV cache at high n_ctx
PROMPT_SHORT = "Explain the architecture of Mixture of Experts neural networks in detail:"
PROMPT_LONG = """The following is a comprehensive analysis of modern neural network architectures. """ * 200  # ~1000 tokens
PROMPT = PROMPT_SHORT  # Change to PROMPT_LONG for KV stress test

# ---------------------------------------------------------------------------
# Measure VRAM
# ---------------------------------------------------------------------------
def get_vram_mb():
    """Get current GPU VRAM usage in MB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out.split('\n')[0])
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Run inference and measure
# ---------------------------------------------------------------------------
def run():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Model: {model_path} ({model_path.stat().st_size/1e9:.1f} GB)")
    print(f"Config: n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}, n_threads={N_THREADS}")
    type_names = {None: 'f16', 1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 20: 'iq4_nl', 12: 'q4_k', 13: 'q5_k'}
    tk_name = type_names.get(TYPE_K, str(TYPE_K))
    tv_name = type_names.get(TYPE_V, str(TYPE_V))
    print(f"KV cache: type_k={tk_name}, type_v={tv_name}, flash_attn={FLASH_ATTN}")
    print("Loading model...", flush=True)

    from llama_cpp import Llama

    vram_before = get_vram_mb()

    # Build kwargs
    kwargs = dict(
        model_path=str(model_path),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_ubatch=N_UBATCH,
        verbose=False,
    )
    if TYPE_K is not None:
        kwargs["type_k"] = TYPE_K
    if TYPE_V is not None:
        kwargs["type_v"] = TYPE_V
    if FLASH_ATTN:
        kwargs["flash_attn"] = True
    if OP_OFFLOAD:
        kwargs["op_offload"] = True
    if SWA_FULL:
        kwargs["swa_full"] = True

    t_load = time.perf_counter()
    llm = Llama(**kwargs)
    load_time = time.perf_counter() - t_load

    vram_after_load = get_vram_mb()
    print(f"Model loaded in {load_time:.1f}s (VRAM: {vram_after_load:.0f} MB)", flush=True)

    # Warmup
    print("Warming up...", flush=True)
    _ = llm(PROMPT, max_tokens=10, echo=False)
    vram_warmup = get_vram_mb()

    # Benchmark
    print(f"Generating {MAX_TOKENS} tokens...", flush=True)
    t0 = time.perf_counter()
    output = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
    elapsed = time.perf_counter() - t0

    vram_peak = get_vram_mb()

    n_tokens = output["usage"]["completion_tokens"]
    tok_per_sec = n_tokens / elapsed

    print(f"\n--- RESULTS ---")
    print(f"tok_per_sec: {tok_per_sec:.3f}")
    print(f"tokens_generated: {n_tokens}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"n_gpu_layers: {N_GPU_LAYERS}")
    print(f"n_ctx: {N_CTX}")
    print(f"type_k: {tk_name}")
    print(f"type_v: {tv_name}")
    print(f"flash_attn: {FLASH_ATTN}")
    print(f"vram_before_mb: {vram_before:.0f}")
    print(f"vram_after_load_mb: {vram_after_load:.0f}")
    print(f"vram_peak_mb: {vram_peak:.0f}")
    print(f"vram_model_mb: {vram_peak - vram_before:.0f}")
    print(f"model: Q3_K_M")
    print(f"---------------")

    # Print generated text preview (quality check)
    text = output["choices"][0]["text"][:300]
    print(f"\nGenerated: {text}...")


if __name__ == "__main__":
    run()
