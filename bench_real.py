"""
bench_real.py — Phase 3: Real inference on actual Qwen3.5-35B-A3B-Q3_K_M GGUF.

No more simulations. Real llama.cpp inference on RTX 3070 8GB.
Measures actual tok/s under different GPU layer offload configurations.

Usage: uv run --with llama-cpp-python python bench_real.py
"""
import os, time, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# EXPERIMENT KNOBS — modify these
# ---------------------------------------------------------------------------

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"

# GPU layer offload: how many transformer layers to keep in VRAM
# RTX 3070 8GB: ~2.6GB free after always-hot layers
# Qwen3.5-35B-A3B has 64 layers total
# 0 = CPU only (slowest, no VRAM used)
# 10 = partial offload (some layers in VRAM)
# 20 = more offload
# 64 = full offload (requires >8GB — will OOM, use as test)
N_GPU_LAYERS = 5

# Context window
N_CTX = 512

# Threads for CPU inference (experts not in VRAM)
N_THREADS = 10

# Batch size
N_BATCH = 64

# Test prompt
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

# Number of tokens to generate
MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Run inference and measure
# ---------------------------------------------------------------------------

def run():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        # Check for incomplete download
        incomplete = list(Path("./models/.cache/huggingface/download").glob("*.incomplete"))
        if incomplete:
            size_gb = incomplete[0].stat().st_size / 1e9
            total_gb = 16.4
            pct = size_gb / total_gb * 100
            print(f"Model still downloading: {size_gb:.1f}/{total_gb:.1f} GB ({pct:.0f}%)")
        else:
            print(f"Model not found at {MODEL_PATH}")
            print("Run: uv run --with huggingface_hub python /tmp/download_qwen.py")
        sys.exit(1)

    print(f"Model: {model_path} ({model_path.stat().st_size/1e9:.1f} GB)")
    print(f"Config: n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}, n_threads={N_THREADS}")
    print("Loading model...", flush=True)

    from llama_cpp import Llama

    t_load = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        verbose=False,
    )
    load_time = time.perf_counter() - t_load
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    # Warmup
    print("Warming up...", flush=True)
    _ = llm(PROMPT, max_tokens=10, echo=False)

    # Benchmark
    print(f"Generating {MAX_TOKENS} tokens...", flush=True)
    t0 = time.perf_counter()
    output = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
    elapsed = time.perf_counter() - t0

    n_tokens = output["usage"]["completion_tokens"]
    tok_per_sec = n_tokens / elapsed

    print(f"\n--- RESULTS ---")
    print(f"tok_per_sec: {tok_per_sec:.3f}")
    print(f"tokens_generated: {n_tokens}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"n_gpu_layers: {N_GPU_LAYERS}")
    print(f"model: Q3_K_M")
    print(f"---------------")

    # Print generated text preview
    text = output["choices"][0]["text"][:200]
    print(f"\nGenerated: {text}...")


if __name__ == "__main__":
    run()
