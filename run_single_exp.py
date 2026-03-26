"""
run_single_exp.py — Run a single benchmark experiment in isolated process.
Prints structured JSON result to stdout.

Usage: uv run python run_single_exp.py --n_gpu 16 --n_ctx 512 --n_threads 10 \
       --n_batch 32 --n_ubatch 32 --type_k 8 --type_v 8 --flash_attn
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
MAX_TOKENS = 256
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1'}

def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out.split('\n')[0])
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, required=True)
    parser.add_argument("--n_ctx", type=int, required=True)
    parser.add_argument("--n_threads", type=int, default=10)
    parser.add_argument("--n_batch", type=int, default=32)
    parser.add_argument("--n_ubatch", type=int, default=32)
    parser.add_argument("--type_k", type=int, default=8)
    parser.add_argument("--type_v", type=int, default=8)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_true")
    args = parser.parse_args()
    
    flash = not args.no_flash_attn

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(json.dumps({"error": "MODEL_NOT_FOUND"}))
        sys.exit(1)

    from llama_cpp import Llama

    vram_before = get_vram_mb()

    kwargs = dict(
        model_path=str(model_path),
        n_gpu_layers=args.n_gpu,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        verbose=False,
    )
    kwargs["type_k"] = args.type_k
    kwargs["type_v"] = args.type_v
    if flash:
        kwargs["flash_attn"] = True

    try:
        t_load = time.perf_counter()
        llm = Llama(**kwargs)
        load_time = time.perf_counter() - t_load
        vram_after_load = get_vram_mb()

        # Warmup
        _ = llm(PROMPT, max_tokens=10, echo=False)

        # Benchmark
        t0 = time.perf_counter()
        output = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
        elapsed = time.perf_counter() - t0

        vram_peak = get_vram_mb()
        n_tokens = output["usage"]["completion_tokens"]
        tok_per_sec = n_tokens / elapsed
        text_preview = output["choices"][0]["text"][:200]

        result = {
            "tok_per_sec": round(tok_per_sec, 3),
            "vram_peak_mb": round(vram_peak),
            "n_tokens": n_tokens,
            "elapsed": round(elapsed, 2),
            "load_time": round(load_time, 1),
            "vram_after_load": round(vram_after_load),
            "text_preview": text_preview[:100],
        }
        print(json.dumps(result))

    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ["out of memory", "oom", "cuda", "alloc", "failed to load"]):
            print(json.dumps({"error": "OOM", "detail": str(e)[:200]}))
        else:
            print(json.dumps({"error": "FAIL", "detail": str(e)[:200]}))
        sys.exit(1)

if __name__ == "__main__":
    main()
