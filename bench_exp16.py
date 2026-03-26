"""
bench_exp16.py — Experiment 16: n_threads_batch tuning
Separate batch thread count from generation thread count.
Current best: 6.587 tok/s (exp15: n_ubatch=32)
"""
import os, time, sys, subprocess, json
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

experiments = [
    # (n_threads, n_threads_batch, n_ubatch, label)
    (10, 10, 32, "baseline_exp15"),   # reproduce best
    (10, 16, 32, "batch_threads_16"),  # more threads for batch
    (10, 8, 32, "batch_threads_8"),   # fewer batch threads
    (10, 12, 32, "batch_threads_12"),
    (10, 10, 16, "ubatch_16"),        # smaller ubatch
    (10, 10, 64, "ubatch_64"),        # bigger ubatch
    (10, 10, 8, "ubatch_8"),          # tiny ubatch
    (10, 16, 16, "batch16_ubatch16"), # combined
]

def benchmark(n_threads, n_threads_batch, n_ubatch, label):
    from llama_cpp import Llama
    model_path = Path(MODEL_PATH)
    
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=5,
        n_ctx=512,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        n_batch=64,
        n_ubatch=n_ubatch,
        verbose=False,
    )
    
    # warmup
    _ = llm(PROMPT, max_tokens=10, echo=False)
    
    # benchmark
    t0 = time.perf_counter()
    output = llm(PROMPT, max_tokens=200, echo=False)
    elapsed = time.perf_counter() - t0
    
    n_tokens = output["usage"]["completion_tokens"]
    tps = n_tokens / elapsed
    return tps, n_tokens

print("Running batch_thread experiments...")
results = []
for n_t, n_tb, n_ub, label in experiments:
    try:
        tps, n_tok = benchmark(n_t, n_tb, n_ub, label)
        results.append((tps, n_t, n_tb, n_ub, label))
        print(f"{label}: {tps:.3f} tok/s (threads={n_t}, batch_threads={n_tb}, ubatch={n_ub})")
        sys.stdout.flush()
    except Exception as e:
        print(f"{label}: ERROR {e}")

print("\n=== RANKING ===")
results.sort(reverse=True)
for tps, n_t, n_tb, n_ub, label in results:
    marker = " *** BEST ***" if tps == results[0][0] else ""
    print(f"{tps:.3f} | threads={n_t} batch_threads={n_tb} ubatch={n_ub} | {label}{marker}")
