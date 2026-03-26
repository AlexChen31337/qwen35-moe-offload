"""
bench_exp23.py — Exploit offload_kqv=True win (7.040 tok/s) 
Now explore: more GPU layers with KQV offload, flash_attn combo, layer sweep
"""
import os, time, sys
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

def benchmark(label, n_runs=3, **kwargs):
    from llama_cpp import Llama
    model_path = Path(MODEL_PATH)
    
    try:
        llm = Llama(model_path=str(model_path), verbose=False, **kwargs)
    except Exception as e:
        print(f"{label}: LOAD ERROR: {e}")
        return None
    
    _ = llm(PROMPT, max_tokens=10, echo=False)  # warmup
    
    tps_list = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        output = llm(PROMPT, max_tokens=200, echo=False)
        elapsed = time.perf_counter() - t0
        tps_list.append(output["usage"]["completion_tokens"] / elapsed)
    
    tps = max(tps_list)
    print(f"{label}: {tps:.3f} tok/s")
    sys.stdout.flush()
    del llm
    return tps

# KQV offload baseline (our new champion)
BASE_KQV = dict(n_ctx=512, n_threads=10, n_batch=64, n_ubatch=32, offload_kqv=True)

results = []

# Exp 23: KQV offload with different n_gpu_layers
for nlayers in [3, 5, 7, 10, 12, 15]:
    tps = benchmark(f"exp23_kqv_layers{nlayers}", n_gpu_layers=nlayers, **BASE_KQV)
    if tps: results.append((tps, f"kqv_layers{nlayers}"))

# Exp 24: KQV + flash_attn
tps = benchmark("exp24_kqv_flash", n_gpu_layers=5, flash_attn=True, **BASE_KQV)
if tps: results.append((tps, "kqv_flash"))

# Exp 25: KQV + n_threads sweep
for nt in [8, 9, 10, 11, 12]:
    tps = benchmark(f"exp25_kqv_t{nt}", n_gpu_layers=5, n_threads=nt, n_threads_batch=nt,
                   n_ctx=512, n_batch=64, n_ubatch=32, offload_kqv=True)
    if tps: results.append((tps, f"kqv_t{nt}"))

# Exp 26: KQV + n_ctx variations (smaller ctx = less KV = faster)
for ctx in [128, 256, 512]:
    if ctx == 512: continue  # already have baseline
    tps = benchmark(f"exp26_kqv_ctx{ctx}", n_gpu_layers=5, n_ctx=ctx, **BASE_KQV)
    if tps: results.append((tps, f"kqv_ctx{ctx}"))

print("\n=== RANKING ===")
results.sort(reverse=True)
for tps, label in results:
    marker = " *** BEST ***" if tps == results[0][0] else ""
    print(f"{tps:.3f} | {label}{marker}")
