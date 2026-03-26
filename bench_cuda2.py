"""
bench_cuda2.py — Deep exploration around new best (12.569 tok/s at L15, t=8)
Target: push past 13 tok/s
"""
import os, time, sys
from pathlib import Path

CUBLAS12 = "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
CUDART12 = "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CUBLAS12}:{CUDART12}:{ld}"

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
        out = llm(PROMPT, max_tokens=200, echo=False)
        tps_list.append(out["usage"]["completion_tokens"] / (time.perf_counter() - t0))
    
    tps = max(tps_list)
    print(f"{label}: {tps:.3f} tok/s")
    sys.stdout.flush()
    del llm
    return tps

results = []

# Best so far: L15, t=8, offload_kqv=True
BASE_BEST = dict(n_gpu_layers=15, n_threads=8, n_threads_batch=8, 
                 n_batch=64, n_ubatch=32, offload_kqv=True, n_ctx=512)
BASE_L16 = dict(n_gpu_layers=16, n_threads=10, n_threads_batch=10, 
                n_batch=64, n_ubatch=32, offload_kqv=True, n_ctx=512)

# 1. Thread sweep around t=8 for L15 and L16
print("=== THREAD SWEEP L15 & L16 ===")
for nlayers in [15, 16]:
    for nt in [6, 7, 8, 9, 10]:
        tps = benchmark(f"L{nlayers}_t{nt}", n_gpu_layers=nlayers, n_threads=nt, 
                       n_threads_batch=nt, n_batch=64, n_ubatch=32, 
                       offload_kqv=True, n_ctx=512)
        if tps: results.append((tps, f"L{nlayers}_t{nt}"))

# 2. Batch/ubatch sweep with best layer config
print("\n=== BATCH SWEEP ===")
for n_batch, n_ubatch in [(32, 32), (64, 16), (64, 32), (64, 64), (128, 32), (128, 64)]:
    for nlayers in [15, 16]:
        key = f"L{nlayers}_b{n_batch}_ub{n_ubatch}"
        tps = benchmark(key, n_gpu_layers=nlayers, n_threads=8, n_threads_batch=8,
                       n_batch=n_batch, n_ubatch=n_ubatch, offload_kqv=True, n_ctx=512)
        if tps: results.append((tps, key))

# 3. Flash attn + different threads 
print("\n=== FLASH ATTN COMBOS ===")
for nt in [6, 7, 8]:
    tps = benchmark(f"L15_flash_t{nt}", n_gpu_layers=15, n_threads=nt, 
                   n_batch=64, n_ubatch=32, offload_kqv=True, flash_attn=True, n_ctx=512)
    if tps: results.append((tps, f"L15_flash_t{nt}"))

# 4. Try smaller context to see if it helps
print("\n=== CTX SWEEP ===")
for ctx in [128, 256, 512]:
    for nlayers in [15, 16]:
        tps = benchmark(f"L{nlayers}_ctx{ctx}", n_gpu_layers=nlayers, n_threads=8,
                       n_batch=64, n_ubatch=32, offload_kqv=True, n_ctx=ctx)
        if tps: results.append((tps, f"L{nlayers}_ctx{ctx}"))

print("\n=== FINAL RANKING ===")
results.sort(reverse=True)
for tps, label in results[:10]:
    marker = " *** BEST ***" if tps == results[0][0] else ""
    print(f"{tps:.3f} | {label}{marker}")
