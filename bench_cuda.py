"""
bench_cuda.py — CUDA-enabled inference! Real GPU acceleration.
Now that libggml-cuda.so loads, sweep GPU layers properly.
"""
import os, time, sys
from pathlib import Path

# Must set before import to ensure CUDA libs are found
CUBLAS12 = "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
CUDART12 = "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CUBLAS12}:{CUDART12}:{ld}"

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

def benchmark(label, n_runs=2, **kwargs):
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

results = []

# Sweep GPU layers with CUDA
# RTX 3070 8GB — ~6.4GB free (after openclaw 982MB + system ~800MB = ~6.4GB)
# Q3_K_M: each layer ~290MB, so we can fit ~22 layers before OOM
print("=== CUDA GPU SWEEP ===")
for nlayers in [0, 5, 10, 15, 20, 25, 30, 33]:
    tps = benchmark(f"cuda_L{nlayers}", 
                   n_gpu_layers=nlayers, n_ctx=512, n_threads=10, 
                   n_batch=64, n_ubatch=32, offload_kqv=True)
    if tps: 
        results.append((tps, f"cuda_L{nlayers}"))

print("\n=== RANKING SO FAR ===")
results.sort(reverse=True)
for tps, label in results[:5]:
    print(f"{tps:.3f} | {label}")

# If we found a sweet spot, refine
if results:
    best_tps, best_label = results[0]
    best_layers = int(best_label.split('L')[1])
    print(f"\nBest: {best_layers} layers at {best_tps:.3f} tok/s")
    
    # Fine-tune around best
    print("\n=== FINE-TUNING ===")
    for nlayers in [best_layers - 2, best_layers - 1, best_layers + 1, best_layers + 2]:
        if nlayers < 0 or nlayers > 40: continue
        if f"cuda_L{nlayers}" in [r[1] for r in results]: continue
        tps = benchmark(f"cuda_L{nlayers}_fine", 
                       n_gpu_layers=nlayers, n_ctx=512, n_threads=10, 
                       n_batch=64, n_ubatch=32, offload_kqv=True)
        if tps: results.append((tps, f"cuda_L{nlayers}_fine"))
    
    # Also try: best_layers + flash_attn
    tps = benchmark(f"cuda_L{best_layers}_flash", 
                   n_gpu_layers=best_layers, n_ctx=512, n_threads=10, 
                   n_batch=64, n_ubatch=32, offload_kqv=True, flash_attn=True)
    if tps: results.append((tps, f"cuda_L{best_layers}_flash"))
    
    # And: best_layers + fewer threads (GPU does more work)
    for nt in [6, 8]:
        tps = benchmark(f"cuda_L{best_layers}_t{nt}", 
                       n_gpu_layers=best_layers, n_ctx=512, n_threads=nt, 
                       n_threads_batch=nt, n_batch=64, n_ubatch=32, offload_kqv=True)
        if tps: results.append((tps, f"cuda_L{best_layers}_t{nt}"))

print("\n=== FINAL RANKING ===")
results.sort(reverse=True)
for tps, label in results:
    marker = " *** BEST ***" if tps == results[0][0] else ""
    print(f"{tps:.3f} | {label}{marker}")
