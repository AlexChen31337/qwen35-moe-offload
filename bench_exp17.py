"""
bench_exp17.py — Experiments 17+: mlock, numa, cpu affinity, repeat best config
Current best: 6.587 tok/s (exp15: n_gpu_layers=5, n_threads=10, n_batch=64, n_ubatch=32)
"""
import os, time, sys
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"

def benchmark(label, **kwargs):
    from llama_cpp import Llama
    model_path = Path(MODEL_PATH)
    
    try:
        llm = Llama(
            model_path=str(model_path),
            verbose=False,
            **kwargs
        )
    except Exception as e:
        print(f"{label}: LOAD ERROR: {e}")
        return None
    
    # warmup
    _ = llm(PROMPT, max_tokens=10, echo=False)
    
    # benchmark x3 for stability
    tps_list = []
    for _ in range(2):
        t0 = time.perf_counter()
        output = llm(PROMPT, max_tokens=200, echo=False)
        elapsed = time.perf_counter() - t0
        n_tokens = output["usage"]["completion_tokens"]
        tps_list.append(n_tokens / elapsed)
    
    tps = max(tps_list)  # take best of 2
    print(f"{label}: {tps:.3f} tok/s (best of 2 runs)")
    sys.stdout.flush()
    del llm
    return tps

# Base config (current best)
BASE = dict(
    n_gpu_layers=5,
    n_ctx=512,
    n_threads=10,
    n_threads_batch=10,
    n_batch=64,
    n_ubatch=32,
)

results = []

# Exp 17: mlock to pin model in RAM (prevent swapping)
tps = benchmark("exp17_mlock_true", use_mlock=True, **BASE)
if tps: results.append((tps, "exp17_mlock_true"))

# Exp 18: disable mmap (force all data in RAM, no page faults)
tps = benchmark("exp18_no_mmap", use_mmap=False, **BASE)
if tps: results.append((tps, "exp18_no_mmap"))

# Exp 19: mlock + no mmap combined
tps = benchmark("exp19_mlock_no_mmap", use_mlock=True, use_mmap=False, **BASE)
if tps: results.append((tps, "exp19_mlock_no_mmap"))

# Exp 20: KV cache in fp16 type
# type_k=1 = fp16, type_v=1 = fp16 (default is usually f16 already but explicit)
# type_k=8 = q8_0 (quantized KV cache - saves VRAM for more GPU layers)
try:
    tps = benchmark("exp20_kv_q8", type_k=8, type_v=8, **BASE)
    if tps: results.append((tps, "exp20_kv_q8"))
except Exception as e:
    print(f"exp20_kv_q8: SKIP ({e})")

# Exp 21: KV cache q4 (aggressive)
try:
    tps = benchmark("exp21_kv_q4", type_k=12, type_v=12, **BASE)
    if tps: results.append((tps, "exp21_kv_q4"))
except Exception as e:
    print(f"exp21_kv_q4: SKIP ({e})")

# Exp 22: offload KV to GPU too
tps = benchmark("exp22_offload_kqv", offload_kqv=True, **BASE)
if tps: results.append((tps, "exp22_offload_kqv"))

print("\n=== RANKING ===")
results.sort(reverse=True)
for tps, label in results:
    print(f"{tps:.3f} | {label}")
