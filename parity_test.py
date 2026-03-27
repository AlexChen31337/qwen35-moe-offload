"""Quick parity test: Python llama-cpp-python vs Rust bench."""
import time, subprocess, json
from pathlib import Path
from llama_cpp import Llama

MODEL = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "The meaning of life is"
N_GEN = 128

print("=== Python parity test ===")
print("Config: n_gpu=16, n_ctx=512, batch=128/64, q8_0, flash=True, n_threads=12")

vram_before = 0
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    vram_before = float(out.split('\n')[0])
except:
    pass

llm = Llama(
    model_path=MODEL,
    n_gpu_layers=16,
    n_ctx=512,
    n_threads=12,
    n_batch=128,
    n_ubatch=64,
    type_k=8, type_v=8,
    flash_attn=True,
    verbose=False,
)

# Warmup
_ = llm(PROMPT, max_tokens=10, echo=False)

# Bench
t0 = time.perf_counter()
output = llm(PROMPT, max_tokens=N_GEN, echo=False)
elapsed = time.perf_counter() - t0

n_tokens = output["usage"]["completion_tokens"]
tok_s = n_tokens / elapsed

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    vram = float(out.split('\n')[0])
except:
    vram = 0

print(f"Python: {tok_s:.3f} tok/s, {n_tokens} tokens in {elapsed:.2f}s, VRAM={vram:.0f}MB")
