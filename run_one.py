"""
run_one.py — Run a single experiment via command-line args.
Usage: uv run --with llama-cpp-python python run_one.py <exp_num> <ngl> <ctx> <batch> <ubatch> <type_k> <type_v> <desc>
Outputs JSON to stdout on success.
"""
import json, time, sys, subprocess, gc, os
os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/lib/ollama/cuda_v12')

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"
MAX_TOKENS = 256

T = {'f16': 1, 'q8_0': 8, 'q4_0': 2, 'q4_1': 3, 'q5_0': 6, 'q5_1': 7, 'iq4_nl': 20, 'q4_k': 12, 'q5_k': 13}
TNAME = {v: k for k, v in T.items()}

def vram_mb():
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip().split('\n')[0])
    except: return 0.0

def main():
    args = sys.argv[1:]
    if len(args) < 8:
        print("Usage: run_one.py <exp> <ngl> <ctx> <batch> <ubatch> <type_k> <type_v> <desc>", file=sys.stderr)
        sys.exit(1)
    
    exp_num = int(args[0])
    ngl, ctx, batch, ubatch = int(args[1]), int(args[2]), int(args[3]), int(args[4])
    type_k_name, type_v_name = args[5], args[6]
    desc = ' '.join(args[7:])
    
    type_k = T[type_k_name]
    type_v = T[type_v_name]
    
    from llama_cpp import Llama
    
    v0 = vram_mb()
    kw = dict(model_path=MODEL_PATH, n_gpu_layers=ngl, n_ctx=ctx,
              n_threads=10, n_batch=batch, n_ubatch=ubatch,
              verbose=False, type_k=type_k, type_v=type_v, flash_attn=True)
    
    t0 = time.perf_counter()
    llm = Llama(**kw)
    lt = time.perf_counter() - t0
    v1 = vram_mb()
    
    # warmup
    _ = llm(PROMPT, max_tokens=10, echo=False)
    
    # bench 2 trials
    best = 0
    for _ in range(2):
        t = time.perf_counter()
        out = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
        e = time.perf_counter() - t
        nt = out["usage"]["completion_tokens"]
        tps = nt / e
        if tps > best:
            best, be, bn = tps, e, nt
    
    vp = vram_mb()
    text_preview = out["choices"][0]["text"][:100]
    
    r = {'exp': exp_num, 'tps': round(best, 3), 'tokens': bn, 'elapsed': round(be, 2),
         'vram_peak': vp, 'vram_model': round(vp - v0), 'load_s': round(lt, 1),
         'ngl': ngl, 'ctx': ctx, 'batch': batch, 'ubatch': ubatch,
         'type_k': type_k_name, 'type_v': type_v_name, 'desc': desc, 'status': 'ok',
         'preview': text_preview}
    
    print(json.dumps(r))

if __name__ == "__main__":
    main()
