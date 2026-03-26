"""
run_experiments.py — Phase 4 autoresearch continuation.
Explores: n_ctx scaling, q4_0 KV, more GPU layers, n_threads tuning.
Appends results to results_phase4.tsv starting from exp 15.
"""
import os, time, sys, subprocess, json, traceback
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
TSV_PATH = "./results_phase4.tsv"

# GGML type constants
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q4_0 = 2

TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0'}

MAX_TOKENS = 256
PROMPT_SHORT = "Explain the architecture of Mixture of Experts neural networks in detail:"

def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out.split('\n')[0])
    except Exception:
        return 0.0

def run_single(n_gpu_layers, n_ctx, n_threads, n_batch, n_ubatch, type_k, type_v, flash_attn, prompt=None):
    """Run a single benchmark. Returns dict with results or None on OOM/error."""
    from llama_cpp import Llama

    if prompt is None:
        prompt = PROMPT_SHORT

    model_path = Path(MODEL_PATH)
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    
    print(f"\n{'='*60}")
    print(f"Running: n_gpu={n_gpu_layers}, n_ctx={n_ctx}, n_threads={n_threads}, "
          f"n_batch={n_batch}, n_ubatch={n_ubatch}, KV={tk_name}/{tv_name}, flash={flash_attn}")
    print(f"{'='*60}", flush=True)

    vram_before = get_vram_mb()

    kwargs = dict(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        verbose=False,
    )
    if type_k is not None:
        kwargs["type_k"] = type_k
    if type_v is not None:
        kwargs["type_v"] = type_v
    if flash_attn:
        kwargs["flash_attn"] = True

    try:
        t_load = time.perf_counter()
        llm = Llama(**kwargs)
        load_time = time.perf_counter() - t_load
        vram_after_load = get_vram_mb()
        print(f"  Loaded in {load_time:.1f}s (VRAM: {vram_after_load:.0f} MB)", flush=True)

        # Warmup
        _ = llm(prompt, max_tokens=10, echo=False)

        # Benchmark
        print(f"  Generating {MAX_TOKENS} tokens...", flush=True)
        t0 = time.perf_counter()
        output = llm(prompt, max_tokens=MAX_TOKENS, echo=False)
        elapsed = time.perf_counter() - t0

        vram_peak = get_vram_mb()
        n_tokens = output["usage"]["completion_tokens"]
        tok_per_sec = n_tokens / elapsed

        # Quality check
        text = output["choices"][0]["text"][:200]
        print(f"  Result: {tok_per_sec:.3f} tok/s, {n_tokens} tokens, VRAM: {vram_peak:.0f} MB")
        print(f"  Preview: {text[:100]}...")

        del llm
        time.sleep(2)  # Let GPU memory settle

        return {
            "tok_per_sec": tok_per_sec,
            "vram_peak_mb": vram_peak,
            "n_tokens": n_tokens,
            "elapsed": elapsed,
            "load_time": load_time,
            "vram_after_load": vram_after_load,
        }

    except Exception as e:
        err_str = str(e)
        print(f"  ERROR: {err_str}", flush=True)
        # Try to clean up
        try:
            del llm
        except:
            pass
        time.sleep(3)
        
        if "out of memory" in err_str.lower() or "oom" in err_str.lower() or "cuda" in err_str.lower() or "alloc" in err_str.lower():
            return {"error": "OOM", "detail": err_str}
        return {"error": "FAIL", "detail": err_str}


def append_tsv(exp_num, tok_per_sec, vram_peak, n_ctx, type_k_name, type_v_name, flash_attn, n_gpu_layers, status, description, n_batch=32, n_ubatch=32, n_threads=10):
    """Append a result line to the TSV."""
    line = f"{exp_num}\t{tok_per_sec:.3f}\t{vram_peak:.0f}\t{n_ctx}\t{type_k_name}\t{type_v_name}\t{flash_attn}\t{n_gpu_layers}\t{status}\t{description}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"  >> TSV: exp {exp_num}: {tok_per_sec:.3f} tok/s [{status}]")


def main():
    best_tok = 10.200  # Phase 4 exp 14
    best_config = "exp 14: n_gpu=10, n_ctx=512, q8_0 KV"
    exp_num = 15

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Phase 4 Autoresearch — starting from exp {exp_num}")
    print(f"Current best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Model: {model_path} ({model_path.stat().st_size/1e9:.1f} GB)")
    print()

    # ===================================================================
    # AXIS 1: n_ctx scaling with q8_0 KV (baseline GPU layers = 10)
    # 512 already done. Try 1024, 2048, 4096, 8192.
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 1: Context length scaling (q8_0 KV, n_gpu=10)")
    print("="*70)

    ctx_results = {}
    for n_ctx in [1024, 2048, 4096, 8192]:
        result = run_single(
            n_gpu_layers=10, n_ctx=n_ctx, n_threads=10,
            n_batch=32, n_ubatch=32,
            type_k=GGML_Q8_0, type_v=GGML_Q8_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"n_ctx={n_ctx}, q8_0 KV, n_gpu=10"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: n_gpu=10, n_ctx={n_ctx}, q8_0 KV"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], n_ctx, "q8_0", "q8_0", True, 10, status, desc)
            ctx_results[n_ctx] = result["tok_per_sec"]
        elif result and result.get("error") == "OOM":
            append_tsv(exp_num, 0, 0, n_ctx, "q8_0", "q8_0", True, 10, "OOM", f"n_ctx={n_ctx} OOM — too large for 8GB")
            ctx_results[n_ctx] = "OOM"
            # No point trying larger
            for nc in [c for c in [1024, 2048, 4096, 8192] if c > n_ctx]:
                exp_num += 1
                append_tsv(exp_num, 0, 0, nc, "q8_0", "q8_0", True, 10, "skip", f"n_ctx={nc} skipped — {n_ctx} already OOM")
            break
        else:
            append_tsv(exp_num, 0, 0, n_ctx, "q8_0", "q8_0", True, 10, "error", f"n_ctx={n_ctx} error: {result.get('detail','unknown')[:50] if result else 'null'}")
        
        exp_num += 1

    # ===================================================================
    # AXIS 2: q4_0 KV at n_ctx=512 (baseline comparison)
    # Previous exp 4 showed q4_0 was slower at 5 GPU layers.
    # But now with 10 GPU layers, re-test.
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 2: q4_0 KV at various GPU layers")
    print("="*70)

    for n_gpu in [10, 12, 14]:
        result = run_single(
            n_gpu_layers=n_gpu, n_ctx=512, n_threads=10,
            n_batch=32, n_ubatch=32,
            type_k=GGML_Q4_0, type_v=GGML_Q4_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"q4_0 KV, n_gpu={n_gpu}, n_ctx=512"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: n_gpu={n_gpu}, n_ctx=512, q4_0 KV"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], 512, "q4_0", "q4_0", True, n_gpu, status, desc)
        elif result and result.get("error") == "OOM":
            append_tsv(exp_num, 0, 0, 512, "q4_0", "q4_0", True, n_gpu, "OOM", f"q4_0 KV, n_gpu={n_gpu} — OOM")
            break
        else:
            append_tsv(exp_num, 0, 0, 512, "q4_0", "q4_0", True, n_gpu, "error", f"q4_0 n_gpu={n_gpu} error")
        
        exp_num += 1

    # ===================================================================
    # AXIS 3: More GPU layers with q8_0 at n_ctx=512
    # Exp 13 showed 11 layers regressed. Try 12, 14, 16 anyway.
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 3: More GPU layers (q8_0 KV, n_ctx=512)")
    print("="*70)

    for n_gpu in [12, 14, 16]:
        result = run_single(
            n_gpu_layers=n_gpu, n_ctx=512, n_threads=10,
            n_batch=32, n_ubatch=32,
            type_k=GGML_Q8_0, type_v=GGML_Q8_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"n_gpu={n_gpu}, q8_0 KV, n_ctx=512"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: n_gpu={n_gpu}, n_ctx=512, q8_0 KV"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], 512, "q8_0", "q8_0", True, n_gpu, status, desc)
        elif result and result.get("error") == "OOM":
            append_tsv(exp_num, 0, 0, 512, "q8_0", "q8_0", True, n_gpu, "OOM", f"n_gpu={n_gpu} — OOM")
            # Skip higher
            break
        else:
            append_tsv(exp_num, 0, 0, 512, "q8_0", "q8_0", True, n_gpu, "error", f"n_gpu={n_gpu} error")
        
        exp_num += 1

    # ===================================================================
    # AXIS 4: n_threads tuning (q8_0 KV, n_gpu=10, n_ctx=512)
    # Always tested at 10 threads. Try 6, 8, 12, 14.
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 4: Thread count tuning")
    print("="*70)

    for n_threads in [6, 8, 12, 14]:
        result = run_single(
            n_gpu_layers=10, n_ctx=512, n_threads=n_threads,
            n_batch=32, n_ubatch=32,
            type_k=GGML_Q8_0, type_v=GGML_Q8_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"n_threads={n_threads}, n_gpu=10, q8_0 KV"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: n_gpu=10, n_ctx=512, q8_0 KV, n_threads={n_threads}"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], 512, "q8_0", "q8_0", True, 10, status, desc + f" (n_threads={n_threads})")
        else:
            append_tsv(exp_num, 0, 0, 512, "q8_0", "q8_0", True, 10, "error", f"n_threads={n_threads} error")
        
        exp_num += 1

    # ===================================================================
    # AXIS 5: q4_0 KV + larger context (can q4_0 unlock bigger n_ctx?)
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 5: q4_0 KV + context scaling (n_gpu=10)")
    print("="*70)

    for n_ctx in [1024, 2048, 4096, 8192]:
        result = run_single(
            n_gpu_layers=10, n_ctx=n_ctx, n_threads=10,
            n_batch=32, n_ubatch=32,
            type_k=GGML_Q4_0, type_v=GGML_Q4_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"q4_0 KV, n_ctx={n_ctx}, n_gpu=10"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: n_gpu=10, n_ctx={n_ctx}, q4_0 KV"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], n_ctx, "q4_0", "q4_0", True, 10, status, desc)
        elif result and result.get("error") == "OOM":
            append_tsv(exp_num, 0, 0, n_ctx, "q4_0", "q4_0", True, 10, "OOM", f"q4_0 n_ctx={n_ctx} — OOM")
            break
        else:
            append_tsv(exp_num, 0, 0, n_ctx, "q4_0", "q4_0", True, 10, "error", f"q4_0 n_ctx={n_ctx} error")
        
        exp_num += 1

    # ===================================================================
    # AXIS 6: Combined — q4_0 + more GPU layers + best n_ctx
    # Find max n_ctx that fits, then push GPU layers
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 6: Combined optimization — q4_0 + GPU layers + context")
    print("="*70)

    # Try q4_0 with 12 GPU layers at various context sizes
    for n_gpu in [12, 14]:
        for n_ctx in [512, 1024, 2048]:
            result = run_single(
                n_gpu_layers=n_gpu, n_ctx=n_ctx, n_threads=10,
                n_batch=32, n_ubatch=32,
                type_k=GGML_Q4_0, type_v=GGML_Q4_0, flash_attn=True
            )
            
            if result and "error" not in result:
                status = "keep" if result["tok_per_sec"] > best_tok else "discard"
                desc = f"q4_0 KV, n_gpu={n_gpu}, n_ctx={n_ctx}"
                if result["tok_per_sec"] > best_tok:
                    best_tok = result["tok_per_sec"]
                    best_config = f"exp {exp_num}: n_gpu={n_gpu}, n_ctx={n_ctx}, q4_0 KV"
                    desc += " — NEW BEST"
                append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], n_ctx, "q4_0", "q4_0", True, n_gpu, status, desc)
            elif result and result.get("error") == "OOM":
                append_tsv(exp_num, 0, 0, n_ctx, "q4_0", "q4_0", True, n_gpu, "OOM", f"q4_0 n_gpu={n_gpu} n_ctx={n_ctx} — OOM")
                break  # No point trying larger ctx at this GPU count
            else:
                append_tsv(exp_num, 0, 0, n_ctx, "q4_0", "q4_0", True, n_gpu, "error", f"combined error")
            
            exp_num += 1

    # ===================================================================
    # AXIS 7: q8_0 + more GPU layers at larger n_ctx
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 7: q8_0 + GPU layers + larger context")
    print("="*70)

    for n_gpu in [8, 10, 12]:
        for n_ctx in [1024, 2048]:
            # Skip combos we already tested in axis 1 (n_gpu=10)
            if n_gpu == 10 and n_ctx in ctx_results:
                exp_num += 1
                continue
                
            result = run_single(
                n_gpu_layers=n_gpu, n_ctx=n_ctx, n_threads=10,
                n_batch=32, n_ubatch=32,
                type_k=GGML_Q8_0, type_v=GGML_Q8_0, flash_attn=True
            )
            
            if result and "error" not in result:
                status = "keep" if result["tok_per_sec"] > best_tok else "discard"
                desc = f"q8_0 KV, n_gpu={n_gpu}, n_ctx={n_ctx}"
                if result["tok_per_sec"] > best_tok:
                    best_tok = result["tok_per_sec"]
                    best_config = f"exp {exp_num}: n_gpu={n_gpu}, n_ctx={n_ctx}, q8_0 KV"
                    desc += " — NEW BEST"
                append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], n_ctx, "q8_0", "q8_0", True, n_gpu, status, desc)
            elif result and result.get("error") == "OOM":
                append_tsv(exp_num, 0, 0, n_ctx, "q8_0", "q8_0", True, n_gpu, "OOM", f"q8_0 n_gpu={n_gpu} n_ctx={n_ctx} — OOM")
            else:
                append_tsv(exp_num, 0, 0, n_ctx, "q8_0", "q8_0", True, n_gpu, "error", f"q8_0 n_gpu={n_gpu} n_ctx={n_ctx} error")
            
            exp_num += 1

    # ===================================================================
    # AXIS 8: Batch size experiments at best config
    # ===================================================================
    print("\n" + "="*70)
    print("AXIS 8: Batch size tuning at best config")
    print("="*70)

    for n_batch, n_ubatch in [(64, 64), (128, 128), (16, 16), (64, 32), (128, 32)]:
        result = run_single(
            n_gpu_layers=10, n_ctx=512, n_threads=10,
            n_batch=n_batch, n_ubatch=n_ubatch,
            type_k=GGML_Q8_0, type_v=GGML_Q8_0, flash_attn=True
        )
        
        if result and "error" not in result:
            status = "keep" if result["tok_per_sec"] > best_tok else "discard"
            desc = f"n_batch={n_batch}, n_ubatch={n_ubatch}, n_gpu=10, q8_0 KV"
            if result["tok_per_sec"] > best_tok:
                best_tok = result["tok_per_sec"]
                best_config = f"exp {exp_num}: batch={n_batch}/{n_ubatch}, n_gpu=10, q8_0 KV"
                desc += " — NEW BEST"
            append_tsv(exp_num, result["tok_per_sec"], result["vram_peak_mb"], 512, "q8_0", "q8_0", True, 10, status, desc)
        else:
            append_tsv(exp_num, 0, 0, 512, "q8_0", "q8_0", True, 10, "error", f"batch {n_batch}/{n_ubatch} error")
        
        exp_num += 1

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 4 AUTORESEARCH COMPLETE")
    print(f"Best: {best_tok:.3f} tok/s — {best_config}")
    print("="*70)
    
    return best_tok, best_config, exp_num


if __name__ == "__main__":
    best_tok, best_config, final_exp = main()
    print(f"\nFinal best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Total experiments: {final_exp - 1}")
