"""
orchestrate_phase4b.py — Phase 4 round 2: Combining winners.
Continues from exp 60.
"""
import json, os, subprocess, sys, time
from pathlib import Path

TSV_PATH = "./results_phase4.tsv"
TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1'}
Q8_0 = 8
Q4_0 = 2

def run_exp(n_gpu, n_ctx, n_threads=10, n_batch=32, n_ubatch=32, type_k=Q8_0, type_v=Q8_0, flash=True, timeout=300):
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    print(f"\n  >> n_gpu={n_gpu}, n_ctx={n_ctx}, n_threads={n_threads}, "
          f"batch={n_batch}/{n_ubatch}, KV={tk_name}/{tv_name}, flash={flash}", flush=True)
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")
    
    cmd = [
        sys.executable, "run_single_exp.py",
        "--n_gpu", str(n_gpu), "--n_ctx", str(n_ctx),
        "--n_threads", str(n_threads), "--n_batch", str(n_batch),
        "--n_ubatch", str(n_ubatch), "--type_k", str(type_k), "--type_v", str(type_v),
    ]
    if not flash:
        cmd.append("--no_flash_attn")
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                            cwd="/tmp/qwen35-moe-offload", env=env)
        for line in proc.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {"error": "NO_JSON", "detail": f"stdout={proc.stdout[:200]}"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "detail": f">{timeout}s"}
    except Exception as e:
        return {"error": "LAUNCH_FAIL", "detail": str(e)[:200]}


def append_tsv(exp, tok, vram, n_ctx, tk, tv, flash, n_gpu, n_batch, n_ubatch, n_threads, status, desc):
    line = f"{exp}\t{tok:.3f}\t{vram:.0f}\t{n_ctx}\t{tk}\t{tv}\t{flash}\t{n_gpu}\t{n_batch}\t{n_ubatch}\t{n_threads}\t{status}\t{desc}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"     TSV exp {exp}: {tok:.3f} tok/s [{status}] {desc}", flush=True)


def main():
    best_tok = 11.850
    best_config = "exp 46: n_gpu=16, batch=64/64, q8_0 KV, n_ctx=512"
    exp = 60

    print("="*70)
    print("Phase 4b — Combining Winners")
    print(f"Best: {best_tok:.3f} tok/s ({best_config})")
    print("="*70)

    # ===================================================================
    # BLOCK A: batch=64/64 + context scaling (q8_0, n_gpu=16)
    # This is the winning batch size, let's test at larger contexts
    # ===================================================================
    print("\n[BLOCK A] batch=64/64 + context scaling (q8_0 KV, n_gpu=16)")
    for n_ctx in [1024, 2048, 4096, 8192, 16384, 32768]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else ("usable" if tok > 8.0 else "discard")
            desc = f"n_gpu=16, batch=64/64, q8_0 KV, n_ctx={n_ctx}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, 10, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, 10, r["error"], f"batch64 n_ctx={n_ctx} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # BLOCK B: q4_0 KV + batch=64/64 (combining q4_0 wins with batch wins)
    # ===================================================================
    print("\n[BLOCK B] q4_0 KV + batch=64/64 at various configs")
    for n_gpu in [14, 16]:
        for n_ctx in [512, 1024, 2048, 4096, 8192]:
            r = run_exp(n_gpu=n_gpu, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q4_0, type_v=Q4_0)
            if "error" not in r:
                tok = r["tok_per_sec"]
                vram = r["vram_peak_mb"]
                status = "keep" if tok > best_tok else "discard"
                desc = f"q4_0 KV, n_gpu={n_gpu}, batch=64/64, n_ctx={n_ctx}"
                if tok > best_tok:
                    best_tok = tok
                    best_config = f"exp {exp}: {desc}"
                    desc += " — NEW BEST"
                    print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
                append_tsv(exp, tok, vram, n_ctx, "q4_0", "q4_0", True, n_gpu, 64, 64, 10, status, desc)
            else:
                append_tsv(exp, 0, 0, n_ctx, "q4_0", "q4_0", True, n_gpu, 64, 64, 10, r["error"], f"q4_0 batch64 n_gpu={n_gpu} n_ctx={n_ctx} — {r['error']}")
                if r["error"] == "OOM":
                    break
            exp += 1

    # ===================================================================
    # BLOCK C: n_gpu=15 (sweet spot between 14 and 16?)
    # ===================================================================
    print("\n[BLOCK C] n_gpu=15 exploration")
    for kv_type, kv_name in [(Q8_0, "q8_0"), (Q4_0, "q4_0")]:
        for n_batch, n_ubatch in [(32, 32), (64, 64)]:
            r = run_exp(n_gpu=15, n_ctx=512, n_batch=n_batch, n_ubatch=n_ubatch, type_k=kv_type, type_v=kv_type)
            if "error" not in r:
                tok = r["tok_per_sec"]
                vram = r["vram_peak_mb"]
                status = "keep" if tok > best_tok else "discard"
                desc = f"n_gpu=15, {kv_name} KV, batch={n_batch}/{n_ubatch}"
                if tok > best_tok:
                    best_tok = tok
                    best_config = f"exp {exp}: {desc}"
                    desc += " — NEW BEST"
                    print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
                append_tsv(exp, tok, vram, 512, kv_name, kv_name, True, 15, n_batch, n_ubatch, 10, status, desc)
            else:
                append_tsv(exp, 0, 0, 512, kv_name, kv_name, True, 15, n_batch, n_ubatch, 10, r["error"], f"n_gpu=15 {kv_name} — {r['error']}")
            exp += 1

    # ===================================================================
    # BLOCK D: Thread tuning at n_gpu=16, batch=64/64
    # ===================================================================
    print("\n[BLOCK D] Thread tuning (n_gpu=16, batch=64/64, q8_0, n_ctx=512)")
    for n_threads in [6, 8, 12, 14, 16]:
        r = run_exp(n_gpu=16, n_ctx=512, n_threads=n_threads, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch=64/64, q8_0 KV, n_threads={n_threads}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, n_threads, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, n_threads, r["error"], f"n_threads={n_threads} — {r['error']}")
        exp += 1

    # ===================================================================
    # BLOCK E: Extended context with q4_0 at n_gpu=16, batch=64/64
    # ===================================================================
    print("\n[BLOCK E] q4_0 + batch=64/64 + ultra-long context")
    for n_ctx in [16384, 32768, 65536]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q4_0, type_v=Q4_0, timeout=600)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > 5.0 else "discard"
            desc = f"q4_0 KV, n_gpu=16, batch=64/64, n_ctx={n_ctx}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
            append_tsv(exp, tok, vram, n_ctx, "q4_0", "q4_0", True, 16, 64, 64, 10, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, "q4_0", "q4_0", True, 16, 64, 64, 10, r["error"], f"q4_0 n_ctx={n_ctx} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # BLOCK F: n_gpu=17 (squeeze one more layer?)
    # ===================================================================
    print("\n[BLOCK F] n_gpu=17 — squeeze one more layer")
    for kv_type, kv_name in [(Q8_0, "q8_0"), (Q4_0, "q4_0")]:
        r = run_exp(n_gpu=17, n_ctx=512, n_batch=64, n_ubatch=64, type_k=kv_type, type_v=kv_type)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=17, {kv_name} KV, batch=64/64"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, kv_name, kv_name, True, 17, 64, 64, 10, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, kv_name, kv_name, True, 17, 64, 64, 10, r["error"], f"n_gpu=17 {kv_name} — {r['error']}")
        exp += 1

    # ===================================================================
    # BLOCK G: Optimal batch at n_gpu=14 + q4_0 (the second-best branch)
    # ===================================================================
    print("\n[BLOCK G] Batch tuning at n_gpu=14 + q4_0")
    for n_batch, n_ubatch in [(64, 64), (128, 128), (48, 48), (96, 96)]:
        r = run_exp(n_gpu=14, n_ctx=512, n_batch=n_batch, n_ubatch=n_ubatch, type_k=Q4_0, type_v=Q4_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"q4_0 KV, n_gpu=14, batch={n_batch}/{n_ubatch}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q4_0", "q4_0", True, 14, n_batch, n_ubatch, 10, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q4_0", "q4_0", True, 14, n_batch, n_ubatch, 10, r["error"], f"q4_0 batch {n_batch}/{n_ubatch} — {r['error']}")
        exp += 1

    print("\n" + "="*70)
    print(f"PHASE 4b COMPLETE — Best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Experiments this round: {exp - 60}")
    print("="*70)


if __name__ == "__main__":
    main()
