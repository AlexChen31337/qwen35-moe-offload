"""
orchestrate_phase4.py — Phase 4 continuation: subprocess-isolated experiments.
Each experiment runs in its own process to avoid CUDA state corruption.
"""
import json, os, subprocess, sys, time
from pathlib import Path

TSV_PATH = "./results_phase4.tsv"
TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1'}

# GGML constants
Q8_0 = 8
Q4_0 = 2
F16 = 1

def run_exp(n_gpu, n_ctx, n_threads=10, n_batch=32, n_ubatch=32, type_k=Q8_0, type_v=Q8_0, flash=True, timeout=180):
    """Run single experiment in subprocess. Returns parsed JSON or error dict."""
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    print(f"\n  >> n_gpu={n_gpu}, n_ctx={n_ctx}, n_threads={n_threads}, "
          f"batch={n_batch}/{n_ubatch}, KV={tk_name}/{tv_name}, flash={flash}", flush=True)
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")
    
    cmd = [
        sys.executable, "run_single_exp.py",
        "--n_gpu", str(n_gpu),
        "--n_ctx", str(n_ctx),
        "--n_threads", str(n_threads),
        "--n_batch", str(n_batch),
        "--n_ubatch", str(n_ubatch),
        "--type_k", str(type_k),
        "--type_v", str(type_v),
    ]
    if not flash:
        cmd.append("--no_flash_attn")
    
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/tmp/qwen35-moe-offload", env=env
        )
        # Parse stdout for JSON
        for line in proc.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        # No JSON found
        stderr_tail = proc.stderr[-300:] if proc.stderr else ""
        return {"error": "NO_JSON", "detail": f"stdout={proc.stdout[:200]}, stderr={stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "detail": f">{timeout}s"}
    except Exception as e:
        return {"error": "LAUNCH_FAIL", "detail": str(e)[:200]}


def append_tsv(exp, tok, vram, n_ctx, tk, tv, flash, n_gpu, status, desc):
    line = f"{exp}\t{tok:.3f}\t{vram:.0f}\t{n_ctx}\t{tk}\t{tv}\t{flash}\t{n_gpu}\t{status}\t{desc}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"     TSV exp {exp}: {tok:.3f} tok/s [{status}] {desc}", flush=True)


def main():
    best_tok = 10.240  # exp 24: n_gpu=16, q8_0 KV, n_ctx=512
    best_config = "exp 24: n_gpu=16, n_ctx=512, q8_0 KV"
    exp = 31  # Continue from where the errors started

    print("="*70)
    print("Phase 4 Autoresearch — Subprocess-Isolated Continuation")
    print(f"Best so far: {best_tok:.3f} tok/s ({best_config})")
    print("="*70)

    # ===================================================================
    # BLOCK A: Push GPU layers higher with q8_0 (the winning formula)
    # n_gpu=16 hit 10.240. Try 18, 20, 22, 24.
    # ===================================================================
    print("\n[BLOCK A] Pushing GPU layers: 18, 20, 22, 24 (q8_0 KV, n_ctx=512)")
    for n_gpu in [18, 20, 22, 24]:
        r = run_exp(n_gpu=n_gpu, n_ctx=512, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu={n_gpu}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu={n_gpu}, n_ctx=512, q8_0 KV"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, n_gpu, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, n_gpu, r["error"], f"n_gpu={n_gpu} — {r['error']}")
            if r["error"] == "OOM":
                print(f"  OOM at n_gpu={n_gpu} — stopping GPU layer push")
                exp += 1
                break
        exp += 1

    # ===================================================================
    # BLOCK B: n_gpu=16 + context scaling (q8_0)
    # ===================================================================
    print("\n[BLOCK B] n_gpu=16 + context scaling (q8_0 KV)")
    for n_ctx in [1024, 2048, 4096]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, q8_0 KV, n_ctx={n_ctx}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu=16, n_ctx={n_ctx}, q8_0 KV"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, n_ctx, "q8_0", "q8_0", True, 16, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, "q8_0", "q8_0", True, 16, r["error"], f"n_gpu=16 n_ctx={n_ctx} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # BLOCK C: n_gpu=16 + thread tuning
    # ===================================================================
    print("\n[BLOCK C] n_gpu=16 + thread tuning (q8_0 KV, n_ctx=512)")
    for n_threads in [6, 8, 12, 14, 16]:
        r = run_exp(n_gpu=16, n_ctx=512, n_threads=n_threads, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, n_threads={n_threads}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu=16, n_threads={n_threads}, q8_0 KV"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, r["error"], f"n_threads={n_threads} — {r['error']}")
        exp += 1

    # ===================================================================
    # BLOCK D: q4_0 retry at high GPU layers (isolated process should work)
    # ===================================================================
    print("\n[BLOCK D] q4_0 KV at high GPU layers")
    for n_gpu in [10, 14, 16, 18, 20]:
        r = run_exp(n_gpu=n_gpu, n_ctx=512, type_k=Q4_0, type_v=Q4_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"q4_0 KV, n_gpu={n_gpu}, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu={n_gpu}, n_ctx=512, q4_0 KV"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q4_0", "q4_0", True, n_gpu, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q4_0", "q4_0", True, n_gpu, r["error"], f"q4_0 n_gpu={n_gpu} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # BLOCK E: q4_0 + context scaling at best GPU layers
    # ===================================================================
    print("\n[BLOCK E] q4_0 KV + context scaling")
    for n_gpu in [16, 14]:
        for n_ctx in [1024, 2048, 4096, 8192]:
            r = run_exp(n_gpu=n_gpu, n_ctx=n_ctx, type_k=Q4_0, type_v=Q4_0)
            if "error" not in r:
                tok = r["tok_per_sec"]
                vram = r["vram_peak_mb"]
                status = "keep" if tok > best_tok else "discard"
                desc = f"q4_0 KV, n_gpu={n_gpu}, n_ctx={n_ctx}"
                if tok > best_tok:
                    best_tok = tok
                    best_config = f"exp {exp}: n_gpu={n_gpu}, n_ctx={n_ctx}, q4_0 KV"
                    desc += " — NEW BEST"
                    print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
                append_tsv(exp, tok, vram, n_ctx, "q4_0", "q4_0", True, n_gpu, status, desc)
            else:
                append_tsv(exp, 0, 0, n_ctx, "q4_0", "q4_0", True, n_gpu, r["error"], f"q4_0 n_gpu={n_gpu} n_ctx={n_ctx} — {r['error']}")
                if r["error"] == "OOM":
                    break
            exp += 1

    # ===================================================================
    # BLOCK F: Batch size tuning at best config
    # ===================================================================
    print("\n[BLOCK F] Batch size tuning at best known config (n_gpu=16, q8_0, n_ctx=512)")
    for n_batch, n_ubatch in [(64, 64), (128, 128), (16, 16), (64, 32), (128, 32), (256, 32), (256, 256)]:
        r = run_exp(n_gpu=16, n_ctx=512, n_batch=n_batch, n_ubatch=n_ubatch, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch={n_batch}/{n_ubatch}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu=16, batch={n_batch}/{n_ubatch}, q8_0 KV"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, r["error"], f"batch {n_batch}/{n_ubatch} — {r['error']}")
        exp += 1

    # ===================================================================
    # BLOCK G: Asymmetric KV (q8_0 keys, q4_0 values) at high GPU layers
    # ===================================================================
    print("\n[BLOCK G] Asymmetric KV: q8_0 keys + q4_0 values")
    for n_gpu in [16, 18, 20]:
        r = run_exp(n_gpu=n_gpu, n_ctx=512, type_k=Q8_0, type_v=Q4_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"asymmetric q8_0/q4_0, n_gpu={n_gpu}, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: n_gpu={n_gpu}, q8_0K/q4_0V, n_ctx=512"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q4_0", True, n_gpu, status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q4_0", True, n_gpu, r["error"], f"asym n_gpu={n_gpu} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # BLOCK H: Best config (from above) + larger context
    # Push context as far as possible with best throughput config
    # ===================================================================
    print("\n[BLOCK H] Best throughput config + context push")
    # Find what the best n_gpu is by now
    # Try various combos with best known
    for n_ctx in [1024, 2048, 4096, 8192, 16384]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, type_k=Q8_0, type_v=Q8_0, timeout=300)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            desc = f"n_gpu=16, q8_0 KV, n_ctx={n_ctx} — context push"
            status = "keep" if tok > 8.0 else "discard"  # Keep if usable speed
            append_tsv(exp, tok, vram, n_ctx, "q8_0", "q8_0", True, 16, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, "q8_0", "q8_0", True, 16, r["error"], f"ctx push {n_ctx} — {r['error']}")
            if r["error"] == "OOM":
                break
        exp += 1

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print(f"PHASE 4 COMPLETE — Best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Total experiments run: {exp - 31}")
    print("="*70)


if __name__ == "__main__":
    main()
