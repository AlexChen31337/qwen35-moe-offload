"""
orchestrate_phase4c.py — Phase 4 round 3: Retry with freed VRAM.
Ollama unloaded, ~6.8GB available for model.
Continues from exp 79.
"""
import json, os, subprocess, sys, time

TSV_PATH = "./results_phase4.tsv"
TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0'}
Q8_0 = 8
Q4_0 = 2

def run_exp(n_gpu, n_ctx, n_threads=10, n_batch=32, n_ubatch=32, type_k=Q8_0, type_v=Q8_0, flash=True, timeout=300):
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    print(f"\n  >> n_gpu={n_gpu}, n_ctx={n_ctx}, n_threads={n_threads}, "
          f"batch={n_batch}/{n_ubatch}, KV={tk_name}/{tv_name}", flush=True)
    
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
        return {"error": "NO_JSON", "detail": f"rc={proc.returncode} stderr={proc.stderr[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    except Exception as e:
        return {"error": "LAUNCH_FAIL", "detail": str(e)[:200]}


def tsv(exp, tok, vram, n_ctx, tk, tv, flash, n_gpu, nb, nub, nt, status, desc):
    line = f"{exp}\t{tok:.3f}\t{vram:.0f}\t{n_ctx}\t{tk}\t{tv}\t{flash}\t{n_gpu}\t{nb}\t{nub}\t{nt}\t{status}\t{desc}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"     exp {exp}: {tok:.3f} tok/s [{status}]", flush=True)


def main():
    best = 11.850
    best_cfg = "exp 46"
    exp = 79

    print("="*60)
    print(f"Phase 4c — VRAM freed (ollama unloaded). Best: {best:.3f}")
    print("="*60)

    # ===================================================================
    # A: n_gpu=15 (was all OOM with ollama loaded)
    # ===================================================================
    print("\n[A] n_gpu=15 retry")
    for kv, kn in [(Q8_0,"q8_0"),(Q4_0,"q4_0")]:
        for nb,nub in [(32,32),(64,64)]:
            r = run_exp(n_gpu=15, n_ctx=512, n_batch=nb, n_ubatch=nub, type_k=kv, type_v=kv)
            if "error" not in r:
                t = r["tok_per_sec"]; v = r["vram_peak_mb"]
                s = "keep" if t > best else "discard"
                d = f"n_gpu=15, {kn} KV, batch={nb}/{nub}"
                if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
                tsv(exp, t, v, 512, kn, kn, True, 15, nb, nub, 10, s, d)
            else:
                tsv(exp, 0, 0, 512, kn, kn, True, 15, nb, nub, 10, r["error"], f"n_gpu=15 {kn} b{nb} — {r['error']}")
                if r["error"]=="OOM": break
            exp += 1

    # ===================================================================
    # B: n_gpu=17 (squeeze one more layer)
    # ===================================================================
    print("\n[B] n_gpu=17 retry")
    for kv, kn in [(Q8_0,"q8_0"),(Q4_0,"q4_0")]:
        for nb,nub in [(32,32),(64,64)]:
            r = run_exp(n_gpu=17, n_ctx=512, n_batch=nb, n_ubatch=nub, type_k=kv, type_v=kv)
            if "error" not in r:
                t = r["tok_per_sec"]; v = r["vram_peak_mb"]
                s = "keep" if t > best else "discard"
                d = f"n_gpu=17, {kn} KV, batch={nb}/{nub}"
                if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
                tsv(exp, t, v, 512, kn, kn, True, 17, nb, nub, 10, s, d)
            else:
                tsv(exp, 0, 0, 512, kn, kn, True, 17, nb, nub, 10, r["error"], f"n_gpu=17 {kn} b{nb} — {r['error']}")
            exp += 1

    # ===================================================================
    # C: Thread tuning at n_gpu=16, batch=64/64 (was all OOM)
    # ===================================================================
    print("\n[C] Thread tuning retry (n_gpu=16, batch=64/64, q8_0)")
    for nt in [6, 8, 12, 14]:
        r = run_exp(n_gpu=16, n_ctx=512, n_threads=nt, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            t = r["tok_per_sec"]; v = r["vram_peak_mb"]
            s = "keep" if t > best else "discard"
            d = f"n_gpu=16, batch=64/64, q8_0 KV, n_threads={nt}"
            if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
            tsv(exp, t, v, 512, "q8_0", "q8_0", True, 16, 64, 64, nt, s, d)
        else:
            tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, nt, r["error"], f"threads={nt} — {r['error']}")
        exp += 1

    # ===================================================================
    # D: q4_0 KV + batch=64/64 at 14 and 16 GPU layers
    # ===================================================================
    print("\n[D] q4_0 + batch=64/64 retry")
    for n_gpu in [14, 16]:
        for n_ctx in [512, 1024, 2048, 4096]:
            r = run_exp(n_gpu=n_gpu, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q4_0, type_v=Q4_0)
            if "error" not in r:
                t = r["tok_per_sec"]; v = r["vram_peak_mb"]
                s = "keep" if t > best else "discard"
                d = f"q4_0 KV, n_gpu={n_gpu}, batch=64/64, n_ctx={n_ctx}"
                if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
                tsv(exp, t, v, n_ctx, "q4_0", "q4_0", True, n_gpu, 64, 64, 10, s, d)
            else:
                tsv(exp, 0, 0, n_ctx, "q4_0", "q4_0", True, n_gpu, 64, 64, 10, r["error"], f"q4_0 n_gpu={n_gpu} n_ctx={n_ctx} — {r['error']}")
                if r["error"]=="OOM": break
            exp += 1

    # ===================================================================
    # E: Extended context push with q8_0 batch=64/64 (16K, 32K)
    # ===================================================================
    print("\n[E] Ultra-long context (q8_0, batch=64/64, n_gpu=16)")
    for n_ctx in [16384, 32768]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0, timeout=600)
        if "error" not in r:
            t = r["tok_per_sec"]; v = r["vram_peak_mb"]
            s = "keep" if t > 5.0 else "discard"
            d = f"q8_0 KV, n_gpu=16, batch=64/64, n_ctx={n_ctx}"
            if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"
            tsv(exp, t, v, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, 10, s, d)
        else:
            tsv(exp, 0, 0, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, 10, r["error"], f"q8_0 n_ctx={n_ctx} — {r['error']}")
            if r["error"]=="OOM": break
        exp += 1

    # ===================================================================
    # F: q4_0 ultra-long context (16K, 32K, 65K)
    # ===================================================================
    print("\n[F] Ultra-long context (q4_0, batch=64/64, n_gpu=16)")
    for n_ctx in [16384, 32768, 65536]:
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q4_0, type_v=Q4_0, timeout=600)
        if "error" not in r:
            t = r["tok_per_sec"]; v = r["vram_peak_mb"]
            s = "keep" if t > 5.0 else "discard"
            d = f"q4_0 KV, n_gpu=16, batch=64/64, n_ctx={n_ctx}"
            if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"
            tsv(exp, t, v, n_ctx, "q4_0", "q4_0", True, 16, 64, 64, 10, s, d)
        else:
            tsv(exp, 0, 0, n_ctx, "q4_0", "q4_0", True, 16, 64, 64, 10, r["error"], f"q4_0 n_ctx={n_ctx} — {r['error']}")
            if r["error"]=="OOM": break
        exp += 1

    # ===================================================================
    # G: Batch size tuning at n_gpu=14 + q4_0 (the 11.038 config)
    # ===================================================================
    print("\n[G] Batch tuning n_gpu=14, q4_0")
    for nb,nub in [(64,64),(48,48),(96,96),(128,128)]:
        r = run_exp(n_gpu=14, n_ctx=512, n_batch=nb, n_ubatch=nub, type_k=Q4_0, type_v=Q4_0)
        if "error" not in r:
            t = r["tok_per_sec"]; v = r["vram_peak_mb"]
            s = "keep" if t > best else "discard"
            d = f"q4_0 KV, n_gpu=14, batch={nb}/{nub}"
            if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
            tsv(exp, t, v, 512, "q4_0", "q4_0", True, 14, nb, nub, 10, s, d)
        else:
            tsv(exp, 0, 0, 512, "q4_0", "q4_0", True, 14, nb, nub, 10, r["error"], f"q4_0 batch={nb}/{nub} — {r['error']}")
        exp += 1

    # ===================================================================
    # H: n_gpu=18/20 retry (was OOM with ollama)
    # ===================================================================
    print("\n[H] High GPU layers retry (n_gpu=18, 20)")
    for n_gpu in [18, 20]:
        for kv, kn in [(Q8_0,"q8_0"),(Q4_0,"q4_0")]:
            r = run_exp(n_gpu=n_gpu, n_ctx=512, n_batch=32, n_ubatch=32, type_k=kv, type_v=kv)
            if "error" not in r:
                t = r["tok_per_sec"]; v = r["vram_peak_mb"]
                s = "keep" if t > best else "discard"
                d = f"n_gpu={n_gpu}, {kn} KV, batch=32/32"
                if t > best: best=t; best_cfg=f"exp {exp}"; d+=" — NEW BEST"; print(f"  *** NEW BEST: {t:.3f} ***")
                tsv(exp, t, v, 512, kn, kn, True, n_gpu, 32, 32, 10, s, d)
            else:
                tsv(exp, 0, 0, 512, kn, kn, True, n_gpu, 32, 32, 10, r["error"], f"n_gpu={n_gpu} {kn} — {r['error']}")
            exp += 1

    print(f"\n{'='*60}\nPhase 4c DONE — Best: {best:.3f} tok/s ({best_cfg})\nExps: {exp - 79}\n{'='*60}")


if __name__ == "__main__":
    main()
