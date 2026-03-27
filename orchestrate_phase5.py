"""
orchestrate_phase5.py — Phase 5: Beyond parameter tuning.

Explores new axes not tried in Phase 4:
  Axis 1: Batch sweep continuation (48, 56, 72, 80, asymmetric ubatch)
  Axis 2: Context length at optimal config (512 → 32768)
  Axis 3: Flash attention + rope scaling
  Axis 4: Mixed precision (asymmetric KV, iq4_nl)
  Axis 5: op_offload and swa_full flags
  Axis 6: Combined optimizations

Phase 4 best: 11.850 tok/s (exp 46: n_gpu=16, batch=64/64, q8_0 KV, n_ctx=512)
"""
import json, os, subprocess, sys, time
from pathlib import Path

TSV_PATH = "./results_phase5.tsv"
TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 20: 'iq4_nl', 12: 'q4_k', 13: 'q5_k'}

# GGML constants
Q8_0 = 8
Q4_0 = 2
Q4_1 = 3
Q5_0 = 6
Q5_1 = 7
IQ4_NL = 20
Q4_K = 12
Q5_K = 13
F16 = 1


def ensure_vram_free(max_wait=30):
    """Wait until VRAM is back to baseline (< 2000 MB). Also unload any ollama models."""
    # Unload any ollama models
    try:
        subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate",
             "-d", '{"model": "qwen3.5:4b", "keep_alive": 0}'],
            capture_output=True, timeout=5
        )
    except:
        pass
    
    for _ in range(max_wait):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            ).strip()
            vram = float(out.split('\n')[0])
            if vram < 2000:
                return True
        except:
            pass
        time.sleep(1)
    return False


def run_exp(n_gpu, n_ctx, n_threads=10, n_batch=64, n_ubatch=64,
            type_k=Q8_0, type_v=Q8_0, flash=True, timeout=240,
            rope_freq_base=0, long_prompt=False, max_tokens=256,
            op_offload=False, swa_full=False):
    """Run single experiment in subprocess. Returns parsed JSON or error dict."""
    # Ensure VRAM is free before starting
    ensure_vram_free()
    
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    print(f"\n  >> n_gpu={n_gpu}, n_ctx={n_ctx}, batch={n_batch}/{n_ubatch}, "
          f"KV={tk_name}/{tv_name}, flash={flash}", end="", flush=True)
    if rope_freq_base > 0:
        print(f", rope_base={rope_freq_base}", end="", flush=True)
    if long_prompt:
        print(", long_prompt", end="", flush=True)
    if op_offload:
        print(", op_offload", end="", flush=True)
    if swa_full:
        print(", swa_full", end="", flush=True)
    print(flush=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        sys.executable, "run_single_exp_v2.py",
        "--n_gpu", str(n_gpu),
        "--n_ctx", str(n_ctx),
        "--n_threads", str(n_threads),
        "--n_batch", str(n_batch),
        "--n_ubatch", str(n_ubatch),
        "--type_k", str(type_k),
        "--type_v", str(type_v),
        "--max_tokens", str(max_tokens),
    ]
    if not flash:
        cmd.append("--no_flash_attn")
    if rope_freq_base > 0:
        cmd.extend(["--rope_freq_base", str(rope_freq_base)])
    if long_prompt:
        cmd.append("--long_prompt")
    if op_offload:
        cmd.append("--op_offload")
    if swa_full:
        cmd.append("--swa_full")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/tmp/qwen35-moe-offload", env=env
        )
        # Wait for CUDA memory release
        time.sleep(3)
        
        for line in proc.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        return {"error": "NO_JSON", "detail": f"rc={proc.returncode}, stdout={proc.stdout[:300]}, stderr={stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "detail": f">{timeout}s"}
    except Exception as e:
        return {"error": "LAUNCH_FAIL", "detail": str(e)[:200]}


def append_tsv(exp, tok, vram, n_ctx, tk, tv, flash, n_gpu, n_batch, n_ubatch, notes, status, desc):
    """Append a result line to the TSV file."""
    line = f"{exp}\t{tok:.3f}\t{vram:.0f}\t{n_ctx}\t{tk}\t{tv}\t{flash}\t{n_gpu}\t{n_batch}\t{n_ubatch}\t{notes}\t{status}\t{desc}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"     TSV exp {exp}: {tok:.3f} tok/s [{status}] {desc}", flush=True)


def init_tsv():
    """Initialize TSV header if file doesn't exist."""
    if not Path(TSV_PATH).exists():
        with open(TSV_PATH, "w") as f:
            f.write("exp\ttok_per_sec\tvram_peak_mb\tn_ctx\ttype_k\ttype_v\tflash_attn\tn_gpu_layers\tn_batch\tn_ubatch\tnotes\tstatus\tdescription\n")


def get_last_exp():
    """Get the last experiment number from TSV."""
    if not Path(TSV_PATH).exists():
        return 0
    with open(TSV_PATH) as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return 0
    for line in reversed(lines):
        parts = line.strip().split('\t')
        if parts and parts[0].isdigit():
            return int(parts[0])
    return 0


def main():
    init_tsv()
    best_tok = 11.850  # Phase 4 best
    best_config = "phase4 exp 46: n_gpu=16, batch=64/64, q8_0 KV, n_ctx=512"
    exp = get_last_exp() + 1
    start_exp = exp

    print("=" * 70)
    print("Phase 5 Autoresearch — Beyond Parameter Tuning")
    print(f"Phase 4 best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Starting at exp {exp}")
    print("=" * 70)

    # ===================================================================
    # AXIS 1: Batch sweep continuation
    # Phase 4 tried: 16, 32, 64, 128, 256. Try gaps: 48, 56, 72, 80, 96
    # Also try asymmetric ubatch combos
    # ===================================================================
    print("\n[AXIS 1a] Batch sweep: 48, 56, 72, 80, 96 (symmetric)")
    for batch in [48, 56, 72, 80, 96]:
        r = run_exp(n_gpu=16, n_ctx=512, n_batch=batch, n_ubatch=batch, type_k=Q8_0, type_v=Q8_0)
        tk_name, tv_name = "q8_0", "q8_0"
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch={batch}/{batch}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, tk_name, tv_name, True, 16, batch, batch, "", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, tk_name, tv_name, True, 16, batch, batch, r["error"], r["error"], f"batch={batch}/{batch} — {r.get('detail','')[:80]}")
        exp += 1

    print("\n[AXIS 1b] Asymmetric ubatch: n_batch=64 with varying ubatch")
    for ubatch in [16, 32, 48, 128, 256]:
        r = run_exp(n_gpu=16, n_ctx=512, n_batch=64, n_ubatch=ubatch, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch=64/{ubatch}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, ubatch, "asym_ubatch", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, ubatch, r["error"], r["error"], f"batch=64/{ubatch} — {r.get('detail','')[:80]}")
        exp += 1

    print("\n[AXIS 1c] Asymmetric: n_batch=128, varying ubatch")
    for ubatch in [16, 32, 48, 64, 96]:
        r = run_exp(n_gpu=16, n_ctx=512, n_batch=128, n_ubatch=ubatch, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch=128/{ubatch}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 128, ubatch, "asym_ubatch", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 128, ubatch, r["error"], r["error"], f"batch=128/{ubatch} — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 2: Context length sweep at best config
    # ===================================================================
    print("\n[AXIS 2] Context length sweep at best config (n_gpu=16, q8_0, batch=64/64)")
    for n_ctx in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        long = n_ctx > 4096
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0,
                    long_prompt=long, timeout=300 if n_ctx >= 16384 else 240)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else ("usable" if tok > 8.0 else "discard")
            notes = "long_prompt" if long else ""
            desc = f"n_gpu=16, batch=64/64, q8_0 KV, n_ctx={n_ctx}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, notes, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, "q8_0", "q8_0", True, 16, 64, 64, r["error"], r["error"], f"ctx={n_ctx} — {r.get('detail','')[:80]}")
            if r["error"] == "OOM" and n_ctx >= 32768:
                break
        exp += 1

    # ===================================================================
    # AXIS 3: Rope freq base for extended context
    # ===================================================================
    print("\n[AXIS 3a] Rope freq base at n_ctx=8192")
    for rope_base in [10000, 100000, 500000, 1000000]:
        r = run_exp(n_gpu=16, n_ctx=8192, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0,
                    rope_freq_base=rope_base, long_prompt=True)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch=64/64, q8_0 KV, n_ctx=8192, rope_base={rope_base}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 8192, "q8_0", "q8_0", True, 16, 64, 64, f"rope={rope_base}", status, desc)
        else:
            append_tsv(exp, 0, 0, 8192, "q8_0", "q8_0", True, 16, 64, 64, f"rope={rope_base}|{r['error']}", r["error"], f"rope={rope_base} — {r.get('detail','')[:80]}")
        exp += 1

    print("\n[AXIS 3b] Rope freq base at n_ctx=16384")
    for rope_base in [10000, 500000, 1000000]:
        r = run_exp(n_gpu=16, n_ctx=16384, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0,
                    rope_freq_base=rope_base, long_prompt=True, timeout=300)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch=64/64, q8_0 KV, n_ctx=16384, rope_base={rope_base}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
            append_tsv(exp, tok, vram, 16384, "q8_0", "q8_0", True, 16, 64, 64, f"rope={rope_base}", status, desc)
        else:
            append_tsv(exp, 0, 0, 16384, "q8_0", "q8_0", True, 16, 64, 64, f"rope={rope_base}|{r['error']}", r["error"], f"rope={rope_base} ctx=16K — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 4: Mixed precision KV — systematic
    # ===================================================================
    print("\n[AXIS 4a] Reverse asymmetric: q4_0 K + q8_0 V (never tried)")
    for n_gpu in [14, 16]:
        r = run_exp(n_gpu=n_gpu, n_ctx=512, type_k=Q4_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"q4_0K/q8_0V, n_gpu={n_gpu}, batch=64/64, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q4_0", "q8_0", True, n_gpu, 64, 64, "reverse_asym", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q4_0", "q8_0", True, n_gpu, 64, 64, r["error"], r["error"], f"q4_0K/q8_0V n_gpu={n_gpu} — {r.get('detail','')[:80]}")
        exp += 1

    print("\n[AXIS 4b] iq4_nl K/V at n_gpu=16")
    r = run_exp(n_gpu=16, n_ctx=512, type_k=IQ4_NL, type_v=IQ4_NL)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        status = "keep" if tok > best_tok else "discard"
        desc = f"iq4_nl KV, n_gpu=16, batch=64/64, n_ctx=512"
        if tok > best_tok:
            best_tok = tok
            best_config = f"exp {exp}: {desc}"
            desc += " — NEW BEST"
            print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
        append_tsv(exp, tok, vram, 512, "iq4_nl", "iq4_nl", True, 16, 64, 64, "", status, desc)
    else:
        append_tsv(exp, 0, 0, 512, "iq4_nl", "iq4_nl", True, 16, 64, 64, r["error"], r["error"], f"iq4_nl KV n_gpu=16 — {r.get('detail','')[:80]}")
    exp += 1

    print("\n[AXIS 4c] q4_k/q5_k KV types")
    for tk, tv, tk_name, tv_name in [(Q4_K, Q4_K, "q4_k", "q4_k"), (Q5_K, Q5_K, "q5_k", "q5_k"),
                                       (Q8_0, IQ4_NL, "q8_0", "iq4_nl"), (IQ4_NL, Q8_0, "iq4_nl", "q8_0")]:
        r = run_exp(n_gpu=16, n_ctx=512, type_k=tk, type_v=tv)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"{tk_name}K/{tv_name}V, n_gpu=16, batch=64/64, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, tk_name, tv_name, True, 16, 64, 64, "", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, tk_name, tv_name, True, 16, 64, 64, r["error"], r["error"], f"{tk_name}/{tv_name} — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 5: op_offload and swa_full flags
    # ===================================================================
    print("\n[AXIS 5a] op_offload at best config")
    r = run_exp(n_gpu=16, n_ctx=512, type_k=Q8_0, type_v=Q8_0, op_offload=True)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        status = "keep" if tok > best_tok else "discard"
        desc = f"n_gpu=16, q8_0 KV, batch=64/64, op_offload=True"
        if tok > best_tok:
            best_tok = tok
            best_config = f"exp {exp}: {desc}"
            desc += " — NEW BEST"
            print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
        append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, "op_offload", status, desc)
    else:
        append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, f"op_offload|{r['error']}", r["error"], f"op_offload — {r.get('detail','')[:80]}")
    exp += 1

    print("\n[AXIS 5b] swa_full at best config")
    r = run_exp(n_gpu=16, n_ctx=512, type_k=Q8_0, type_v=Q8_0, swa_full=True)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        status = "keep" if tok > best_tok else "discard"
        desc = f"n_gpu=16, q8_0 KV, batch=64/64, swa_full=True"
        if tok > best_tok:
            best_tok = tok
            best_config = f"exp {exp}: {desc}"
            desc += " — NEW BEST"
            print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
        append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, "swa_full", status, desc)
    else:
        append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, f"swa_full|{r['error']}", r["error"], f"swa_full — {r.get('detail','')[:80]}")
    exp += 1

    print("\n[AXIS 5c] Both op_offload + swa_full")
    r = run_exp(n_gpu=16, n_ctx=512, type_k=Q8_0, type_v=Q8_0, op_offload=True, swa_full=True)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        status = "keep" if tok > best_tok else "discard"
        desc = f"n_gpu=16, q8_0 KV, batch=64/64, op_offload+swa_full"
        if tok > best_tok:
            best_tok = tok
            best_config = f"exp {exp}: {desc}"
            desc += " — NEW BEST"
            print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
        append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, "op_offload+swa_full", status, desc)
    else:
        append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, f"both|{r['error']}", r["error"], f"op_offload+swa_full — {r.get('detail','')[:80]}")
    exp += 1

    # ===================================================================
    # AXIS 6: Combined optimizations — best batch × best KV × flags
    # ===================================================================
    print("\n[AXIS 6] Combined optimizations — winner configs + new flags")
    # Try the best batch from Axis 1 with op_offload
    combos = [
        # (n_batch, n_ubatch, type_k, type_v, op_offload, swa_full, n_ctx, notes)
        (48, 48, Q8_0, Q8_0, True, False, 512, "batch48+op_offload"),
        (56, 56, Q8_0, Q8_0, True, False, 512, "batch56+op_offload"),
        (72, 72, Q8_0, Q8_0, True, False, 512, "batch72+op_offload"),
        (64, 64, Q4_0, Q8_0, False, False, 512, "q4_0K/q8_0V batch64"),
        (64, 64, Q8_0, Q4_0, False, False, 512, "q8_0K/q4_0V batch64 (retry)"),
        (64, 64, Q8_0, Q8_0, True, True, 2048, "best+both flags ctx=2048"),
        (64, 64, Q8_0, Q8_0, True, True, 4096, "best+both flags ctx=4096"),
        (64, 64, Q8_0, Q8_0, False, False, 512, "baseline retest (variance check)"),
        (64, 64, Q8_0, Q8_0, False, False, 512, "baseline retest 2 (variance check)"),
        (64, 64, Q8_0, Q8_0, False, False, 512, "baseline retest 3 (variance check)"),
    ]
    for n_batch, n_ubatch, tk, tv, op_off, swa, n_ctx, notes in combos:
        tk_name = TYPE_NAMES.get(tk, str(tk))
        tv_name = TYPE_NAMES.get(tv, str(tv))
        r = run_exp(n_gpu=16, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
                    type_k=tk, type_v=tv, op_offload=op_off, swa_full=swa)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, batch={n_batch}/{n_ubatch}, {tk_name}K/{tv_name}V, n_ctx={n_ctx}, {notes}"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, n_ctx, tk_name, tv_name, True, 16, n_batch, n_ubatch, notes, status, desc)
        else:
            append_tsv(exp, 0, 0, n_ctx, tk_name, tv_name, True, 16, n_batch, n_ubatch, f"{notes}|{r['error']}", r["error"], f"{notes} — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 7: No flash_attn baseline comparison at best config
    # ===================================================================
    print("\n[AXIS 7] No flash_attn comparison")
    r = run_exp(n_gpu=16, n_ctx=512, n_batch=64, n_ubatch=64, type_k=F16, type_v=F16, flash=False)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        desc = f"n_gpu=16, batch=64/64, f16 KV, no flash_attn, n_ctx=512"
        status = "keep" if tok > best_tok else "discard"
        append_tsv(exp, tok, vram, 512, "f16", "f16", False, 16, 64, 64, "no_flash", status, desc)
    else:
        append_tsv(exp, 0, 0, 512, "f16", "f16", False, 16, 64, 64, r["error"], r["error"], f"no flash — {r.get('detail','')[:80]}")
    exp += 1

    # ===================================================================
    # AXIS 8: n_gpu=14 with best batch combos (fits in VRAM more reliably)
    # ===================================================================
    print("\n[AXIS 8] n_gpu=14 batch sweep")
    for batch in [48, 56, 64, 72, 80]:
        r = run_exp(n_gpu=14, n_ctx=512, n_batch=batch, n_ubatch=batch, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=14, batch={batch}/{batch}, q8_0 KV, n_ctx=512"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 14, batch, batch, "", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 14, batch, batch, r["error"], r["error"], f"n_gpu=14 batch={batch} — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 9: Thread count fine-tuning at n_gpu=16 batch=64
    # Phase 4 showed n_threads=10 is best. Try 9, 11 (fine grid)
    # ===================================================================
    print("\n[AXIS 9] Fine thread tuning: 9, 11 threads")
    for n_threads in [9, 11]:
        r = run_exp(n_gpu=16, n_ctx=512, n_threads=n_threads, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0)
        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = f"n_gpu=16, n_threads={n_threads}, batch=64/64, q8_0 KV"
            if tok > best_tok:
                best_tok = tok
                best_config = f"exp {exp}: {desc}"
                desc += " — NEW BEST"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
            append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, f"thr={n_threads}", status, desc)
        else:
            append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, f"thr={n_threads}|{r['error']}", r["error"], f"n_threads={n_threads} — {r.get('detail','')[:80]}")
        exp += 1

    # ===================================================================
    # AXIS 10: Longer generation (512 tokens) — latency measurement
    # ===================================================================
    print("\n[AXIS 10] Longer generation: 512 tokens")
    r = run_exp(n_gpu=16, n_ctx=512, n_batch=64, n_ubatch=64, type_k=Q8_0, type_v=Q8_0, max_tokens=512)
    if "error" not in r:
        tok = r["tok_per_sec"]
        vram = r["vram_peak_mb"]
        desc = f"n_gpu=16, batch=64/64, q8_0 KV, max_tokens=512"
        status = "keep" if tok > best_tok else "discard"
        append_tsv(exp, tok, vram, 512, "q8_0", "q8_0", True, 16, 64, 64, "gen=512tok", status, desc)
        if tok > best_tok:
            best_tok = tok
            best_config = f"exp {exp}: {desc}"
            print(f"  *** NEW BEST: {tok:.3f} tok/s ***")
    else:
        append_tsv(exp, 0, 0, 512, "q8_0", "q8_0", True, 16, 64, 64, f"gen512|{r['error']}", r["error"], f"512tok gen — {r.get('detail','')[:80]}")
    exp += 1

    # ===================================================================
    # SUMMARY
    # ===================================================================
    total = exp - start_exp
    print("\n" + "=" * 70)
    print(f"PHASE 5 ROUND 1 COMPLETE — Best: {best_tok:.3f} tok/s ({best_config})")
    print(f"Total experiments: {total}")
    print("=" * 70)
    return best_tok, best_config, exp


if __name__ == "__main__":
    best_tok, best_config, exp = main()
    # After round 1, we could continue with more targeted experiments
    # based on what we learned
