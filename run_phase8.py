#!/usr/bin/env python3
"""
run_phase8.py — Phase 8: Autonomous experiment runner.
Runs each experiment in an isolated subprocess to avoid CUDA state leaks.
Appends results to results_phase8.tsv.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

RESULTS_FILE = "results_phase8.tsv"
HEADER = "exp_id\ttok_s\tvram_mb\tn_ctx\tkv_type_k\tkv_type_v\tflash_attn\tn_gpu\tn_batch\tn_ubatch\tn_threads\tgen_tokens\tlabel\tstatus\tnotes"

TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 20: 'iq4_nl', 12: 'q4_k', 13: 'q5_k'}

def ensure_header():
    if not Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, 'w') as f:
            f.write(HEADER + '\n')

def run_experiment(exp_id, n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
                   type_k=8, type_v=8, flash_attn=True, max_tokens=256, label="", 
                   long_prompt=False, op_offload=False, swa_full=False):
    """Run a single experiment in subprocess, return result dict."""
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
    if flash_attn:
        cmd.append("--flash_attn")
    else:
        cmd.append("--no_flash_attn")
    if long_prompt:
        cmd.append("--long_prompt")
    if op_offload:
        cmd.append("--op_offload")
    if swa_full:
        cmd.append("--swa_full")
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")
    
    tk_name = TYPE_NAMES.get(type_k, str(type_k))
    tv_name = TYPE_NAMES.get(type_v, str(type_v))
    
    print(f"\n{'='*60}")
    print(f"EXP {exp_id}: {label}")
    print(f"  n_gpu={n_gpu} n_ctx={n_ctx} batch={n_batch}/{n_ubatch} threads={n_threads}")
    print(f"  kv={tk_name}/{tv_name} flash={flash_attn} gen={max_tokens}")
    print(f"{'='*60}", flush=True)
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd="/tmp/qwen35-moe-offload", env=env
        )
        
        # Parse JSON from stdout (last line usually)
        stdout_lines = result.stdout.strip().split('\n')
        json_line = None
        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith('{'):
                json_line = line
                break
        
        if json_line:
            data = json.loads(json_line)
            if "error" in data:
                status = "crash" if data["error"] == "OOM" else "error"
                notes = data.get("detail", data["error"])
                tok_s = 0.0
                vram = 0
            else:
                status = "ok"
                tok_s = data["tok_per_sec"]
                vram = data["vram_peak_mb"]
                notes = (f"n_gpu={n_gpu}, batch={n_batch}/{n_ubatch}, {tk_name} KV, "
                        f"n_ctx={n_ctx}, flash={'1' if flash_attn else '0'} — "
                        f"wall={tok_s:.3f}tok/s, gen={max_tokens}, "
                        f"t_eval={data['elapsed']*1000:.1f}ms, threads={n_threads}")
        else:
            status = "crash"
            tok_s = 0.0
            vram = 0
            notes = f"No JSON output. stderr: {result.stderr[:200]}"
        
    except subprocess.TimeoutExpired:
        status = "timeout"
        tok_s = 0.0
        vram = 0
        notes = "Timeout after 300s"
    except Exception as e:
        status = "error"
        tok_s = 0.0
        vram = 0
        notes = str(e)[:200]
    
    # Append to TSV
    row = f"{exp_id}\t{tok_s:.3f}\t{vram}\t{n_ctx}\t{tk_name}\t{tv_name}\t{flash_attn}\t{n_gpu}\t{n_batch}\t{n_ubatch}\t{n_threads}\t{max_tokens}\t{label}\t{status}\t{notes}"
    with open(RESULTS_FILE, 'a') as f:
        f.write(row + '\n')
    
    print(f"  → {tok_s:.3f} tok/s | VRAM: {vram}MB | {status}")
    return {"exp_id": exp_id, "tok_s": tok_s, "vram": vram, "status": status, "notes": notes}


if __name__ == "__main__":
    # Quick test mode
    ensure_header()
    r = run_experiment(1, label="test_baseline", n_gpu=16, n_batch=256, n_ubatch=64, n_threads=12, max_tokens=256)
    print(f"\nResult: {r}")
