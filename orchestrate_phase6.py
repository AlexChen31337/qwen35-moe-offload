#!/usr/bin/env python3
"""Phase 6 orchestrator — runs Rust bench experiments systematically."""

import subprocess
import json
import time
import os

BENCH = "/tmp/qwen35-moe-offload/run_phase6.sh"
RESULTS = "/tmp/qwen35-moe-offload/results_phase6.tsv"
REPO = "/tmp/qwen35-moe-offload"

# Start from exp_id after existing results
def next_exp_id():
    if not os.path.exists(RESULTS):
        return 1
    with open(RESULTS) as f:
        lines = f.readlines()
    # Skip header, find max exp_id
    max_id = 0
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if parts:
            try:
                max_id = max(max_id, int(parts[0]))
            except ValueError:
                pass
    return max_id + 1

def run_experiment(exp_id: int, n_gpu: int, n_ctx: int, n_batch: int, n_ubatch: int,
                   type_k: int, type_v: int, flash_attn: int, label: str,
                   n_gen: int = 128, prompt: str = "The meaning of life is"):
    """Run a single experiment and return the result line."""
    cmd = [
        BENCH,
        "--exp-id", str(exp_id),
        "--n-gpu", str(n_gpu),
        "--n-ctx", str(n_ctx),
        "--n-batch", str(n_batch),
        "--n-ubatch", str(n_ubatch),
        "--type-k", str(type_k),
        "--type-v", str(type_v),
        "--flash-attn", str(flash_attn),
        "--n-gen", str(n_gen),
        "--label", label,
        "--prompt", prompt,
    ]
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/llama_cpp/lib:"
        "/usr/local/lib/ollama/cuda_v12:" +
        env.get("LD_LIBRARY_PATH", "")
    )
    
    print(f"\n{'='*60}")
    print(f"EXP {exp_id}: n_gpu={n_gpu} n_ctx={n_ctx} batch={n_batch}/{n_ubatch} "
          f"kv={type_k} flash={flash_attn} [{label}]")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        output = result.stdout + result.stderr
        
        # Parse result from the output
        for line in output.split('\n'):
            if line.startswith('exp_id='):
                parts = line.split()
                for p in parts:
                    if p.startswith('tok/s='):
                        tok_s = float(p.split('=')[1])
                        print(f"  → {tok_s:.3f} tok/s")
                        return tok_s
        
        # Fallback: parse from TSV file
        with open(RESULTS) as f:
            last = f.readlines()[-1].strip().split('\t')
            tok_s = float(last[1])
            status = last[11]
            print(f"  → {tok_s:.3f} tok/s (status={status})")
            return tok_s
            
    except subprocess.TimeoutExpired:
        print(f"  → TIMEOUT")
        return 0.0
    except Exception as e:
        print(f"  → ERROR: {e}")
        return 0.0

def git_commit_push(message: str):
    """Commit results and push."""
    os.chdir(REPO)
    subprocess.run(["git", "add", "results_phase6.tsv"], capture_output=True)
    subprocess.run(["git", "commit", "-m", message], capture_output=True)
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/id_ed25519_alexchen"
    subprocess.run(["git", "push", "origin", "phase6/rust-bench"], capture_output=True, env=env)

def main():
    exp_id = next_exp_id()
    best_tok_s = 8.662  # Current best from exp 3
    best_config = "n_gpu=16, batch=64/64, q8_0, n_ctx=512"
    
    # ggml_type constants
    F32 = 0; F16 = 1; Q4_0 = 2; Q5_0 = 6; Q5_1 = 7; Q8_0 = 8; IQ4_NL = 20
    
    experiments = []
    
    # === Axis 1: n_gpu sweep at batch=64/64, q8_0, n_ctx=512 ===
    for n_gpu in [15, 17, 12, 10, 8]:
        experiments.append({
            "n_gpu": n_gpu, "n_ctx": 512, "n_batch": 64, "n_ubatch": 64,
            "type_k": Q8_0, "type_v": Q8_0, "flash_attn": 1,
            "label": f"ngpu_sweep_{n_gpu}"
        })
    
    # === Axis 2: Batch sweep at n_gpu=16, q8_0, n_ctx=512 ===
    for batch, ubatch in [(32, 32), (48, 48), (96, 96), (128, 128), (256, 256),
                          (32, 64), (64, 32), (96, 64), (128, 32)]:
        experiments.append({
            "n_gpu": 16, "n_ctx": 512, "n_batch": batch, "n_ubatch": ubatch,
            "type_k": Q8_0, "type_v": Q8_0, "flash_attn": 1,
            "label": f"batch_{batch}_{ubatch}"
        })
    
    # === Axis 3: KV type sweep at n_gpu=16, batch=64/64, n_ctx=512 ===
    for kv_type, kv_name in [(F16, "f16"), (Q4_0, "q4_0"), (Q5_0, "q5_0"), (IQ4_NL, "iq4_nl"), (F32, "f32")]:
        experiments.append({
            "n_gpu": 16, "n_ctx": 512, "n_batch": 64, "n_ubatch": 64,
            "type_k": kv_type, "type_v": kv_type, "flash_attn": 1,
            "label": f"kv_{kv_name}"
        })
    
    # === Axis 4: Flash attn off at best config ===
    experiments.append({
        "n_gpu": 16, "n_ctx": 512, "n_batch": 64, "n_ubatch": 64,
        "type_k": Q8_0, "type_v": Q8_0, "flash_attn": 0,
        "label": "no_flash"
    })
    
    # === Axis 5: n_ctx ladder at n_gpu=16, batch=64/64, q8_0 ===
    for n_ctx in [1024, 2048, 4096, 8192, 16384, 32768]:
        experiments.append({
            "n_gpu": 16, "n_ctx": n_ctx, "n_batch": 64, "n_ubatch": 64,
            "type_k": Q8_0, "type_v": Q8_0, "flash_attn": 1,
            "label": f"ctx_{n_ctx}"
        })
    
    # === Axis 6: Large context with iq4_nl KV ===
    for n_ctx in [8192, 16384, 32768, 65536]:
        experiments.append({
            "n_gpu": 16, "n_ctx": n_ctx, "n_batch": 64, "n_ubatch": 64,
            "type_k": IQ4_NL, "type_v": IQ4_NL, "flash_attn": 1,
            "label": f"iq4nl_ctx_{n_ctx}"
        })
    
    # === Axis 7: iq4_nl + reduced n_gpu for large context ===
    for n_gpu in [14, 12, 10]:
        experiments.append({
            "n_gpu": n_gpu, "n_ctx": 65536, "n_batch": 64, "n_ubatch": 64,
            "type_k": IQ4_NL, "type_v": IQ4_NL, "flash_attn": 1,
            "label": f"iq4nl_65k_ngpu{n_gpu}"
        })
    
    # Run all experiments
    for i, exp in enumerate(experiments):
        eid = exp_id + i
        tok_s = run_experiment(eid, **exp)
        
        if tok_s > best_tok_s:
            best_tok_s = tok_s
            best_config = str(exp)
            print(f"\n🏆 NEW BEST: {tok_s:.3f} tok/s — {best_config}")
        
        # Commit every 5 experiments
        if (i + 1) % 5 == 0:
            git_commit_push(f"phase6: exp {exp_id}-{eid} — best={best_tok_s:.3f} tok/s")
            print(f"\n📦 Committed experiments {exp_id}-{eid}")
        
        # Brief pause between experiments to let GPU cool
        time.sleep(2)
    
    # Final commit
    git_commit_push(f"phase6: all experiments complete — best={best_tok_s:.3f} tok/s")
    
    print(f"\n{'='*60}")
    print(f"FINAL BEST: {best_tok_s:.3f} tok/s")
    print(f"Config: {best_config}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
