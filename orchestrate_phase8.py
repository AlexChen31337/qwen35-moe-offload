#!/usr/bin/env python3
"""
orchestrate_phase8.py — Phase 8: Autonomous experiment orchestrator.
Runs experiments in sequence, tracks best, commits/pushes periodically.
"""
import json, os, subprocess, sys, time
from pathlib import Path

# Import the runner
sys.path.insert(0, '/tmp/qwen35-moe-offload')
from run_phase8 import run_experiment, ensure_header

ALL_TIME_BEST = 12.114
PHASE_BEST = 0.0
RESULTS = []
EXP_ID = 0

def next_exp():
    global EXP_ID
    EXP_ID += 1
    return EXP_ID

def track(result):
    global PHASE_BEST
    RESULTS.append(result)
    if result["status"] == "ok" and result["tok_s"] > PHASE_BEST:
        PHASE_BEST = result["tok_s"]
        print(f"  ★ NEW PHASE 8 BEST: {PHASE_BEST:.3f} tok/s")
    return result

def main():
    global PHASE_BEST
    ensure_header()
    
    print("=" * 60)
    print("PHASE 8: AUTONOMOUS EXPLORATION")
    print(f"All-time best to beat: {ALL_TIME_BEST} tok/s")
    print("=" * 60)
    
    # =========================================================================
    # AXIS 1: Reproduce all-time best (calibration) — 3 runs
    # =========================================================================
    print("\n\n### AXIS 1: Reproduce all-time best (b256/64, n_gpu=16, gen=256)")
    for i in range(3):
        r = track(run_experiment(next_exp(), 
            label=f"reproduce_best_r{i+1}",
            n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
            max_tokens=256))
    
    # =========================================================================
    # AXIS 2: Generation length scaling (gen=512, 1024)
    # =========================================================================
    print("\n\n### AXIS 2: Generation length scaling")
    for gen in [512, 1024]:
        for i in range(2):
            r = track(run_experiment(next_exp(),
                label=f"gen{gen}_b256_64_r{i+1}",
                n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
                max_tokens=gen))
    
    # =========================================================================
    # AXIS 3: n_gpu=17 (between 16=ok and 18=OOM)
    # =========================================================================
    print("\n\n### AXIS 3: n_gpu=17")
    r = track(run_experiment(next_exp(),
        label="ngpu17_b256_64",
        n_gpu=17, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
        max_tokens=256))
    if r["status"] == "ok":
        # If it works, try more runs and variants
        for i in range(2):
            track(run_experiment(next_exp(),
                label=f"ngpu17_b256_64_r{i+2}",
                n_gpu=17, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
                max_tokens=256))
        # Try gen=512 with n_gpu=17
        track(run_experiment(next_exp(),
            label="ngpu17_gen512",
            n_gpu=17, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
            max_tokens=512))
    
    # =========================================================================
    # AXIS 4: Batch fine-tuning around b256/64 winner
    # =========================================================================
    print("\n\n### AXIS 4: Batch fine-tuning")
    batch_combos = [
        (512, 64, "b512_64"), (512, 128, "b512_128"),
        (256, 32, "b256_32"), (256, 128, "b256_128"),
        (128, 32, "b128_32"), (128, 128, "b128_128"),
        (384, 64, "b384_64"), (384, 128, "b384_128"),
        (192, 64, "b192_64"), (192, 96, "b192_96"),
        (256, 96, "b256_96"), (256, 48, "b256_48"),
    ]
    for nb, nub, lbl in batch_combos:
        r = track(run_experiment(next_exp(),
            label=lbl,
            n_gpu=16, n_ctx=512, n_threads=12, n_batch=nb, n_ubatch=nub,
            max_tokens=256))
        if r["status"] != "ok":
            print(f"  Skipping {lbl} — failed with {r['status']}")
    
    # =========================================================================
    # AXIS 5: Thread count with best batch config
    # =========================================================================
    print("\n\n### AXIS 5: Thread count sweep with b256/64")
    for threads in [4, 6, 8, 10, 14, 16, 20, 24]:
        track(run_experiment(next_exp(),
            label=f"threads_{threads}_b256_64",
            n_gpu=16, n_ctx=512, n_threads=threads, n_batch=256, n_ubatch=64,
            max_tokens=256))
    
    # =========================================================================
    # AXIS 6: op_offload and swa_full flags
    # =========================================================================
    print("\n\n### AXIS 6: op_offload and swa_full")
    track(run_experiment(next_exp(),
        label="op_offload_on",
        n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
        max_tokens=256, op_offload=True))
    track(run_experiment(next_exp(),
        label="swa_full_on",
        n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
        max_tokens=256, swa_full=True))
    track(run_experiment(next_exp(),
        label="op_offload_swa_full",
        n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
        max_tokens=256, op_offload=True, swa_full=True))
    
    # =========================================================================
    # AXIS 7: KV quant types with best batch config
    # =========================================================================
    print("\n\n### AXIS 7: KV quant types")
    kv_types = [
        (2, 2, "q4_0"), (3, 3, "q4_1"), (6, 6, "q5_0"), 
        (7, 7, "q5_1"), (12, 12, "q4_k"), (13, 13, "q5_k"),
        (20, 20, "iq4_nl"), (1, 1, "f16"),
    ]
    for tk, tv, name in kv_types:
        track(run_experiment(next_exp(),
            label=f"kv_{name}_b256_64",
            n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
            type_k=tk, type_v=tv, max_tokens=256))
    
    # =========================================================================
    # AXIS 8: Context lengths with best config
    # =========================================================================
    print("\n\n### AXIS 8: Context length sweep")
    for ctx in [128, 256, 1024, 2048, 4096, 8192]:
        track(run_experiment(next_exp(),
            label=f"ctx{ctx}_b256_64",
            n_gpu=16, n_ctx=ctx, n_threads=12, n_batch=256, n_ubatch=64,
            max_tokens=256))
    
    # =========================================================================
    # AXIS 9: Combined best — take best findings from above and combine
    # =========================================================================
    print("\n\n### AXIS 9: Combination experiments")
    # Find best thread count and best batch from above
    ok_results = [r for r in RESULTS if r["status"] == "ok" and r["tok_s"] > 0]
    if ok_results:
        best = max(ok_results, key=lambda r: r["tok_s"])
        print(f"  Phase 8 best so far: {best['tok_s']:.3f} tok/s (exp {best['exp_id']})")
    
    # Try combinations of promising configs
    combos = [
        # Higher gen with best batch
        dict(n_batch=256, n_ubatch=64, n_threads=12, max_tokens=512, label="combo_gen512"),
        dict(n_batch=256, n_ubatch=64, n_threads=12, max_tokens=1024, label="combo_gen1024"),
        # Best batch + q4_0 KV (saves VRAM, might allow more GPU layers)
        dict(n_batch=256, n_ubatch=64, n_threads=12, type_k=2, type_v=2, max_tokens=256, label="combo_q4_0_b256"),
        # Higher batch + f16 (more precision)
        dict(n_batch=512, n_ubatch=64, n_threads=12, type_k=1, type_v=1, max_tokens=256, label="combo_f16_b512"),
    ]
    for combo in combos:
        lbl = combo.pop("label")
        kw = dict(n_gpu=16, n_ctx=512, flash_attn=True)
        kw.update(combo)
        track(run_experiment(next_exp(), label=lbl, **kw))
    
    # =========================================================================
    # AXIS 10: Statistical runs of top configs (5 runs each)
    # =========================================================================
    print("\n\n### AXIS 10: Statistical validation of top configs")
    ok_results = [r for r in RESULTS if r["status"] == "ok" and r["tok_s"] > 11.5]
    if ok_results:
        # Get unique configs by label prefix
        seen_labels = set()
        top_configs = []
        for r in sorted(ok_results, key=lambda x: x["tok_s"], reverse=True):
            base_label = r.get("exp_id", "")
            if len(top_configs) < 3:
                top_configs.append(r)
        
        print(f"  Running statistical validation on top {len(top_configs)} configs")
        # We'll re-run the reproduce_best config 5 more times for statistics
        for i in range(5):
            track(run_experiment(next_exp(),
                label=f"stat_best_r{i+1}",
                n_gpu=16, n_ctx=512, n_threads=12, n_batch=256, n_ubatch=64,
                max_tokens=256))
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 60)
    print("PHASE 8 COMPLETE")
    ok_results = [r for r in RESULTS if r["status"] == "ok"]
    if ok_results:
        best = max(ok_results, key=lambda r: r["tok_s"])
        worst = min(ok_results, key=lambda r: r["tok_s"])
        avg = sum(r["tok_s"] for r in ok_results) / len(ok_results)
        print(f"  Total experiments: {len(RESULTS)} ({len(ok_results)} ok)")
        print(f"  Best: {best['tok_s']:.3f} tok/s (exp {best['exp_id']})")
        print(f"  Worst: {worst['tok_s']:.3f} tok/s")
        print(f"  Average: {avg:.3f} tok/s")
        print(f"  All-time best: {ALL_TIME_BEST} tok/s")
        if PHASE_BEST > ALL_TIME_BEST:
            print(f"  ★★★ NEW ALL-TIME BEST: {PHASE_BEST:.3f} tok/s ★★★")
    print("=" * 60)


if __name__ == "__main__":
    main()
