"""
driver.py — Crash-safe experiment driver.
Runs each experiment as a subprocess so crashes don't kill the loop.
"""
import json, subprocess, sys, os
from pathlib import Path

LOG_FILE = "experiments_p4v2.jsonl"
ENV = {**os.environ, 'LD_LIBRARY_PATH': '/usr/local/lib/ollama/cuda_v12:' + os.environ.get('LD_LIBRARY_PATH', '')}

def load_log():
    results = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

def log_result(r):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(r) + '\n')

def run_exp(exp_num, ngl, ctx, batch, ubatch, type_k, type_v, desc):
    """Run one experiment via subprocess. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"EXP {exp_num}: ngl={ngl}, ctx={ctx}, batch={batch}, K={type_k}, V={type_v} | {desc}")
    
    cmd = ['uv', 'run', '--with', 'llama-cpp-python', 'python', 'run_one.py',
           str(exp_num), str(ngl), str(ctx), str(batch), str(ubatch), type_k, type_v, desc]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=ENV,
                              cwd='/tmp/qwen35-moe-offload')
        if proc.returncode == 0:
            # Parse JSON from last line of stdout
            lines = [l for l in proc.stdout.strip().split('\n') if l.startswith('{')]
            if lines:
                r = json.loads(lines[-1])
                print(f"  ✅ {r['tps']:.3f} tok/s | VRAM peak {r.get('vram_peak',0):.0f}MB")
                return r
        
        # Error
        stderr_tail = proc.stderr[-500:] if proc.stderr else ''
        print(f"  ❌ Exit {proc.returncode}: {stderr_tail[-200:]}")
        return {'exp': exp_num, 'tps': 0, 'status': 'error',
                'error': stderr_tail[-200:], 'ngl': ngl, 'ctx': ctx, 'batch': batch,
                'type_k': type_k, 'type_v': type_v, 'desc': desc}
    
    except subprocess.TimeoutExpired:
        print(f"  ❌ TIMEOUT (600s)")
        return {'exp': exp_num, 'tps': 0, 'status': 'timeout',
                'ngl': ngl, 'ctx': ctx, 'batch': batch,
                'type_k': type_k, 'type_v': type_v, 'desc': desc}

def main():
    existing = load_log()
    best_tps = 10.200
    best_kv = ('q8_0', 'q8_0')
    best_ngl = 10
    
    for r in existing:
        if r.get('status') == 'ok' and r.get('tps', 0) > best_tps:
            best_tps = r['tps']
            best_kv = (r['type_k'], r['type_v'])
    
    exp = max((r['exp'] for r in existing), default=14) + 1
    no_improve = 0
    
    def try_exp(ngl, ctx, batch, ubatch, tk, tv, desc):
        nonlocal exp, no_improve, best_tps, best_kv, best_ngl
        r = run_exp(exp, ngl, ctx, batch, ubatch, tk, tv, desc)
        log_result(r)
        if r.get('status') == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            best_kv = (tk, tv)
            best_ngl = ngl
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        elif r.get('status') == 'ok' and r['tps'] > 8.0 and ctx > 512:
            # Still usable at higher ctx - informative but not "improvement"
            print(f"  📊 Usable: {r['tps']:.3f} tok/s at ctx={ctx}")
            no_improve += 1
        else:
            no_improve += 1
        exp += 1
        return r
    
    # ========== AXIS 1: Symmetric KV quant ==========
    # Skip q5_k, q4_k, iq4_nl — they crash with SET_ROWS on CUDA
    # q5_1 and q5_0 already done (exps 15, 16)
    sym_types = ['q4_1', 'q4_0']  # remaining safe symmetric types
    for kv in sym_types:
        if no_improve >= 10: break
        try_exp(10, 512, 32, 32, kv, kv, f'symmetric KV={kv}')
    
    # ========== AXIS 2: Asymmetric KV (K=q8_0, V=lower) ==========
    asym_v = ['q5_1', 'q5_0', 'q4_1', 'q4_0']
    for v in asym_v:
        if no_improve >= 10: break
        try_exp(10, 512, 32, 32, 'q8_0', v, f'asymmetric K=q8_0, V={v}')
    
    # ========== AXIS 2b: Asymmetric KV (K=q5_0, V=q4_0) — both low ==========
    if no_improve < 10:
        try_exp(10, 512, 32, 32, 'q5_0', 'q4_0', 'asymmetric K=q5_0, V=q4_0')
    
    print(f"\n--- KV QUANT PHASE: Best={best_tps:.3f}, KV=({best_kv[0]},{best_kv[1]}) ---")
    
    # ========== AXIS 3: More GPU layers with best KV ==========
    for ngl in [12, 15, 20, 8]:
        if no_improve >= 10: break
        try_exp(ngl, 512, 32, 32, best_kv[0], best_kv[1], f'ngl={ngl} with best KV')
    
    # ========== AXIS 4: Context length with best KV & NGL ==========
    for ctx in [1024, 2048, 4096]:
        if no_improve >= 10: break
        try_exp(best_ngl, ctx, 32, 32, best_kv[0], best_kv[1], f'ctx={ctx}')
    
    # ========== AXIS 5: Batch tuning at ctx=2048 ==========
    for batch in [64, 128, 256, 512]:
        if no_improve >= 10: break
        try_exp(best_ngl, 2048, batch, min(batch, 64), best_kv[0], best_kv[1], f'ctx=2048 batch={batch}')
    
    # ========== AXIS 6: Combined — best NGL + lower KV + higher ctx ==========
    if no_improve < 10:
        try_exp(best_ngl, 1024, 64, 32, best_kv[0], best_kv[1], 'combined: best ngl + ctx=1024 + batch=64')
    if no_improve < 10:
        try_exp(best_ngl, 2048, 64, 32, best_kv[0], best_kv[1], 'combined: best ngl + ctx=2048 + batch=64')
    
    # Final summary
    all_r = load_log()
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total: {len(all_r)} experiments")
    print(f"Best: {best_tps:.3f} tok/s (KV: K={best_kv[0]}, V={best_kv[1]}, NGL={best_ngl})")
    print(f"{'='*60}")
    
    print(f"\n{'Exp':>4} {'TPS':>8} {'VRAM':>6} {'NGL':>4} {'CTX':>5} {'Batch':>5} {'K':>6} {'V':>6} {'Description'}")
    print('-' * 80)
    for r in sorted(all_r, key=lambda x: x.get('tps', 0), reverse=True):
        if r.get('status') == 'ok':
            print(f"{r['exp']:4d} {r['tps']:8.3f} {r.get('vram_peak',0):6.0f} {r['ngl']:4d} {r['ctx']:5d} {r.get('batch',32):5d} {r.get('type_k','?'):>6} {r.get('type_v','?'):>6} {r.get('desc','')}")
        else:
            print(f"{r['exp']:4d}    {'ERR':>5}                                          {r.get('desc','')} [{r.get('status','')}]")

if __name__ == "__main__":
    main()
