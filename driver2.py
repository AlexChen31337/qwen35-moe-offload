"""
driver2.py — Phase 4v2 follow-up experiments.
Fine-tune around the best config (ngl=15, ctx=2048, q4_1 KV).
"""
import json, subprocess, sys, os
from pathlib import Path

LOG_FILE = "experiments_p4v2.jsonl"
ENV = {**os.environ, 'LD_LIBRARY_PATH': '/usr/local/lib/ollama/cuda_v12:' + os.environ.get('LD_LIBRARY_PATH', '')}

def log_result(r):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(r) + '\n')

def load_log():
    results = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

def run_exp(exp_num, ngl, ctx, batch, ubatch, type_k, type_v, desc):
    print(f"\n{'='*60}")
    print(f"EXP {exp_num}: ngl={ngl}, ctx={ctx}, batch={batch}, K={type_k}, V={type_v} | {desc}")
    
    cmd = ['uv', 'run', '--with', 'llama-cpp-python', 'python', 'run_one.py',
           str(exp_num), str(ngl), str(ctx), str(batch), str(ubatch), type_k, type_v, desc]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=ENV,
                              cwd='/tmp/qwen35-moe-offload')
        if proc.returncode == 0:
            lines = [l for l in proc.stdout.strip().split('\n') if l.startswith('{')]
            if lines:
                r = json.loads(lines[-1])
                print(f"  ✅ {r['tps']:.3f} tok/s | VRAM peak {r.get('vram_peak',0):.0f}MB")
                return r
        stderr_tail = proc.stderr[-300:] if proc.stderr else ''
        print(f"  ❌ Exit {proc.returncode}: {stderr_tail[-150:]}")
        return {'exp': exp_num, 'tps': 0, 'status': 'error', 'error': stderr_tail[-150:],
                'ngl': ngl, 'ctx': ctx, 'batch': batch, 'type_k': type_k, 'type_v': type_v, 'desc': desc}
    except subprocess.TimeoutExpired:
        print(f"  ❌ TIMEOUT")
        return {'exp': exp_num, 'tps': 0, 'status': 'timeout',
                'ngl': ngl, 'ctx': ctx, 'batch': batch, 'type_k': type_k, 'type_v': type_v, 'desc': desc}

def main():
    existing = load_log()
    exp = max((r['exp'] for r in existing), default=14) + 1
    best_tps = max((r.get('tps', 0) for r in existing if r.get('status') == 'ok'), default=10.2)
    no_improve = 0
    
    # Fine-tune NGL around 15
    for ngl in [13, 14, 16, 17, 18]:
        r = run_exp(exp, ngl, 2048, 32, 32, 'q4_1', 'q4_1', f'fine-tune ngl={ngl}, ctx=2048')
        log_result(r)
        if r.get('status') == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    # Try ngl=15 with ctx=3072 (between 2048 and 4096)
    r = run_exp(exp, 15, 3072, 32, 32, 'q4_1', 'q4_1', 'ctx=3072')
    log_result(r)
    if r.get('status') == 'ok' and r['tps'] > best_tps:
        best_tps = r['tps']; no_improve = 0; print(f"  🏆 NEW BEST: {best_tps:.3f}")
    else:
        no_improve += 1
    exp += 1
    
    # Try ngl=15 with ubatch variations at ctx=2048
    for ub in [16, 64]:
        r = run_exp(exp, 15, 2048, 32, ub, 'q4_1', 'q4_1', f'ctx=2048 ubatch={ub}')
        log_result(r)
        if r.get('status') == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']; no_improve = 0; print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    # Try n_threads variations
    for nt_batch_combo in [(8, 32), (12, 32), (14, 32)]:
        # Need to modify run_one to accept threads — skip, use defaults
        pass
    
    # Try asymmetric K=q4_1, V=q4_0 at ngl=15 ctx=2048 (asymmetric at best GPU layers)
    r = run_exp(exp, 15, 2048, 32, 32, 'q4_1', 'q4_0', 'asymmetric K=q4_1,V=q4_0 ngl=15 ctx=2048')
    log_result(r)
    if r.get('status') == 'ok' and r['tps'] > best_tps:
        best_tps = r['tps']; no_improve = 0; print(f"  🏆 NEW BEST: {best_tps:.3f}")
    else:
        no_improve += 1
    exp += 1
    
    # Try K=q8_0, V=q4_1 at ngl=15 ctx=2048
    r = run_exp(exp, 15, 2048, 32, 32, 'q8_0', 'q4_1', 'asymmetric K=q8_0,V=q4_1 ngl=15 ctx=2048')
    log_result(r)
    if r.get('status') == 'ok' and r['tps'] > best_tps:
        best_tps = r['tps']; no_improve = 0; print(f"  🏆 NEW BEST: {best_tps:.3f}")
    else:
        no_improve += 1
    exp += 1

    # Print summary
    all_r = load_log()
    print(f"\n{'='*60}")
    print(f"FOLLOW-UP COMPLETE — Best: {best_tps:.3f} tok/s")
    print(f"Total experiments: {len(all_r)}")
    print(f"\nTop 10:")
    print(f"{'Exp':>4} {'TPS':>8} {'VRAM':>6} {'NGL':>4} {'CTX':>5} {'Batch':>5} {'K':>6} {'V':>6} {'Description'}")
    print('-' * 80)
    for r in sorted([x for x in all_r if x.get('status')=='ok'], key=lambda x: x['tps'], reverse=True)[:10]:
        print(f"{r['exp']:4d} {r['tps']:8.3f} {r.get('vram_peak',0):6.0f} {r['ngl']:4d} {r['ctx']:5d} {r.get('batch',32):5d} {r.get('type_k','?'):>6} {r.get('type_v','?'):>6} {r.get('desc','')}")

if __name__ == "__main__":
    main()
