"""
driver5.py — Re-run failed exps + final fine-tuning.
"""
import json, subprocess, os
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
    print(f"\nEXP {exp_num}: ngl={ngl}, ctx={ctx}, batch={batch}/{ubatch}, K={type_k}, V={type_v} | {desc}")
    cmd = ['uv', 'run', '--with', 'llama-cpp-python', 'python', 'run_one.py',
           str(exp_num), str(ngl), str(ctx), str(batch), str(ubatch), type_k, type_v, desc]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=ENV,
                              cwd='/tmp/qwen35-moe-offload')
        if proc.returncode == 0:
            lines = [l for l in proc.stdout.strip().split('\n') if l.startswith('{')]
            if lines:
                r = json.loads(lines[-1])
                print(f"  ✅ {r['tps']:.3f} tok/s | VRAM {r.get('vram_peak',0):.0f}MB")
                return r
        stderr_tail = proc.stderr[-300:] if proc.stderr else ''
        print(f"  ❌ Exit {proc.returncode}: {stderr_tail[-100:]}")
        return {'exp': exp_num, 'tps': 0, 'status': 'error', 'ngl': ngl, 'ctx': ctx, 'batch': batch,
                'type_k': type_k, 'type_v': type_v, 'desc': desc}
    except subprocess.TimeoutExpired:
        return {'exp': exp_num, 'tps': 0, 'status': 'timeout', 'ngl': ngl, 'ctx': ctx, 'batch': batch,
                'type_k': type_k, 'type_v': type_v, 'desc': desc}

def main():
    # Ensure ollama is unloaded
    subprocess.run(['curl', '-s', 'http://localhost:11434/api/generate', '-d', '{"model":"qwen3.5:4b","keep_alive":0}'],
                   capture_output=True, timeout=10)
    import time; time.sleep(2)
    
    existing = load_log()
    exp = max((r['exp'] for r in existing), default=14) + 1
    best_tps = max((r.get('tps', 0) for r in existing if r.get('status') == 'ok'), default=10.2)

    experiments = [
        # Re-run failed experiments (were OOM due to ollama)
        (16, 2048, 16, 16, 'q4_0', 'q4_0', 'RERUN ngl=16 q4_0 batch=16'),
        (16, 2048, 64, 32, 'q4_0', 'q4_0', 'RERUN ngl=16 q4_0 batch=64'),
        (15, 2048, 32, 32, 'q4_0', 'q4_0', 'RERUN ngl=15 q4_0 ctx=2048'),
        (16, 2048, 32, 32, 'q4_0', 'q8_0', 'RERUN ngl=16 asym K=q4_0,V=q8_0'),
        # Try ngl=16 q4_0 ctx=8192 (might work now with free VRAM)  
        (16, 8192, 32, 32, 'q4_0', 'q4_0', 'RERUN ngl=16 q4_0 ctx=8192'),
        # Verify the best: re-run exp54 config  
        (16, 2048, 32, 32, 'q4_0', 'q4_0', 'VERIFY best: ngl=16 q4_0 ctx=2048'),
    ]
    
    for ngl, ctx, batch, ubatch, tk, tv, desc in experiments:
        r = run_exp(exp, ngl, ctx, batch, ubatch, tk, tv, desc)
        log_result(r)
        if r.get('status') == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        exp += 1
    
    # Print final leaderboard
    all_r = load_log()
    print(f"\n{'='*60}")
    print(f"FINAL LEADERBOARD — {len(all_r)} total experiments")
    print(f"Best: {best_tps:.3f} tok/s")
    print(f"{'='*60}")
    print(f"{'Exp':>4} {'TPS':>8} {'VRAM':>6} {'NGL':>4} {'CTX':>5} {'Batch':>5} {'K':>6} {'V':>6} {'Description'}")
    print('-' * 85)
    for r in sorted([x for x in all_r if x.get('status')=='ok'], key=lambda x: x['tps'], reverse=True)[:20]:
        print(f"{r['exp']:4d} {r['tps']:8.3f} {r.get('vram_peak',0):6.0f} {r['ngl']:4d} {r['ctx']:5d} {r.get('batch',32):5d} {r.get('type_k','?'):>6} {r.get('type_v','?'):>6} {r.get('desc','')}")

if __name__ == "__main__":
    main()
