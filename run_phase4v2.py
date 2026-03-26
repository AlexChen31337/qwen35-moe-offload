"""
run_phase4v2.py — Systematic Phase 4v2 experiments.
Runs each experiment in-process, logs to experiments_p4v2.jsonl.
"""
import json, time, sys, subprocess, gc
from pathlib import Path

MODEL_PATH = "./models/Qwen3.5-35B-A3B-Q3_K_M.gguf"
LOG_FILE = "experiments_p4v2.jsonl"
PROMPT = "Explain the architecture of Mixture of Experts neural networks in detail:"
MAX_TOKENS = 256

# GGML type constants
T = {'f16': 1, 'q8_0': 8, 'q4_0': 2, 'q4_1': 3, 'q5_0': 6, 'q5_1': 7, 'iq4_nl': 20, 'q4_k': 12, 'q5_k': 13}
TNAME = {v: k for k, v in T.items()}

def vram_mb():
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip().split('\n')[0])
    except: return 0.0

def run_exp(cfg, exp_num):
    """Run one experiment. Returns result dict."""
    from llama_cpp import Llama
    
    tk, tv = TNAME.get(cfg['type_k'], '?'), TNAME.get(cfg['type_v'], '?')
    desc = cfg.get('desc', '')
    print(f"\n{'='*60}")
    print(f"EXP {exp_num}: ngl={cfg['ngl']}, ctx={cfg['ctx']}, batch={cfg['batch']}, "
          f"K={tk}, V={tv} | {desc}")
    
    v0 = vram_mb()
    try:
        kw = dict(model_path=MODEL_PATH, n_gpu_layers=cfg['ngl'], n_ctx=cfg['ctx'],
                  n_threads=10, n_batch=cfg['batch'], n_ubatch=cfg.get('ubatch', cfg['batch']),
                  verbose=False, type_k=cfg['type_k'], type_v=cfg['type_v'], flash_attn=True)
        
        t0 = time.perf_counter()
        llm = Llama(**kw)
        lt = time.perf_counter() - t0
        v1 = vram_mb()
        print(f"  Loaded {lt:.1f}s, VRAM {v1:.0f}MB (+{v1-v0:.0f})")
        
        # warmup
        _ = llm(PROMPT, max_tokens=10, echo=False)
        
        # bench 2 trials
        best = 0
        for _ in range(2):
            t = time.perf_counter()
            out = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
            e = time.perf_counter() - t
            nt = out["usage"]["completion_tokens"]
            tps = nt / e
            if tps > best:
                best, be, bn = tps, e, nt
        
        vp = vram_mb()
        r = {'exp': exp_num, 'tps': round(best, 3), 'tokens': bn, 'elapsed': round(be, 2),
             'vram_peak': vp, 'vram_model': round(vp - v0), 'load_s': round(lt, 1),
             'ngl': cfg['ngl'], 'ctx': cfg['ctx'], 'batch': cfg['batch'],
             'ubatch': cfg.get('ubatch', cfg['batch']),
             'type_k': tk, 'type_v': tv, 'desc': desc, 'status': 'ok'}
        print(f"  ✅ {best:.3f} tok/s | VRAM peak {vp:.0f}MB")
        
    except Exception as ex:
        r = {'exp': exp_num, 'tps': 0, 'status': 'error', 'error': str(ex)[:200],
             'ngl': cfg['ngl'], 'ctx': cfg['ctx'], 'batch': cfg['batch'],
             'type_k': tk, 'type_v': tv, 'desc': desc}
        print(f"  ❌ {ex}")
    
    try: del llm
    except: pass
    gc.collect()
    time.sleep(2)
    return r

def log(r):
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

def main():
    existing = load_log()
    done_exps = {r['exp'] for r in existing}
    
    # Track best
    best_tps = 10.200  # exp 14 baseline
    best_kv = (T['q8_0'], T['q8_0'])  # (type_k, type_v) 
    
    # Check if we have improvements from existing runs
    for r in existing:
        if r.get('status') == 'ok' and r.get('tps', 0) > best_tps:
            best_tps = r['tps']
            best_kv = (T.get(r['type_k'], T['q8_0']), T.get(r['type_v'], T['q8_0']))
    
    exp = max((r['exp'] for r in existing), default=14) + 1
    no_improve = 0  # consecutive non-improvements
    
    # ========== AXIS 1: Symmetric KV quant ==========
    sym_types = ['q5_1', 'q5_0', 'q5_k', 'q4_1', 'q4_0', 'q4_k', 'iq4_nl']
    for kv in sym_types:
        if no_improve >= 10:
            break
        cfg = {'ngl': 10, 'ctx': 512, 'batch': 32, 'ubatch': 32,
               'type_k': T[kv], 'type_v': T[kv], 'desc': f'symmetric KV={kv}'}
        r = run_exp(cfg, exp)
        log(r)
        if r['status'] == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            best_kv = (T[kv], T[kv])
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    # ========== AXIS 2: Asymmetric KV (K=q8_0, V=lower) ==========
    asym_v = ['q5_1', 'q5_0', 'q4_1', 'q4_0']
    for v in asym_v:
        if no_improve >= 10:
            break
        cfg = {'ngl': 10, 'ctx': 512, 'batch': 32, 'ubatch': 32,
               'type_k': T['q8_0'], 'type_v': T[v], 'desc': f'asymmetric K=q8_0, V={v}'}
        r = run_exp(cfg, exp)
        log(r)
        if r['status'] == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            best_kv = (T['q8_0'], T[v])
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    print(f"\n{'='*60}")
    print(f"KV QUANT PHASE DONE — Best: {best_tps:.3f} tok/s")
    print(f"Best KV: K={TNAME[best_kv[0]]}, V={TNAME[best_kv[1]]}")
    
    # ========== AXIS 3: More GPU layers with best KV ==========
    for ngl in [12, 15, 20]:
        if no_improve >= 10:
            break
        cfg = {'ngl': ngl, 'ctx': 512, 'batch': 32, 'ubatch': 32,
               'type_k': best_kv[0], 'type_v': best_kv[1],
               'desc': f'ngl={ngl} with best KV'}
        r = run_exp(cfg, exp)
        log(r)
        if r['status'] == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    # ========== AXIS 4: Context length with best KV ==========
    for ctx in [1024, 2048, 4096]:
        if no_improve >= 10:
            break
        cfg = {'ngl': 10, 'ctx': ctx, 'batch': 32, 'ubatch': 32,
               'type_k': best_kv[0], 'type_v': best_kv[1],
               'desc': f'ctx={ctx} with best KV'}
        r = run_exp(cfg, exp)
        log(r)
        # For ctx extension, "improvement" = usable at higher ctx even if tps drops slightly
        if r['status'] == 'ok':
            if r['tps'] > best_tps:
                best_tps = r['tps']
                no_improve = 0
                print(f"  🏆 NEW BEST: {best_tps:.3f}")
            elif r['tps'] > 8.0:  # still usable — don't count as fail
                print(f"  📊 Usable at ctx={ctx}: {r['tps']:.3f} tok/s")
                # Don't increment no_improve — this is valuable data
            else:
                no_improve += 1
        else:
            no_improve += 1
        exp += 1
    
    # ========== AXIS 5: Batch tuning at higher ctx ==========
    for batch in [64, 128, 256]:
        if no_improve >= 10:
            break
        cfg = {'ngl': 10, 'ctx': 2048, 'batch': batch, 'ubatch': min(batch, 64),
               'type_k': best_kv[0], 'type_v': best_kv[1],
               'desc': f'ctx=2048, batch={batch}'}
        r = run_exp(cfg, exp)
        log(r)
        if r['status'] == 'ok' and r['tps'] > best_tps:
            best_tps = r['tps']
            no_improve = 0
            print(f"  🏆 NEW BEST: {best_tps:.3f}")
        else:
            no_improve += 1
        exp += 1
    
    # Final summary
    all_results = load_log()
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total experiments: {len(all_results)}")
    print(f"Best: {best_tps:.3f} tok/s")
    print(f"Best KV: K={TNAME[best_kv[0]]}, V={TNAME[best_kv[1]]}")
    print(f"{'='*60}")
    
    # Print table
    print(f"\n{'Exp':>4} {'TPS':>8} {'VRAM':>6} {'NGL':>4} {'CTX':>5} {'K':>6} {'V':>6} {'Description'}")
    print('-' * 70)
    for r in sorted(all_results, key=lambda x: x.get('tps', 0), reverse=True):
        if r.get('status') == 'ok':
            print(f"{r['exp']:4d} {r['tps']:8.3f} {r.get('vram_peak',0):6.0f} {r['ngl']:4d} {r['ctx']:5d} {r['type_k']:>6} {r['type_v']:>6} {r.get('desc','')}")
        else:
            print(f"{r['exp']:4d}    ERROR                                    {r.get('desc','')} — {r.get('error','')[:40]}")


if __name__ == "__main__":
    main()
