"""
run_phase5.py — Phase 5: Run experiments from a plan file.
Reads experiments.json, skips completed ones, runs the rest.
Robust against crashes — just restart.
"""
import json, os, subprocess, sys, time
from pathlib import Path

TSV_PATH = "./results_phase5.tsv"
PLAN_PATH = "./phase5_plan.json"
TYPE_NAMES = {1: 'f16', 8: 'q8_0', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 20: 'iq4_nl', 12: 'q4_k', 13: 'q5_k'}


def ensure_vram_free(max_wait=45):
    """Wait until VRAM is back to baseline (< 2000 MB). Unload ollama models."""
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


def run_exp(cfg, timeout=300):
    """Run single experiment in subprocess."""
    ensure_vram_free()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        sys.executable, "run_single_exp_v2.py",
        "--n_gpu", str(cfg["n_gpu"]),
        "--n_ctx", str(cfg["n_ctx"]),
        "--n_threads", str(cfg.get("n_threads", 10)),
        "--n_batch", str(cfg.get("n_batch", 64)),
        "--n_ubatch", str(cfg.get("n_ubatch", 64)),
        "--type_k", str(cfg.get("type_k", 8)),
        "--type_v", str(cfg.get("type_v", 8)),
        "--max_tokens", str(cfg.get("max_tokens", 256)),
    ]
    if not cfg.get("flash", True):
        cmd.append("--no_flash_attn")
    if cfg.get("rope_freq_base", 0) > 0:
        cmd.extend(["--rope_freq_base", str(cfg["rope_freq_base"])])
    if cfg.get("long_prompt", False):
        cmd.append("--long_prompt")
    if cfg.get("op_offload", False):
        cmd.append("--op_offload")
    if cfg.get("swa_full", False):
        cmd.append("--swa_full")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/tmp/qwen35-moe-offload", env=env
        )
        time.sleep(5)  # Let CUDA release memory

        for line in proc.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        return {"error": "NO_JSON", "detail": f"rc={proc.returncode}, stderr={stderr_tail[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "detail": f">{timeout}s"}
    except Exception as e:
        return {"error": "LAUNCH_FAIL", "detail": str(e)[:200]}


def get_completed_exps():
    """Get set of completed experiment numbers."""
    completed = set()
    if not Path(TSV_PATH).exists():
        return completed
    with open(TSV_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0].isdigit():
                completed.add(int(parts[0]))
    return completed


def init_tsv():
    if not Path(TSV_PATH).exists():
        with open(TSV_PATH, "w") as f:
            f.write("exp\ttok_per_sec\tvram_peak_mb\tn_ctx\ttype_k\ttype_v\tflash_attn\tn_gpu_layers\tn_batch\tn_ubatch\tnotes\tstatus\tdescription\n")


def append_tsv(exp, tok, vram, n_ctx, tk, tv, flash, n_gpu, n_batch, n_ubatch, notes, status, desc):
    line = f"{exp}\t{tok:.3f}\t{vram:.0f}\t{n_ctx}\t{tk}\t{tv}\t{flash}\t{n_gpu}\t{n_batch}\t{n_ubatch}\t{notes}\t{status}\t{desc}\n"
    with open(TSV_PATH, "a") as f:
        f.write(line)
    print(f"  [{exp}] {tok:.3f} tok/s [{status}] {desc}", flush=True)


def main():
    init_tsv()
    plan = json.loads(Path(PLAN_PATH).read_text())
    completed = get_completed_exps()
    best_tok = 11.850  # Phase 4 best

    # Check if we already have a higher best from previous runs
    if Path(TSV_PATH).exists():
        with open(TSV_PATH) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1 and parts[0].isdigit():
                    try:
                        tok = float(parts[1])
                        if tok > best_tok:
                            best_tok = tok
                    except ValueError:
                        pass

    total = len(plan)
    pending = [e for e in plan if e["exp"] not in completed]
    print(f"Phase 5: {total} planned, {len(completed)} done, {len(pending)} remaining")
    print(f"Current best: {best_tok:.3f} tok/s")
    print("=" * 60)

    for i, cfg in enumerate(pending):
        exp = cfg["exp"]
        tk_name = TYPE_NAMES.get(cfg.get("type_k", 8), str(cfg.get("type_k", 8)))
        tv_name = TYPE_NAMES.get(cfg.get("type_v", 8), str(cfg.get("type_v", 8)))
        desc_base = cfg.get("desc", f"n_gpu={cfg['n_gpu']}, n_ctx={cfg['n_ctx']}")

        print(f"\n[{i+1}/{len(pending)}] Exp {exp}: {desc_base}", flush=True)

        r = run_exp(cfg, timeout=cfg.get("timeout", 300))
        notes = cfg.get("notes", "")

        if "error" not in r:
            tok = r["tok_per_sec"]
            vram = r["vram_peak_mb"]
            status = "keep" if tok > best_tok else "discard"
            desc = desc_base
            if tok > best_tok:
                best_tok = tok
                desc += " — NEW BEST"
                status = "keep"
                print(f"  *** NEW BEST: {tok:.3f} tok/s ***", flush=True)
            append_tsv(exp, tok, vram, cfg["n_ctx"], tk_name, tv_name,
                       cfg.get("flash", True), cfg["n_gpu"],
                       cfg.get("n_batch", 64), cfg.get("n_ubatch", 64),
                       notes, status, desc)
        else:
            append_tsv(exp, 0, 0, cfg["n_ctx"], tk_name, tv_name,
                       cfg.get("flash", True), cfg["n_gpu"],
                       cfg.get("n_batch", 64), cfg.get("n_ubatch", 64),
                       f"{notes}|{r['error']}", r["error"],
                       f"{desc_base} — {r['error']}: {r.get('detail','')[:80]}")

    print(f"\n{'='*60}")
    print(f"Phase 5 COMPLETE — Best: {best_tok:.3f} tok/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
