# autoresearch — MoE Flash Offload

This is an autonomous inference optimization experiment on Qwen3.5-35B-A3B running on an RTX 3070 8GB + 16GB RAM + NVMe SSD.

Inspired directly by karpathy/autoresearch: one file to modify, fixed time budget, single metric, loop forever.

## Setup

To start a new experiment run:

1. **Agree on a run tag** — propose based on today's date (e.g. `mar25`). Branch `autoresearch/<tag>` must not exist yet.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read in-scope files**:
   - `README.md` — repo context and hardware specs
   - `harness.py` — fixed: NVMe benchmark, model loading, token generation, metric collection. **Do not modify.**
   - `bench.py` — the ONLY file you edit. Expert cache params, loader strategy, prefetch config, quantization level.
4. **Verify model exists**: Check that `./models/` contains a Qwen3.5-35B-A3B GGUF. If not, run: `uv run python scripts/download_model.py`
5. **Initialize results.tsv**: Create with header row only. Baseline recorded after first run.
6. **Confirm and go.**

## Experimentation

Each experiment runs the model for a **fixed 5-minute generation budget** (wall clock, excludes model load time).

**What you CAN modify in `bench.py`:**
- Expert cache window size (`k`)
- Max cached experts per layer
- Prefetch thread count and depth
- Chunk bundling strategy (gate+down vs gate+up+down vs separate)
- NVMe read block size alignment
- DRAM vs VRAM cache split ratio
- Quantization level (Q4_K_M vs Q5_K_M vs Q8_0)
- Expert predictor threshold (confidence cutoff for prefetch)
- Async prefetch lookahead depth

**What you CANNOT modify:**
- `harness.py` — the fixed benchmark harness
- The prompt (fixed 512-token C4 sample, same seed always)
- Generation length (always 256 tokens)
- Hardware (RTX 3070 8GB, 16GB RAM, NVMe)

**The goal: maximize tok/s.** Secondary: minimize peak VRAM and peak RAM.

**Simplicity criterion** (same as karpathy/autoresearch): A small improvement that adds ugly complexity isn't worth it. Removing something and getting equal results is a win.

## Output Format

After each run, `harness.py` prints:

```
---
tok_per_sec:     4.82
total_seconds:   300.1
peak_vram_mb:    7241.3
peak_ram_mb:     9830.2
nvme_bytes_per_tok: 28311040
cache_hit_rate:  0.712
num_tokens:      256
```

Extract key metric: `grep "^tok_per_sec:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated) — do NOT commit this file:

```
commit	tok_per_sec	vram_gb	ram_gb	cache_hit	status	description
```

- commit: 7-char short hash
- tok_per_sec: e.g. 4.820 — use 0.000 for crashes
- vram_gb: peak VRAM in GB (e.g. 7.1)
- ram_gb: peak RAM in GB (e.g. 9.6)
- cache_hit: hit rate 0.000–1.000 — use 0.000 for crashes
- status: `keep`, `discard`, or `crash`
- description: short text (no tabs!)

## The Experiment Loop

```
LOOP FOREVER:

1. git status — check current branch/commit
2. Edit bench.py with one experimental idea
3. git commit -m "experiment: <short description>"
4. uv run python bench.py > run.log 2>&1
5. grep "^tok_per_sec:\|^peak_vram_mb:\|^peak_ram_mb:\|^cache_hit_rate:" run.log
6. If empty → crashed. Run: tail -n 50 run.log
   - Dumb bug (typo/import) → fix and re-run
   - Fundamentally broken → log crash, git reset, move on
7. Log to results.tsv
8. If tok_per_sec IMPROVED → keep commit (advance branch)
9. If same or worse → git reset --hard HEAD~1 (revert)
```

**NEVER STOP**: Do not pause to ask if you should continue. The human may be asleep. Run until manually interrupted. If out of ideas, try:
- Combining two previous near-misses
- More radical changes (double the cache, halve the prefetch threads, change quantization)
- Read `docs/apple-flash-mapping.md` for new angles
- Try the opposite of what failed last time

**Each experiment: ~5 min generation + ~2 min setup = ~7 min/experiment**
**Overnight (8h): ~68 experiments possible**

## Experiment Ideas (Seed List)

Start with these, then improvise:

1. Baseline (no cache, naive per-token load)
2. Cache k=3, no prefetch
3. Cache k=5, no prefetch
4. Cache k=10, no prefetch
5. Cache k=5, prefetch 4 threads
6. Cache k=5, prefetch 8 threads
7. Bundle gate+down (2x chunk size)
8. Bundle gate+up+down (3x chunk size)
9. Align reads to 32KB boundary
10. Align reads to 128KB boundary
11. Q5_K_M vs Q4_K_M (accuracy vs speed tradeoff)
12. Split cache: hot experts in VRAM, cold in DRAM
13. Predictor threshold 0.3 (aggressive prefetch)
14. Predictor threshold 0.7 (conservative prefetch)
15. Disable predictor entirely — pure LRU cache
16. Increase DRAM cache to 200 experts max
17. Prefetch lookahead depth 2 (predict 2 tokens ahead)
18. Combine k=7 + 8 threads + 128KB alignment + bundle gate+down

---

## Phase 4 — KV Cache Compression (Active Goal)

**Current best:** 6.587 tok/s (exp 15, n_ubatch=32)  
**Phase 3 conclusion:** Parameter tuning exhausted — llama.cpp knobs maxed out at ~6.6 tok/s.

**New axis: KV cache compression.** The KV cache is the second VRAM bottleneck after expert weights. Compressing it unlocks longer context AND potentially faster throughput.

**Direction:**
1. Start with llama.cpp built-in KV quant flags — `--cache-type-k` / `--cache-type-v` (q8_0, q4_0, q4_1) — measure tok/s and VRAM at n_ctx=512, then push n_ctx up (1024, 2048, 4096) if VRAM allows
2. Find the optimal KV quant level where accuracy holds but VRAM drops meaningfully
3. Inspired by TurboQuant (arXiv:2504.19874, ICLR 2026): zero-overhead KV compression via PolarQuant rotation + 1-bit QJL residual — explore whether a Python wrapper around llama.cpp can inject this

**Rule:** No ceiling. No target. Data drives everything. Push until the hardware physically stops you — OOM, thermal throttle, or the metric stops moving after exhausting all axes. The hardware sets the limit, not us.

If KV quant degrades quality measurably (outputs become incoherent), back off and find the right quant level — don't stop, recalibrate.

When one axis is exhausted, stack the next: KV quant → GPU layers → context length → PolarQuant native implementation → async expert prefetch. Keep going.
