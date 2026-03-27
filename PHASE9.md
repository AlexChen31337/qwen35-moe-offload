# Phase 9 — Axes Beyond Ubatch/Threads

## Goal
Beat 11.758 tok/s (Phase 9 best so far: batch=256/96, threads=12, q8_0)

## Setup
```bash
cd /tmp/qwen35-moe-offload
export LD_LIBRARY_PATH="/tmp/llama-cpp-build:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH"
./target/release/bench --help
```

## Baseline
- 11.758 tok/s (batch=256/96, threads=12, q8_0, n_ctx=512, gen=256)
- All-time best: 12.114 tok/s (Phase 6, Python internal timing)

## New Axes to Try
1. Different gen sizes: 192, 224, 288, 320
2. Different batch sizes: 240, 272, 288
3. Different KV types: iq4_nl (sweet spot for 32K contexts)
4. Different n_gpu values: 12, 14, 18

## Run Experiments
```bash
./target/release/bench \
  --model models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  --n-gpu 16 --n-ctx 512 --n-batch 256 --n-ubatch 96 \
  --type-k 8 --type-v 8 \
  --flash-attn 1 --n-threads 12 --n-gen 256 \
  --label "phase9_test" --results-file results_phase9.tsv \
  --exp-id N
```

## Commit Pattern
Every 5 experiments:
```bash
git add results_phase9.tsv
git commit -m "phase9 exp N-M: best=X tok/s"
git push
```

## Report
Every 5 exps: sessions_send(sessionKey="agent:main:main", message="[Phase9] exp N, best=X tok/s")
New best: sessions_send(sessionKey="agent:main:main", message="[Phase9] NEW BEST: X tok/s")

Never stop. Hardware sets the limit.
