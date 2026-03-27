#!/bin/bash
# Phase 6 Rust bench sweep
# Usage: ./run_phase6_sweep.sh
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd /tmp/qwen35-moe-offload

BENCH=./target/release/bench
MODEL=/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
EXP=31

run_exp() {
    local label=$1 ngpu=$2 nctx=$3 nbatch=$4 nubatch=$5 typek=$6 flash=$7 threads=$8
    echo ">>> EXP $EXP: $label (n_gpu=$ngpu n_ctx=$nctx batch=$nbatch/$nubatch kv=$typek flash=$flash threads=$threads)"
    LD_LIBRARY_PATH=$CUDA_LIB $BENCH \
        --model "$MODEL" --n-gpu "$ngpu" --n-ctx "$nctx" \
        --n-batch "$nbatch" --n-ubatch "$nubatch" \
        --type-k "$typek" --type-v "$typek" \
        --flash-attn "$flash" --n-gen 128 --exp-id "$EXP" \
        --label "$label" --n-threads "$threads" 2>&1 | tail -5
    EXP=$((EXP + 1))
    echo ""
}

# Batch 1: Match Phase 5 best configs from Rust
# Phase 5 exp9 best: n_gpu=16, batch=64/128, q8_0, n_ctx=512 → 11.874 tok/s
# We already got 10.855 with these params via Rust.
# The difference is likely in n_ubatch (venv caps at 64).

# Try different n_threads to find optimal
run_exp "threads_1"     16 512 64 64 8 1 1
run_exp "threads_2"     16 512 64 64 8 1 2
run_exp "threads_4"     16 512 64 64 8 1 4
run_exp "threads_8"     16 512 64 64 8 1 8

# Try batch combos that were good in Phase 6 non-Rust
run_exp "batch_128_32"  16 512 128 32 8 1 4
run_exp "batch_128_64"  16 512 128 64 8 1 4

# GPU layer sweep (15 was good before)
run_exp "ngpu_15_b64"   15 512 64 64 8 1 4
run_exp "ngpu_15_b128"  15 512 128 32 8 1 4

# Context size vs performance
run_exp "ctx_256"       16 256 64 64 8 1 4
run_exp "ctx_8192"      16 8192 64 64 8 1 4

echo "=== SWEEP COMPLETE: $((EXP - 31)) experiments ==="
