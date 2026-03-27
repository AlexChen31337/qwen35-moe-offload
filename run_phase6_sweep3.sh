#!/bin/bash
# Phase 6 sweep 3 — zooming in on batch=256/64 with t=8 sweet spot
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd /tmp/qwen35-moe-offload

BENCH=./target/release/bench
MODEL=/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
EXP=56

run_exp() {
    local label=$1 ngpu=$2 nctx=$3 nbatch=$4 nubatch=$5 typek=$6 flash=$7 threads=$8
    echo ">>> EXP $EXP: $label"
    LD_LIBRARY_PATH=$CUDA_LIB $BENCH \
        --model "$MODEL" --n-gpu "$ngpu" --n-ctx "$nctx" \
        --n-batch "$nbatch" --n-ubatch "$nubatch" \
        --type-k "$typek" --type-v "$typek" \
        --flash-attn "$flash" --n-gen 128 --exp-id "$EXP" \
        --label "$label" --n-threads "$threads" 2>&1 | tail -5
    EXP=$((EXP + 1))
    echo ""
}

# Batch size exploration around the winner (256/64 was 11.512)
run_exp "b512_64"       16 512 512 64 8 1 8
run_exp "b256_128"      16 512 256 128 8 1 8
run_exp "b256_32"       16 512 256 32 8 1 8
run_exp "b192_64"       16 512 192 64 8 1 8
run_exp "b384_64"       16 512 384 64 8 1 8

# f16 KV was 11.317 — combine with large batch
run_exp "b256_64_f16"   16 512 256 64 1 1 8
run_exp "b128_64_f16"   16 512 128 64 1 1 8

# Try 12 threads (also competitive)
run_exp "t12_b256_64"   16 512 256 64 8 1 12
run_exp "t12_b128_64"   16 512 128 64 8 1 12

# Best config, 256 gen tokens to confirm
run_exp "b256_64_gen256" 16 512 256 64 8 1 8

echo "=== SWEEP 3 COMPLETE ==="
