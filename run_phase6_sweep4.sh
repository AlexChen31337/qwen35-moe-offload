#!/bin/bash
# Phase 6 sweep 4 — fine-tuning around the 11.845 winner
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd /tmp/qwen35-moe-offload

BENCH=./target/release/bench
MODEL=/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
EXP=66

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

# Fine-tune threads around 12
run_exp "t11_b128_64"   16 512 128 64 8 1 11
run_exp "t14_b128_64"   16 512 128 64 8 1 14
run_exp "t16_b128_64"   16 512 128 64 8 1 16

# Repeat the winner 3x to confirm consistency
run_exp "t12_b128_64_r1" 16 512 128 64 8 1 12
run_exp "t12_b128_64_r2" 16 512 128 64 8 1 12
run_exp "t12_b128_64_r3" 16 512 128 64 8 1 12

# Try 12 threads with various batch combos
run_exp "t12_b256_64"   16 512 256 64 8 1 12
run_exp "t12_b192_64"   16 512 192 64 8 1 12
run_exp "t12_b64_64"    16 512 64 64 8 1 12
run_exp "t12_b128_32"   16 512 128 32 8 1 12
run_exp "t12_b128_128"  16 512 128 128 8 1 12

# 12 threads with f16 KV
run_exp "t12_f16_b128_64" 16 512 128 64 1 1 12

echo "=== SWEEP 4 COMPLETE ==="
