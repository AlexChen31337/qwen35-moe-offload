#!/bin/bash
# Phase 6 Rust bench sweep 2 — based on sweep 1 findings: threads=8 is best
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd /tmp/qwen35-moe-offload

BENCH=./target/release/bench
MODEL=/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
EXP=41

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

# All with threads=8 (best from sweep 1)

# Threads fine-tune
run_exp "threads_6"     16 512 64 64 8 1 6
run_exp "threads_10"    16 512 64 64 8 1 10
run_exp "threads_12"    16 512 64 64 8 1 12

# Batch combos with 8 threads
run_exp "t8_b128_32"    16 512 128 32 8 1 8
run_exp "t8_b128_64"    16 512 128 64 8 1 8
run_exp "t8_b64_128"    16 512 64 128 8 1 8
run_exp "t8_b32_32"     16 512 32 32 8 1 8
run_exp "t8_b256_64"    16 512 256 64 8 1 8

# GPU layers with 8 threads
run_exp "t8_ngpu15"     15 512 64 64 8 1 8
run_exp "t8_ngpu14"     14 512 64 64 8 1 8

# KV types with 8 threads 
run_exp "t8_f16"        16 512 64 64 1 1 8
run_exp "t8_q4_0"       16 512 64 64 2 1 8

# Context scaling with 8 threads
run_exp "t8_ctx1024"    16 1024 64 64 8 1 8
run_exp "t8_ctx2048"    16 2048 64 64 8 1 8
run_exp "t8_ctx4096"    16 4096 64 64 8 1 8

echo "=== SWEEP 2 COMPLETE: $((EXP - 41)) experiments ==="
