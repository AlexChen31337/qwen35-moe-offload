#!/bin/bash
# Phase 6 final — stability test + aggressive exploration
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd /tmp/qwen35-moe-offload

BENCH=./target/release/bench
MODEL=/tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
EXP=78

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

# Stability: run the "winner" config 5x to get mean and variance
for i in 1 2 3 4 5; do
    run_exp "stability_t8_b256_64_$i" 16 512 256 64 8 1 8
done

# Stability: secondary winner 5x
for i in 1 2 3 4 5; do
    run_exp "stability_t12_b128_64_$i" 16 512 128 64 8 1 12
done

# n_threads=9 (between 8 and 12 sweet spot)
run_exp "t9_b256_64"    16 512 256 64 8 1 9
run_exp "t9_b128_64"    16 512 128 64 8 1 9

# Mixed KV types (type_k != type_v)
# Need to modify the binary for this — skip for now

echo "=== FINAL SWEEP COMPLETE ==="
