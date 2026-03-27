#!/bin/bash
# Phase 6 experiment sweep — comprehensive coverage
# Uses Rust bench binary for zero Python overhead

set -e

BENCH="./target/release/bench"
export LD_LIBRARY_PATH="/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/llama_cpp/lib:/usr/local/lib/ollama/cuda_v12"

# Base best config: n_gpu=16, batch=128/64, q8_0, n_ctx=512, flash=1, n_threads=12

EXP=74

echo "=== Phase 6 Experiment Sweep ==="
echo "Starting from exp $EXP"

# --- Axis 1: n_threads fine-sweep around 12 (best=12) ---
echo "--- n_threads sweep ---"
for THREADS in 10 11 13 9 8; do
    echo "exp $EXP: n_threads=$THREADS"
    $BENCH --n-gpu 16 --n-ctx 512 --n-batch 128 --n-ubatch 64 --type-k 8 --type-v 8 \
           --flash-attn 1 --n-threads $THREADS --n-gen 128 --exp-id $EXP --label "t${THREADS}_b128_64" 2>&1 | tail -5
    EXP=$((EXP + 1))
    sleep 2
done

# --- Axis 2: n_ctx scaling at best config ---
echo "--- n_ctx scaling ---"
for CTX in 1024 2048 4096 8192; do
    echo "exp $EXP: n_ctx=$CTX"
    $BENCH --n-gpu 16 --n-ctx $CTX --n-batch 128 --n-ubatch 64 --type-k 8 --type-v 8 \
           --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "ctx${CTX}_t12" 2>&1 | tail -5
    EXP=$((EXP + 1))
    sleep 2
done

# --- Axis 3: n_ctx scaling with q4_0 KV (saves VRAM, enables larger ctx) ---
echo "--- n_ctx scaling with q4_0 KV ---"
for CTX in 4096 8192 16384 32768; do
    echo "exp $EXP: n_ctx=$CTX q4_0 KV"
    $BENCH --n-gpu 16 --n-ctx $CTX --n-batch 128 --n-ubatch 64 --type-k 2 --type-v 2 \
           --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "ctx${CTX}_q4kv_t12" 2>&1 | tail -5
    EXP=$((EXP + 1))
    sleep 2
done

# --- Axis 4: iq4_nl KV at various contexts ---
echo "--- iq4_nl KV sweep ---"
for CTX in 512 1024 4096 16384 32768; do
    echo "exp $EXP: n_ctx=$CTX iq4_nl KV"
    $BENCH --n-gpu 16 --n-ctx $CTX --n-batch 128 --n-ubatch 64 --type-k 20 --type-v 20 \
           --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "ctx${CTX}_iq4nl_t12" 2>&1 | tail -5
    EXP=$((EXP + 1))
    sleep 2
done

# --- Axis 5: batch size fine-tuning ---
echo "--- batch tuning ---"
for BATCH_UBATCH in "64,96" "96,64" "96,96" "64,128" "160,64" "192,64" "128,32" "128,128"; do
    IFS=',' read -r NB NUB <<< "$BATCH_UBATCH"
    echo "exp $EXP: batch=$NB/$NUB"
    $BENCH --n-gpu 16 --n-ctx 512 --n-batch $NB --n-ubatch $NUB --type-k 8 --type-v 8 \
           --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "b${NB}_${NUB}_t12" 2>&1 | tail -5
    EXP=$((EXP + 1))
    sleep 2
done

# --- Axis 6: n_gpu=17 (between 16=ok and 18=OOM) ---
echo "--- n_gpu=17 test ---"
echo "exp $EXP: n_gpu=17"
$BENCH --n-gpu 17 --n-ctx 512 --n-batch 128 --n-ubatch 64 --type-k 8 --type-v 8 \
       --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "ngpu17_t12" 2>&1 | tail -5
EXP=$((EXP + 1))

# --- Axis 7: flash_attn disabled comparison ---
echo "--- no flash attn ---"
echo "exp $EXP: no flash attn"
$BENCH --n-gpu 16 --n-ctx 512 --n-batch 128 --n-ubatch 64 --type-k 1 --type-v 1 \
       --flash-attn 0 --n-threads 12 --n-gen 128 --exp-id $EXP --label "noflash_f16_t12" 2>&1 | tail -5
EXP=$((EXP + 1))

# --- Axis 8: f16 KV with batch=256/64 (was 11.512 at batch=256/64 q8_0) ---
echo "--- f16 KV + 256/64 ---"
echo "exp $EXP: f16 + batch=256/64"
$BENCH --n-gpu 16 --n-ctx 512 --n-batch 256 --n-ubatch 64 --type-k 1 --type-v 1 \
       --flash-attn 1 --n-threads 12 --n-gen 128 --exp-id $EXP --label "f16_b256_64_t12" 2>&1 | tail -5
EXP=$((EXP + 1))

echo "=== Sweep complete: $(($EXP - 74)) experiments ==="
