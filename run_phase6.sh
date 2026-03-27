#!/bin/bash
# Phase 6 Rust benchmark runner wrapper
# Sets up library paths for llama.cpp CUDA libraries

LLAMA_LIB=/tmp/qwen35-moe-offload/.venv/lib/python3.11/site-packages/llama_cpp/lib
CUDA_LIB=/usr/local/lib/ollama/cuda_v12
export LD_LIBRARY_PATH="$LLAMA_LIB:$CUDA_LIB:$LD_LIBRARY_PATH"

exec /tmp/qwen35-moe-offload/target/release/bench "$@"
