# Autoresearch Goal — Qwen3.5-35B-A3B on RTX 3070 8GB

## Mission
Maximize tok/s for Qwen3.5-35B-A3B-Q3_K_M on RTX 3070 8GB. No ceiling. No predicted targets. No asking for direction. Hardware sets the limit.

## Current Best
- **11.874 tok/s** (Phase 5, exp 9: n_gpu=16, batch=64/128, q8_0, n_ctx=512)
- **11.850 tok/s** (Phase 4, exp 46: n_gpu=16, batch=64/64, q8_0, n_ctx=512)

## Hard Facts
- OOM ceiling: n_gpu=18 (both q8_0 and q4_0)
- n_ctx=8192+ segfaults with batch=64 on llama-cpp-python 0.3.19 → upgrade to fix
- n_ctx=16384 works at batch=32/32 (11.634 tok/s) but not at batch=64
- q4_0 KV unlocks n_ctx=2048 at 11.785 tok/s with n_gpu=16
- Python overhead: negligible (<1ms/call), inference is C++/CUDA

## Axes Not Yet Exhausted
1. **Post-upgrade n_ctx scaling** — after upgrading llama-cpp-python, retest n_ctx=8192/16384/32768 at batch=64/128
2. **PolarQuant end-to-end** — run bench_polarquant.py on real model, compare tok/s vs q8_0
3. **QJL end-to-end** — run bench_qjl.py on real model, compare tok/s vs PolarQuant
4. **Batch fine-tuning** — try batch=64/96, 64/160, 48/128, 48/160 around the 64/128 winner
5. **n_gpu=17** — between 16 (ok) and 18 (OOM), worth trying
6. **Rust benchmark runner** — crates/bench/ calling libllama.so via C FFI (Phase 6)

## Available Tools
- `uv run python bench_kv.py` — main benchmark (modify knobs at top of file)
- `uv run python orchestrate_phase5.py` — batch runner
- `uv run python run_single_exp_v2.py` — isolated subprocess runner (no CUDA state leak)
- `uv run python bench_polarquant.py` — PolarQuant compression benchmark
- `uv run python bench_qjl.py` — QJL compression benchmark
- `uv run python scripts/polar_kv.py` — PolarQuant math primitives
- `cargo test --workspace` — test Rust crates
- `nvidia-smi` — VRAM monitoring
- `results_phase5.tsv` — append results (TAB-separated)

## Reporting Protocol
- New best found → `sessions_send sessionKey="agent:main:main" message="[Phase5] NEW BEST: X tok/s (exp N) — config"`
- Every 10 experiments → `sessions_send sessionKey="agent:main:main" message="[Phase5] Progress: exp N, best=X tok/s"`
- Axis exhausted → push commit, move to next axis, keep going
- Never stop → if all axes done, go back and test combinations

## Git Protocol
- Branch: `phase5/native-compression`
- Push: `git push github-alexchen:AlexChen31337/qwen35-moe-offload.git phase5/native-compression`
- Commit every improvement: `git commit -m "phase5 exp N: X tok/s — description"`
- Commit every 10 experiments even if no improvement

## Done When
Never. Hardware physically stops you.

## Phase 7 Findings (2026-03-27)
- PolarQuant loses to simple uniform quant on Gaussian KV distributions
- QJL needs CUDA kernel to be viable (250-16000x slower on CPU)
- llama.cpp has NO API to extract/inject KV cache entries — integration wall
- All-time best: 12.114 tok/s (Phase 6, exp 117: n_gpu=16, batch=256/64, q8_0, n_threads=12, gen=256)
- Theoretical ceiling: 310K context at 12.09 tok/s with hybrid iq4_nl+uniform RAM offload

## Phase 8 Axes (unexplored — agent decides freely)
- **llama.cpp patch**: fork llama.cpp, add KV extract/inject API, integrate uniform quant offload
- **Different runtime**: vLLM, llama-cpp-python upgrade, mlx-lm, or candle (Rust ML framework)
- **Speculative decoding**: draft model + target model — can it beat 12.114 on this hardware?
- **Continuous batching**: serve multiple requests, measure aggregate tok/s
- **Generation length effect**: does tok/s improve at gen=512, 1024? (Phase 6 found gen=256 > gen=128)
- **MoE expert routing analysis**: which experts activate most? Can we pre-load hot experts to VRAM?
- **Quantization of weights (not KV)**: try Q2_K, Q4_K_M variants — different GGUF files
- **Flash attention variants**: try different flash_attn implementations
- Agent decides — no ceiling, no targets, hardware sets the limit

## PRIMARY BENCHMARK TOOL: Rust Binary (NOT Python)
The Rust FFI benchmark crate is built and verified. USE IT for all experiments.

```bash
cd /tmp/qwen35-moe-offload
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=/tmp/llama-cpp-build

# Build (only needed once)
cargo build -p bench --release 2>&1 | tail -3

# Run experiment
./target/release/bench \
  --model /tmp/qwen35-moe-offload/models/Qwen3.5-35B-A3B-Q3_K_M.gguf \
  --n-gpu 16 --n-ctx 512 --n-batch 256 --n-ubatch 64 \
  --kv-type q8_0 --flash-attn --n-threads 12 --gen 256

# Check available args
./target/release/bench --help
```

**Why Rust:** Calls libllama.so directly via C FFI. Zero Python overhead. Nanosecond-precision timing. libllama.so is at /tmp/llama-cpp-build/libllama.so (CUDA-enabled, links libcublas.so.13).

**Python fallback:** Only use bench_kv.py if Rust binary fails to build or crashes. Never use Python as primary measurement tool.
