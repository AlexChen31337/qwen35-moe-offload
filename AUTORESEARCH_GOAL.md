# Autoresearch Goal — Qwen3.5-35B-A3B on RTX 3070 8GB

## Mission
Maximize tok/s for Qwen3.5-35B-A3B-Q3_K_M on RTX 3070 8GB. No ceiling. No predicted targets. No asking for direction. Hardware sets the limit.

## NON-NEGOTIABLE: Autonomous Self-Direction (hardened 2026-03-28)

**The loop decides what to try next. Always. No exceptions.**

- ❌ NEVER wait for a pre-specified experiment matrix — that's not autoresearch, that's scripted benchmarking
- ❌ NEVER ask permission before trying a new axis
- ❌ NEVER stop because "the obvious axes are done" — invent new ones
- ❌ NEVER pre-announce what you're about to try and wait for approval
- ✅ ALWAYS decide the next experiment based on what you just observed
- ✅ ALWAYS treat unexpected results as the most interesting signal
- ✅ ALWAYS follow the data — if IQ2_M was 75% faster than Q3_K_M, ask WHY and explore the implication
- ✅ ALWAYS commit and keep going — results.tsv is your memory

**The loop is the researcher. You own the experimental design.**

If in doubt: run the experiment first, analyse after. Curiosity > caution.

## Current Best
- **29.899 tok/s** (Phase 12, exp 62: IQ2_XXS, n_gpu=27, batch=32/16, q8_0 KV, n_ctx=256, flash=1, op_offload=1, threads=8)
- **29.858 tok/s** (Phase 12, exp 65: same config, consistent replicate)
- **28.923 tok/s** (Phase 12, exp 27: same model, threads=11)
- **25.242 tok/s** (Phase 11, exp 69: IQ2_XXS, n_gpu=27, batch=32/16 — old ATB before Phase 12)
- **21.621 tok/s** (Phase 11, exp 29: IQ2_M, n_gpu=24 — old ATB for IQ2_M)
- **12.331 tok/s** (Phase 10: Q3_K_M, n_gpu=17 — best for original quant)

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

## Phase 12 Findings (2026-03-28)
- **NEW ALL-TIME BEST: 29.899 tok/s** (IQ2_XXS, n_gpu=27, batch=32/16, q8_0 KV, threads=8, op_offload=1)
- IQ2_XXS (10GB) is the optimal quantization for RTX 3070 8GB — smallest model = most GPU layers = fastest
- n_gpu=27 is the sweet spot for IQ2_XXS (28 collapses to 8 tok/s, 26 caps at ~27 tok/s)
- threads=8 marginally beats 11 (~29.1 vs 28.9) — less CPU thread contention with high GPU offload
- op_offload=1 is critical (+40% over op_offload=0)
- batch=32/16 is optimal; batch=64/32 collapses to ~7 tok/s at n_gpu=27 (OOM-adjacent)
- Q2_K_XL (12GB) only fits n_gpu=22 → 17.9 tok/s (much worse than IQ2_XXS)
- IQ3_S (13GB) only fits n_gpu=20 → 10.8 tok/s
- IQ2_M (11GB) at n_gpu=26 → 27.0 tok/s (good but IQ2_XXS still wins)
- Massive variance due to concurrent system load (swerex, Ollama, other subagents) — results range 5-30 tok/s for same config
- Ollama model keepalive can consume 700-1400MB VRAM — must unload before bench
- Progress: 12.1 → 21.6 → 25.2 → 29.9 tok/s across Phases 6-12 (147% improvement!)

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
