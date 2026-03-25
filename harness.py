"""
MoE Flash Offload benchmark harness.
Fixed infrastructure — do NOT modify this file.
Edit bench.py instead.

Usage: uv run python bench.py > run.log 2>&1
"""
import os, time, gc, threading
from pathlib import Path
import torch
import numpy as np

# ---------------------------------------------------------------------------
# FIXED CONSTANTS — do not change
# ---------------------------------------------------------------------------
PROMPT_TOKENS = 512        # fixed prompt length
GENERATION_TOKENS = 256    # always generate exactly this many tokens
TIME_BUDGET_SEC = 300      # 5 minutes wall clock generation time
RANDOM_SEED = 42
MODEL_DIR = Path("./models")
RESULTS_LOG = Path("run.log")

# Fixed C4 prompt (deterministic, same across all experiments)
FIXED_PROMPT = (
    "The development of large language models has fundamentally changed how we think about "
    "artificial intelligence. These systems, trained on vast corpora of text, have demonstrated "
    "remarkable capabilities across a wide range of tasks. However, their deployment on consumer "
    "hardware remains challenging due to the substantial memory requirements. A 7 billion parameter "
    "model requires over 14 gigabytes of memory in half-precision floating point format, exceeding "
    "the capabilities of most personal devices. This has motivated research into efficient inference "
    "techniques that can reduce memory requirements while maintaining acceptable generation speed. "
    "One promising direction is the use of sparse activation patterns, where only a subset of model "
    "parameters are active for any given input token. Mixture of Experts architectures take this "
    "to an extreme, routing each token to a small subset of specialized expert networks. The "
    "Qwen3.5-35B-A3B model, for instance, activates only 9 of its 256 experts per token, resulting "
    "in 96.5 percent of expert weights being idle at any given moment. This sparsity presents an "
    "opportunity for efficient inference through selective weight loading from flash memory storage."
)


# ---------------------------------------------------------------------------
# Hardware monitoring
# ---------------------------------------------------------------------------
class HardwareMonitor:
    """Polls VRAM and RAM usage in background thread."""

    def __init__(self, interval_sec: float = 0.5):
        self.interval = interval_sec
        self._peak_vram_mb = 0.0
        self._peak_ram_mb = 0.0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        import psutil
        while self._running:
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e6
                self._peak_vram_mb = max(self._peak_vram_mb, vram)
            ram = psutil.Process().memory_info().rss / 1e6
            self._peak_ram_mb = max(self._peak_ram_mb, ram)
            time.sleep(self.interval)

    @property
    def peak_vram_mb(self) -> float:
        return self._peak_vram_mb

    @property
    def peak_ram_mb(self) -> float:
        return self._peak_ram_mb


# ---------------------------------------------------------------------------
# NVMe IO tracker
# ---------------------------------------------------------------------------
class NVMeTracker:
    """Wraps file reads to count total bytes loaded from NVMe."""

    def __init__(self):
        self._bytes_read = 0

    def record_read(self, nbytes: int):
        self._bytes_read += nbytes

    @property
    def total_bytes(self) -> int:
        return self._bytes_read

    def reset(self):
        self._bytes_read = 0


# ---------------------------------------------------------------------------
# Public API for bench.py
# ---------------------------------------------------------------------------

def get_model_path() -> Path:
    """Return path to the GGUF model file."""
    candidates = sorted(MODEL_DIR.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"No GGUF model found in {MODEL_DIR}. "
            "Run: uv run python scripts/download_model.py"
        )
    return candidates[0]


def run_benchmark(generate_fn, warmup_tokens: int = 16) -> dict:
    """
    Run the standardised benchmark.

    Args:
        generate_fn: callable(prompt_tokens, max_new_tokens, nvme_tracker) -> (tokens_generated, cache_hit_rate)
            - prompt_tokens: list[int] of length PROMPT_TOKENS
            - max_new_tokens: int
            - nvme_tracker: NVMeTracker instance
            Returns (num_tokens_generated: int, cache_hit_rate: float)

    Returns dict of metrics.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    monitor = HardwareMonitor()
    nvme_tracker = NVMeTracker()

    # Fake tokenize: just use ASCII codepoints as token IDs (for benchmarking)
    prompt_tokens = [ord(c) % 32000 for c in FIXED_PROMPT[:PROMPT_TOKENS]]
    prompt_tokens = prompt_tokens[:PROMPT_TOKENS]
    while len(prompt_tokens) < PROMPT_TOKENS:
        prompt_tokens.append(1)  # pad

    # Warmup (excluded from timing)
    generate_fn(prompt_tokens[:32], warmup_tokens, nvme_tracker)
    nvme_tracker.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Timed generation
    monitor.start()
    t0 = time.perf_counter()
    tokens_generated, cache_hit_rate = generate_fn(prompt_tokens, GENERATION_TOKENS, nvme_tracker)
    elapsed = time.perf_counter() - t0
    monitor.stop()

    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0.0
    nvme_per_tok = nvme_tracker.total_bytes / tokens_generated if tokens_generated > 0 else 0

    results = {
        "tok_per_sec": round(tok_per_sec, 3),
        "total_seconds": round(elapsed, 1),
        "peak_vram_mb": round(monitor.peak_vram_mb, 1),
        "peak_ram_mb": round(monitor.peak_ram_mb, 1),
        "nvme_bytes_per_tok": int(nvme_per_tok),
        "cache_hit_rate": round(cache_hit_rate, 3),
        "num_tokens": tokens_generated,
    }

    # Print in autoresearch format
    print("---")
    for k, v in results.items():
        print(f"{k}:{' ' * max(1, 20 - len(k))}{v}")

    return results
