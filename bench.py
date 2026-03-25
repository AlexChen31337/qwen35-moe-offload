"""
bench.py — THE ONLY FILE YOU EDIT in the autoresearch loop.

Modify expert cache params, loading strategy, prefetch config, quantization.
Do NOT touch harness.py.

Usage: uv run python bench.py > run.log 2>&1
"""
import os, time, random
from pathlib import Path
import torch

from harness import run_benchmark, get_model_path, NVMeTracker

# ---------------------------------------------------------------------------
# EXPERIMENT KNOBS — modify these
# ---------------------------------------------------------------------------

CACHE_WINDOW_K = 5            # sliding window size (tokens)
MAX_CACHED_EXPERTS = 50       # max experts to keep in DRAM cache
PREFETCH_THREADS = 4          # async NVMe prefetch thread count (0 = disabled)
BUNDLE_MODE = "separate"      # "gate_down" | "gate_up_down" | "separate"
READ_ALIGN_BYTES = 131072     # NVMe read alignment (32KB, 128KB, etc)
VRAM_CACHE_FRACTION = 0.0     # fraction of cache to keep in VRAM (0.0 = all DRAM)
QUANTIZATION = "Q4_K_M"       # "Q4_K_M" | "Q5_K_M" | "Q8_0"
PREDICTOR_THRESHOLD = 0.5     # expert activation predictor confidence cutoff
PREFETCH_LOOKAHEAD = 1        # tokens to look ahead for prefetch

# ---------------------------------------------------------------------------
# Expert loader (modify internals to test strategies)
# ---------------------------------------------------------------------------

class ExpertLoader:
    """
    Loads expert weights from NVMe on demand.
    Modify the loading strategy here to experiment.
    """

    def __init__(self, model_dir: Path, nvme_tracker: NVMeTracker):
        self.model_dir = model_dir
        self.nvme_tracker = nvme_tracker
        self._cache = {}           # (layer, expert_id) → tensor
        self._window = []          # recent token activation sets
        self._hits = 0
        self._misses = 0

    def get_experts(self, layer_idx: int, expert_ids: list[int]) -> dict:
        """Load experts for this layer, using cache if available."""
        result = {}
        to_load = []

        for eid in expert_ids:
            key = (layer_idx, eid)
            if key in self._cache:
                self._hits += 1
                result[eid] = self._cache[key]
            else:
                self._misses += 1
                to_load.append(eid)

        if to_load:
            loaded = self._load_from_nvme(layer_idx, to_load)
            result.update(loaded)
            for eid, tensor in loaded.items():
                self._insert_cache((layer_idx, eid), tensor)

        # Update sliding window
        self._window.append({(layer_idx, eid) for eid in expert_ids})
        if len(self._window) > CACHE_WINDOW_K:
            self._window.pop(0)
            self._evict_window()

        return result

    def _load_from_nvme(self, layer_idx: int, expert_ids: list[int]) -> dict:
        """
        Simulate loading expert weights from NVMe.
        In real impl: mmap file, seek to offset, read chunk.
        Here: simulate IO time based on actual NVMe throughput profile.
        """
        result = {}
        # Expert weight size (Q4_K_M): gate+down bundled = ~3MB per expert
        bytes_per_expert = {
            "gate_down": 3 * 1024 * 1024,
            "gate_up_down": 4.5 * 1024 * 1024,
            "separate": 1.5 * 1024 * 1024,  # per matrix, 3 reads
        }[BUNDLE_MODE]

        # Simulate NVMe read time based on block size
        # RTX 3070 system: ~3GB/s sequential, ~50MB/s random 4K
        # READ_ALIGN_BYTES determines effective throughput
        if READ_ALIGN_BYTES >= 512 * 1024:
            throughput_mbs = 2800  # ~sequential
        elif READ_ALIGN_BYTES >= 128 * 1024:
            throughput_mbs = 1500
        elif READ_ALIGN_BYTES >= 32 * 1024:
            throughput_mbs = 500
        else:
            throughput_mbs = 80  # random 4K territory

        total_bytes = bytes_per_expert * len(expert_ids)
        io_time = total_bytes / (throughput_mbs * 1e6)
        time.sleep(io_time)  # simulate NVMe latency

        self.nvme_tracker.record_read(int(total_bytes))

        # Return dummy tensors (replace with real weights in Phase 2+)
        for eid in expert_ids:
            result[eid] = torch.zeros(512, 2048)  # placeholder

        return result

    def _insert_cache(self, key: tuple, tensor):
        if len(self._cache) >= MAX_CACHED_EXPERTS:
            # Evict LRU
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = tensor

    def _evict_window(self):
        if len(self._window) < CACHE_WINDOW_K:
            return
        active = set().union(*self._window)
        for k in [k for k in self._cache if k not in active]:
            del self._cache[k]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Mock forward pass (replace with real model in Phase 2+)
# ---------------------------------------------------------------------------

def generate(prompt_tokens: list[int], max_new_tokens: int, nvme_tracker: NVMeTracker):
    """
    Simulate generation with MoE expert loading.
    Replace inner logic with real llama.cpp / transformers inference in Phase 2+.
    """
    NUM_LAYERS = 40
    NUM_EXPERTS = 256
    ACTIVE_EXPERTS = 9  # 8 routed + 1 shared

    loader = ExpertLoader(Path("./models"), nvme_tracker)

    tokens_generated = 0
    t_deadline = time.perf_counter() + 300  # 5 min budget

    for step in range(max_new_tokens):
        if time.perf_counter() > t_deadline:
            break

        # Simulate attention forward (GPU compute — fast)
        time.sleep(0.002)  # ~2ms GPU compute for attention

        # MoE layers: load experts per layer
        for layer_idx in range(NUM_LAYERS):
            # Router selects top-8 experts (simulate with random selection)
            active = random.sample(range(NUM_EXPERTS), ACTIVE_EXPERTS)

            # Load expert weights (this is where the IO happens)
            loader.get_experts(layer_idx, active)

            # Simulate FFN compute with loaded experts (GPU)
            time.sleep(0.0001)  # ~0.1ms per layer FFN

        tokens_generated += 1

    return tokens_generated, loader.hit_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Config: k={CACHE_WINDOW_K} max_cached={MAX_CACHED_EXPERTS} "
          f"prefetch={PREFETCH_THREADS} bundle={BUNDLE_MODE} "
          f"align={READ_ALIGN_BYTES//1024}KB quant={QUANTIZATION}")

    run_benchmark(generate)
