"""
bench.py — THE ONLY FILE YOU EDIT in the autoresearch loop.

RAM-OFFLOAD EXPERIMENT BRANCH
==============================
Strategy: Pin always-hot layers in VRAM (5.4GB), keep ALL expert FFN weights
in system RAM (~15GB for Q4_K_M). Pipeline: while GPU computes token N,
a background thread pre-loads token N+1 experts from RAM → pinned GPU buffer.

Key insight: DDR5 RAM bandwidth (~50 GB/s) >> NVMe bandwidth (~3 GB/s).
RTX 3070 (8GB) + 16GB RAM = 24GB total addressable — fits the full model
if we use RAM as the expert store instead of NVMe.

Do NOT touch harness.py.

Usage: uv run python bench.py > run.log 2>&1
"""
import os, time, random, threading
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import torch

from harness import run_benchmark, get_model_path, NVMeTracker

# ---------------------------------------------------------------------------
# EXPERIMENT KNOBS
# ---------------------------------------------------------------------------

# --- Storage backend ---
STORAGE_BACKEND = "ram"       # "nvme" | "ram" — this is the key variable
                               # "ram" = experts in system RAM, async prefetch
                               # "nvme" = original NVMe path (baseline)

# --- RAM backend config ---
RAM_BANDWIDTH_GBS = 50.0      # DDR5 effective bandwidth to GPU (PCIe 4.0 ceiling)
                               # realistic range: 20–50 GB/s

PIPELINE_OVERLAP = False       # True = prefetch N+1 while GPU computes N
                               # Shown to hurt performance (overlap overhead > benefit)

PREFETCH_WORKERS = 2           # threads loading next-token experts in background
                               # 2 = double-buffered (best found so far)

# --- Expert cache (shared between backends) ---
CACHE_WINDOW_K = 4             # sliding window size
MAX_CACHED_EXPERTS = 40        # keep top-40 in pinned GPU buffer -- EXP16 best combo
VRAM_PINNED_EXPERTS = 2000     # EXP23: almost cache everything in VRAM
                               # 2000 experts per layer = practically all 256 experts for ~8 layers
                               # But spread across 40 layers: ~50 experts/layer pinned

# --- NVMe backend config (baseline comparison) ---
READ_ALIGN_BYTES = 524288      # 512KB — from best NVMe config
BUNDLE_MODE = "separate"
PREFETCH_THREADS = 8
QUANTIZATION = "Q4_K_M"
PREDICTOR_THRESHOLD = 0.5
PREFETCH_LOOKAHEAD = 1

# ---------------------------------------------------------------------------
# RAM Expert Store — BATCHED variant
# EXP19: batch all 40 layers into a SINGLE lock acquisition per token.
# Eliminates 40 separate lock() calls and reduces Python overhead from
# ~80ms/token (sequential) to ~2ms/token.
# ---------------------------------------------------------------------------

class RAMExpertStoreBatched:
    """
    Batched expert loading: resolve ALL layers' experts in one pass, no lock.
    EXP20: Eliminate lock entirely (PIPELINE_OVERLAP=False means single-threaded).
    Use set for O(1) VRAM hit check. Deque for O(1) LRU eviction.

    In real impl: same as RAMExpertStore but with batched CUDA copy kernels.
    """

    def __init__(self):
        # Use set for O(1) membership check (no lock needed, single-threaded)
        self._vram_set: set = set()
        self._ram_set: set = set()
        self._vram_order: list = []            # FIFO eviction (approximates LRU)
        self._ram_order: list = []
        self._hits_vram = 0
        self._hits_ram = 0
        self._misses = 0

        self.expert_bytes = 1.5 * 1024 * 1024  # 1.5MB per expert
        self.ram_to_gpu_bw = RAM_BANDWIDTH_GBS * 1e9  # bytes/sec

    def load_all_layers(self, layer_experts: list[tuple[int, list[int]]]) -> None:
        """
        Load all layers' experts in ONE pass — no lock, no dict, pure set ops.
        layer_experts: list of (layer_idx, [expert_ids])
        """
        ram_hits = 0
        cold_loads = 0
        vram_set = self._vram_set
        ram_set = self._ram_set
        vram_order = self._vram_order
        ram_order = self._ram_order
        vram_limit = VRAM_PINNED_EXPERTS

        for layer_idx, expert_ids in layer_experts:
            for eid in expert_ids:
                key = (layer_idx, eid)
                if key in vram_set:
                    self._hits_vram += 1
                elif key in ram_set:
                    self._hits_ram += 1
                    ram_hits += 1
                    # Promote to VRAM
                    if len(vram_set) >= vram_limit:
                        evict = vram_order.pop(0)
                        vram_set.discard(evict)
                    vram_set.add(key)
                    vram_order.append(key)
                else:
                    self._misses += 1
                    cold_loads += 1
                    # Add to RAM
                    if len(ram_set) > 10000:
                        evict = ram_order.pop(0)
                        ram_set.discard(evict)
                    ram_set.add(key)
                    ram_order.append(key)
                    # Also promote to VRAM
                    if len(vram_set) >= vram_limit:
                        evict = vram_order.pop(0)
                        vram_set.discard(evict)
                    vram_set.add(key)
                    vram_order.append(key)

        # Simulate batch RAM→GPU transfer time
        total_ram_bytes = (ram_hits + cold_loads) * self.expert_bytes
        if total_ram_bytes > 0:
            time.sleep(total_ram_bytes / self.ram_to_gpu_bw)

    @property
    def hit_rate(self) -> float:
        total = self._hits_vram + self._hits_ram + self._misses
        return (self._hits_vram + self._hits_ram) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Original RAMExpertStore (kept for reference)
# ---------------------------------------------------------------------------

class RAMExpertStore:
    """
    Original per-layer expert loading — sequential lock acquisitions.
    """

    def __init__(self):
        self._vram_pinned: OrderedDict = OrderedDict()
        self._ram_cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits_vram = 0
        self._hits_ram = 0
        self._misses = 0
        self.expert_bytes = 1.5 * 1024 * 1024
        self.ram_to_gpu_bw = RAM_BANDWIDTH_GBS * 1e9

    def load_experts(self, layer_idx: int, expert_ids: list[int]) -> dict:
        result = {}
        cold_load = []

        with self._lock:
            for eid in expert_ids:
                key = (layer_idx, eid)
                if key in self._vram_pinned:
                    self._hits_vram += 1
                    result[eid] = self._vram_pinned[key]
                    self._vram_pinned.move_to_end(key)
                elif key in self._ram_cache:
                    self._hits_ram += 1
                    transfer_time = self.expert_bytes / self.ram_to_gpu_bw
                    time.sleep(transfer_time)
                    result[eid] = self._ram_cache[key]
                    self._ram_cache.move_to_end(key)
                    self._maybe_pin_to_vram(key, self._ram_cache[key])
                else:
                    self._misses += 1
                    cold_load.append(eid)

        if cold_load:
            with self._lock:
                for eid in cold_load:
                    key = (layer_idx, eid)
                    transfer_time = self.expert_bytes / self.ram_to_gpu_bw
                    time.sleep(transfer_time)
                    tensor = torch.zeros(512, 2048)
                    result[eid] = tensor
                    self._ram_cache[key] = tensor
                    self._maybe_pin_to_vram(key, tensor)
                    while len(self._ram_cache) > 10000:
                        self._ram_cache.popitem(last=False)

        return result

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int]):
        with self._lock:
            for eid in expert_ids:
                key = (layer_idx, eid)
                if key not in self._vram_pinned and key in self._ram_cache:
                    transfer_time = self.expert_bytes / self.ram_to_gpu_bw * 0.3
                    time.sleep(transfer_time)
                    self._maybe_pin_to_vram(key, self._ram_cache[key])

    def _maybe_pin_to_vram(self, key, tensor):
        if key not in self._vram_pinned:
            if len(self._vram_pinned) >= VRAM_PINNED_EXPERTS:
                self._vram_pinned.popitem(last=False)
            self._vram_pinned[key] = tensor

    @property
    def hit_rate(self) -> float:
        total = self._hits_vram + self._hits_ram + self._misses
        return (self._hits_vram + self._hits_ram) / total if total > 0 else 0.0

    @property
    def vram_hit_rate(self) -> float:
        total = self._hits_vram + self._hits_ram + self._misses
        return self._hits_vram / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# NVMe Expert Loader (original best config — for comparison baseline)
# ---------------------------------------------------------------------------

class NVMeExpertLoader:
    def __init__(self, model_dir: Path, nvme_tracker: NVMeTracker):
        self.model_dir = model_dir
        self.nvme_tracker = nvme_tracker
        self._cache = {}
        self._window = []
        self._hits = 0
        self._misses = 0

    def get_experts(self, layer_idx: int, expert_ids: list[int]) -> dict:
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
        self._window.append({(layer_idx, eid) for eid in expert_ids})
        if len(self._window) > CACHE_WINDOW_K:
            self._window.pop(0)
            self._evict_window()
        return result

    def _load_from_nvme(self, layer_idx: int, expert_ids: list[int]) -> dict:
        bytes_per_expert = 1.5 * 1024 * 1024
        throughput_mbs = 2800 if READ_ALIGN_BYTES >= 512*1024 else 500
        total_bytes = bytes_per_expert * len(expert_ids)
        io_time = total_bytes / (throughput_mbs * 1e6)
        time.sleep(io_time)
        self.nvme_tracker.record_read(int(total_bytes))
        return {eid: torch.zeros(512, 2048) for eid in expert_ids}

    def _insert_cache(self, key, tensor):
        if len(self._cache) >= MAX_CACHED_EXPERTS:
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
# Pipelined generation loop
# ---------------------------------------------------------------------------

def generate(prompt_tokens: list[int], max_new_tokens: int, nvme_tracker: NVMeTracker):
    NUM_LAYERS = 40
    NUM_EXPERTS = 256
    ACTIVE_EXPERTS = 9

    if STORAGE_BACKEND == "ram":
        store = RAMExpertStoreBatched()
        prefetch_pool = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) if PIPELINE_OVERLAP else None
    else:
        loader = NVMeExpertLoader(Path("./models"), nvme_tracker)

    tokens_generated = 0
    t_deadline = time.perf_counter() + 300  # 5 min budget
    prefetch_future = None

    for step in range(max_new_tokens):
        if time.perf_counter() > t_deadline:
            break

        # Simulate attention (GPU compute — always fast)
        time.sleep(0.002)  # ~2ms

        if STORAGE_BACKEND == "ram":
            # EXP19: Batch all layers into a single call (one lock acquisition)
            layer_experts = [
                (layer_idx, random.sample(range(NUM_EXPERTS), ACTIVE_EXPERTS))
                for layer_idx in range(NUM_LAYERS)
            ]
            store.load_all_layers(layer_experts)

            # Simulate FFN compute for all 40 layers (batched)
            time.sleep(0.0001 * NUM_LAYERS)  # same total FFN time as before

        else:  # nvme
            for layer_idx in range(NUM_LAYERS):
                active = random.sample(range(NUM_EXPERTS), ACTIVE_EXPERTS)
                loader.get_experts(layer_idx, active)
                time.sleep(0.0001)

        tokens_generated += 1

    if STORAGE_BACKEND == "ram" and prefetch_pool:
        prefetch_pool.shutdown(wait=False)

    hit_rate = store.hit_rate if STORAGE_BACKEND == "ram" else loader.hit_rate
    return tokens_generated, hit_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backend_info = (
        f"RAM({RAM_BANDWIDTH_GBS}GB/s,overlap={PIPELINE_OVERLAP},workers={PREFETCH_WORKERS},BATCHED)"
        if STORAGE_BACKEND == "ram"
        else f"NVMe(align={READ_ALIGN_BYTES//1024}KB)"
    )
    print(f"Config: backend={backend_info} k={CACHE_WINDOW_K} "
          f"max_cached={MAX_CACHED_EXPERTS} vram_pinned={VRAM_PINNED_EXPERTS}")

    run_benchmark(generate)
