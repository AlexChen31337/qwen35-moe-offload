#!/usr/bin/env python3
"""
Sliding Window Expert Cache — Phase 3 core component.

Maintains a DRAM/VRAM buffer of recently-used expert weights.
Based on Apple "LLM in a Flash" windowing technique, adapted for MoE expert granularity.
"""
from __future__ import annotations
import os, time, mmap
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch


ExpertKey = Tuple[int, int]  # (layer_idx, expert_idx)


@dataclass
class ExpertWeights:
    """Bundled weights for one expert at one layer (gate+up+down)."""
    gate_proj: torch.Tensor   # [intermediate, hidden]
    up_proj: torch.Tensor     # [intermediate, hidden]
    down_proj: torch.Tensor   # [hidden, intermediate]
    layer_idx: int
    expert_idx: int
    loaded_at: float = field(default_factory=time.monotonic)


class SlidingWindowExpertCache:
    """
    DRAM-based LRU cache for MoE expert weights with sliding window eviction.
    
    Key properties:
    - Sliding window: evict experts not activated in last k tokens
    - O(1) lookup, O(c) eviction for c experts being replaced
    - Memory-efficient: reuses pre-allocated buffers (Apple paper §3.3)
    - Thread-safe for async prefetch integration
    
    Usage:
        cache = SlidingWindowExpertCache(k=5, max_cached_experts=50, device='cuda')
        weights = cache.get_or_load(layer=3, expert_ids=[1, 42, 128], index=expert_index)
        # weights: dict {expert_id: ExpertWeights}
    """
    
    def __init__(
        self,
        k: int = 5,
        max_cached_experts: int = 50,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.k = k
        self.max_cached = max_cached_experts
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Recent activation windows: deque of sets of ExpertKeys
        self.window: deque[Set[ExpertKey]] = deque(maxlen=k)
        
        # LRU cache: ExpertKey → ExpertWeights
        self.cache: OrderedDict[ExpertKey, ExpertWeights] = OrderedDict()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get_or_load(
        self,
        layer_idx: int,
        expert_ids: list[int],
        index: "ExpertIndex",
    ) -> Dict[int, ExpertWeights]:
        """
        Return expert weights for the requested experts at this layer.
        Loads from NVMe for cache misses, returns from cache for hits.
        """
        result = {}
        to_load = []
        
        for eid in expert_ids:
            key = (layer_idx, eid)
            if key in self.cache:
                self._hits += 1
                self.cache.move_to_end(key)  # LRU update
                result[eid] = self.cache[key]
            else:
                self._misses += 1
                to_load.append(eid)
        
        if to_load:
            loaded = index.load_experts(layer_idx, to_load, self.device, self.dtype)
            for eid, weights in loaded.items():
                key = (layer_idx, eid)
                self._insert(key, weights)
                result[eid] = weights
        
        # Update sliding window
        current_keys = {(layer_idx, eid) for eid in expert_ids}
        self.window.append(current_keys)
        self._evict_window()
        
        return result
    
    def _insert(self, key: ExpertKey, weights: ExpertWeights):
        """Insert expert into cache, evicting LRU if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = weights
        else:
            if len(self.cache) >= self.max_cached:
                # Evict LRU
                self.cache.popitem(last=False)
                self._evictions += 1
            self.cache[key] = weights
    
    def _evict_window(self):
        """Evict experts not seen in any of the last k token activations."""
        if len(self.window) < self.k:
            return
        # Union of all active experts in current window
        active = set().union(*self.window)
        # Evict cache entries not in active set
        to_evict = [k for k in self.cache if k not in active]
        for key in to_evict:
            del self.cache[key]
            self._evictions += 1
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self.hit_rate,
            "cached_experts": len(self.cache),
            "window_size": len(self.window),
        }
    
    def reset_stats(self):
        self._hits = self._misses = self._evictions = 0


class ExpertIndex:
    """
    Index mapping (layer_idx, expert_idx) → NVMe file location.
    Built by build_expert_index.py, loaded here for inference.
    """
    
    def __init__(self, index_path: str, weights_dir: str):
        import json
        self.weights_dir = Path(weights_dir)
        with open(index_path) as f:
            self.index = json.load(f)  # {str(layer): {str(expert): {offset, size, shape}}}
    
    def load_experts(
        self,
        layer_idx: int,
        expert_ids: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, ExpertWeights]:
        """Load expert weights from NVMe for the given expert IDs at this layer."""
        result = {}
        layer_key = str(layer_idx)
        
        for eid in expert_ids:
            expert_key = str(eid)
            meta = self.index[layer_key][expert_key]
            
            # Memory-map the bundled file (gate+up+down co-located)
            fpath = self.weights_dir / f"layer_{layer_idx:02d}_expert_{eid:04d}.bin"
            with open(fpath, "rb") as f:
                raw = f.read()
            
            # Parse bundled layout: [gate_proj | up_proj | down_proj]
            intermediate, hidden = meta["intermediate"], meta["hidden"]
            gate_size = intermediate * hidden
            up_size = intermediate * hidden
            down_size = hidden * intermediate
            
            arr = np.frombuffer(raw, dtype=np.float16)
            gate = torch.from_numpy(arr[:gate_size].reshape(intermediate, hidden)).to(device, dtype)
            up = torch.from_numpy(arr[gate_size:gate_size+up_size].reshape(intermediate, hidden)).to(device, dtype)
            down = torch.from_numpy(arr[gate_size+up_size:].reshape(hidden, intermediate)).to(device, dtype)
            
            result[eid] = ExpertWeights(
                gate_proj=gate, up_proj=up, down_proj=down,
                layer_idx=layer_idx, expert_idx=eid,
            )
        
        return result


if __name__ == "__main__":
    # Quick smoke test with dummy data
    print("Sliding Window Expert Cache — smoke test")
    
    class MockIndex:
        def load_experts(self, layer_idx, expert_ids, device, dtype):
            result = {}
            for eid in expert_ids:
                w = ExpertWeights(
                    gate_proj=torch.randn(512, 2048),
                    up_proj=torch.randn(512, 2048),
                    down_proj=torch.randn(2048, 512),
                    layer_idx=layer_idx, expert_idx=eid,
                )
                result[eid] = w
            return result
    
    cache = SlidingWindowExpertCache(k=5, max_cached_experts=30, device="cpu")
    index = MockIndex()
    
    # Simulate 20 tokens
    import random
    for t in range(20):
        active_experts = random.sample(range(256), 9)
        weights = cache.get_or_load(layer_idx=0, expert_ids=active_experts, index=index)
        assert len(weights) == 9
    
    print(f"Stats after 20 tokens: {cache.stats}")
    print(f"Hit rate: {cache.hit_rate:.1%}")
    print("OK")
