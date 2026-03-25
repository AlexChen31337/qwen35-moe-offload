#!/usr/bin/env python3
"""
Benchmark NVMe read throughput: sequential vs random.
This determines how fast we can stream expert weights from disk.
"""
import os, time, random, tempfile, struct
import numpy as np


BLOCK_SIZES = [4 * 1024, 32 * 1024, 512 * 1024, 4 * 1024 * 1024]  # 4K, 32K, 512K, 4M
FILE_SIZE = 512 * 1024 * 1024  # 512MB test file
NUM_READS = 200


def create_test_file(path: str, size: int):
    print(f"Creating {size/1e6:.0f}MB test file at {path}...")
    data = os.urandom(size)
    with open(path, "wb") as f:
        f.write(data)
    return path


def benchmark_sequential(path: str, block_size: int, count: int = 50) -> float:
    """Returns MB/s for sequential reads of block_size."""
    total_bytes = 0
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        for _ in range(count):
            buf = f.read(block_size)
            if not buf:
                f.seek(0)
                buf = f.read(block_size)
            total_bytes += len(buf)
    elapsed = time.perf_counter() - t0
    return total_bytes / elapsed / 1e6  # MB/s


def benchmark_random(path: str, block_size: int, count: int = NUM_READS) -> float:
    """Returns MB/s for random reads of block_size."""
    file_size = os.path.getsize(path)
    max_offset = file_size - block_size
    offsets = [random.randint(0, max_offset) & ~(4095)  # align to 4K
               for _ in range(count)]
    
    total_bytes = 0
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        for offset in offsets:
            f.seek(offset)
            buf = f.read(block_size)
            total_bytes += len(buf)
    elapsed = time.perf_counter() - t0
    return total_bytes / elapsed / 1e6  # MB/s


def expert_load_simulation():
    """Simulate loading one Q4_K_M expert block (gate+down bundled, ~3MB)."""
    EXPERT_BLOCK_SIZE = 3 * 1024 * 1024  # 3MB — bundled gate+down for one expert
    NUM_EXPERTS_PER_TOKEN = 9
    NUM_TOKENS = 20
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
        f.write(os.urandom(FILE_SIZE))
    
    try:
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            for _ in range(NUM_TOKENS):
                for _ in range(NUM_EXPERTS_PER_TOKEN):
                    offset = random.randint(0, FILE_SIZE - EXPERT_BLOCK_SIZE) & ~(4095)
                    f.seek(offset)
                    f.read(EXPERT_BLOCK_SIZE)
        elapsed = time.perf_counter() - t0
        
        total_reads = NUM_TOKENS * NUM_EXPERTS_PER_TOKEN
        avg_ms = elapsed / total_reads * 1000
        mbps = (EXPERT_BLOCK_SIZE * total_reads) / elapsed / 1e6
        
        print(f"\n--- Expert Load Simulation ---")
        print(f"Expert block size: {EXPERT_BLOCK_SIZE/1e6:.1f}MB (bundled gate+down)")
        print(f"Experts per token: {NUM_EXPERTS_PER_TOKEN}")
        print(f"Total tokens: {NUM_TOKENS}")
        print(f"Avg load time per expert: {avg_ms:.1f}ms")
        print(f"Total time for {NUM_TOKENS} tokens: {elapsed*1000:.0f}ms")
        print(f"Effective throughput: {mbps:.0f} MB/s")
        print(f"Projected tok/s (IO bound): {1000/avg_ms/NUM_EXPERTS_PER_TOKEN:.1f}")
    finally:
        os.unlink(path)


def main():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    
    try:
        create_test_file(path, FILE_SIZE)
        # Drop OS page cache (requires root, skip if not available)
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except PermissionError:
            print("Note: could not drop page cache (run as root for accurate results)")
        
        print("\n=== NVMe Throughput Benchmark ===\n")
        print(f"{'Block Size':>12} | {'Sequential MB/s':>16} | {'Random MB/s':>12} | {'Ratio':>8}")
        print("-" * 60)
        
        for bs in BLOCK_SIZES:
            seq_mbs = benchmark_sequential(path, bs)
            rand_mbs = benchmark_random(path, bs)
            ratio = seq_mbs / rand_mbs if rand_mbs > 0 else float('inf')
            print(f"{bs//1024:>10}KB | {seq_mbs:>14.0f}   | {rand_mbs:>10.0f}   | {ratio:>6.1f}x")
        
        expert_load_simulation()
        
        print("\n=== Recommendation ===")
        print("Expert block size should be >= 512KB for good NVMe throughput.")
        print("Bundling gate+down (~3MB) is optimal for this NVMe.")
        
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
