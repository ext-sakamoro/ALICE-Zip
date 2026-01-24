#!/usr/bin/env python3
"""
ALICE-Zip Benchmark Suite
==========================

Comprehensive benchmarks comparing ALICE-Zip compression with traditional methods.

Usage:
    python run_benchmarks.py [--output results.json] [--quick]

Author: Moroya Sakamoto
License: MIT
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import zlib
import lzma

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alice_zip import ALICEZip, ALICE_VERSION

# Check optional dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    data_type: str
    original_size: int
    alice_compressed_size: int
    zip_compressed_size: int
    lzma_compressed_size: int
    alice_ratio: float
    zip_ratio: float
    lzma_ratio: float
    alice_compress_ms: float
    alice_decompress_ms: float
    zip_compress_ms: float
    zip_decompress_ms: float
    lzma_compress_ms: float
    lzma_decompress_ms: float
    alice_lossless: bool
    winner: str
    notes: str = ""


def benchmark_data(name: str, data: np.ndarray, data_type: str, notes: str = "") -> BenchmarkResult:
    """Run benchmark on a single dataset"""
    original_size = data.nbytes
    raw_bytes = data.tobytes()

    # ZIP compression
    start = time.perf_counter()
    zip_compressed = zlib.compress(raw_bytes, level=9)
    zip_compress_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    zip_decompressed = zlib.decompress(zip_compressed)
    zip_decompress_time = (time.perf_counter() - start) * 1000

    # LZMA compression
    start = time.perf_counter()
    lzma_compressed = lzma.compress(raw_bytes, preset=6)
    lzma_compress_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    lzma_decompressed = lzma.decompress(lzma_compressed)
    lzma_decompress_time = (time.perf_counter() - start) * 1000

    # ALICE-Zip compression (using benchmark method since compress is Commercial)
    results = ALICEZip.benchmark_vs_traditional(data)

    alice_compressed_size = results.get('alice_zip', {}).get('compressed_size', original_size)
    alice_ratio = original_size / alice_compressed_size if alice_compressed_size > 0 else 1.0
    alice_compress_time = results.get('alice_zip', {}).get('compress_time_ms', 0)
    alice_decompress_time = results.get('alice_zip', {}).get('decompress_time_ms', 0)
    alice_lossless = results.get('alice_zip', {}).get('lossless', False)

    zip_ratio = original_size / len(zip_compressed)
    lzma_ratio = original_size / len(lzma_compressed)

    # Determine winner
    ratios = {
        'alice': alice_ratio,
        'zip': zip_ratio,
        'lzma': lzma_ratio
    }
    winner = max(ratios, key=ratios.get)

    return BenchmarkResult(
        name=name,
        data_type=data_type,
        original_size=original_size,
        alice_compressed_size=alice_compressed_size,
        zip_compressed_size=len(zip_compressed),
        lzma_compressed_size=len(lzma_compressed),
        alice_ratio=alice_ratio,
        zip_ratio=zip_ratio,
        lzma_ratio=lzma_ratio,
        alice_compress_ms=alice_compress_time,
        alice_decompress_ms=alice_decompress_time,
        zip_compress_ms=zip_compress_time,
        zip_decompress_ms=zip_decompress_time,
        lzma_compress_ms=lzma_compress_time,
        lzma_decompress_ms=lzma_decompress_time,
        alice_lossless=alice_lossless,
        winner=winner,
        notes=notes
    )


def generate_test_data() -> List[tuple]:
    """Generate test datasets"""
    datasets = []

    # 1. Polynomial data (ALICE should excel)
    x = np.arange(1000, dtype=np.float32)
    polynomial_data = x**2 + 2*x + 1
    datasets.append(("polynomial_quadratic", polynomial_data, "numerical", "y = x² + 2x + 1"))

    x = np.arange(500, dtype=np.float32)
    cubic_data = x**3 - x**2 + x
    datasets.append(("polynomial_cubic", cubic_data.astype(np.float32), "numerical", "y = x³ - x² + x"))

    # 2. Sine wave data (ALICE should excel)
    t = np.linspace(0, 1, 1000, dtype=np.float32)
    sine_data = np.sin(2 * np.pi * 10 * t)
    datasets.append(("sine_single", sine_data, "numerical", "Single 10Hz sine wave"))

    # Multi-tone sine
    multi_sine = (np.sin(2 * np.pi * 5 * t) +
                  0.5 * np.sin(2 * np.pi * 10 * t) +
                  0.25 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
    datasets.append(("sine_multi", multi_sine, "numerical", "5Hz + 10Hz + 20Hz"))

    # 3. Linear data (ALICE should excel)
    linear_data = np.linspace(0, 100, 1000, dtype=np.float32)
    datasets.append(("linear", linear_data, "numerical", "Linear ramp 0-100"))

    # 4. Constant data (all methods should compress well)
    constant_data = np.full(1000, 42.0, dtype=np.float32)
    datasets.append(("constant", constant_data, "numerical", "All values = 42"))

    # 5. Random data (ALICE should fallback to LZMA)
    random_data = np.random.randn(1000).astype(np.float32)
    datasets.append(("random_gaussian", random_data, "numerical", "Gaussian random"))

    random_uniform = np.random.uniform(0, 1, 1000).astype(np.float32)
    datasets.append(("random_uniform", random_uniform, "numerical", "Uniform random [0,1]"))

    # 6. Perlin-like noise (ALICE should be competitive)
    # Approximate with sum of sines at different frequencies
    perlin_approx = np.zeros(1000, dtype=np.float32)
    for i in range(1, 6):
        freq = 2 ** i
        amp = 1.0 / (2 ** i)
        perlin_approx += amp * np.sin(2 * np.pi * freq * t)
    datasets.append(("perlin_approx", perlin_approx, "numerical", "Sum of octaves (fBm-like)"))

    # 7. Step function (challenging for ALICE)
    step_data = np.zeros(1000, dtype=np.float32)
    step_data[500:] = 1.0
    datasets.append(("step_function", step_data, "numerical", "Step at midpoint"))

    # 8. Sparse data (ALICE might use RLE-like approach)
    sparse_data = np.zeros(1000, dtype=np.float32)
    sparse_data[::100] = 1.0
    datasets.append(("sparse", sparse_data, "numerical", "1% non-zero values"))

    # 9. Exponential decay (polynomial approximation)
    exp_data = np.exp(-np.linspace(0, 5, 1000)).astype(np.float32)
    datasets.append(("exponential_decay", exp_data, "numerical", "e^(-x)"))

    # 10. Large dataset
    large_polynomial = (np.arange(10000, dtype=np.float32) ** 2)
    datasets.append(("large_polynomial", large_polynomial, "numerical", "10K samples, x²"))

    return datasets


def run_benchmarks(quick: bool = False) -> List[BenchmarkResult]:
    """Run all benchmarks"""
    datasets = generate_test_data()

    if quick:
        # Only run a subset for quick testing
        datasets = datasets[:5]

    results = []
    print(f"Running {len(datasets)} benchmarks...")
    print("-" * 80)

    for name, data, data_type, notes in datasets:
        print(f"Benchmarking: {name}...", end=" ", flush=True)
        try:
            result = benchmark_data(name, data, data_type, notes)
            results.append(result)
            print(f"Winner: {result.winner} ({result.alice_ratio:.1f}x / {result.zip_ratio:.1f}x / {result.lzma_ratio:.1f}x)")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table"""
    print("\n" + "=" * 100)
    print("ALICE-Zip Benchmark Results")
    print("=" * 100)
    print(f"ALICE-Zip Version: {ALICE_VERSION[0]}.{ALICE_VERSION[1]}")
    print("-" * 100)

    # Header
    print(f"{'Name':<25} {'Type':<12} {'Size':>8} {'ALICE':>10} {'ZIP':>10} {'LZMA':>10} {'Winner':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<25} {r.data_type:<12} {r.original_size:>8} "
              f"{r.alice_ratio:>9.1f}x {r.zip_ratio:>9.1f}x {r.lzma_ratio:>9.1f}x {r.winner:>8}")

    print("-" * 100)

    # Summary statistics
    alice_wins = sum(1 for r in results if r.winner == 'alice')
    zip_wins = sum(1 for r in results if r.winner == 'zip')
    lzma_wins = sum(1 for r in results if r.winner == 'lzma')

    print(f"\nSummary:")
    print(f"  ALICE-Zip wins: {alice_wins}/{len(results)}")
    print(f"  ZIP wins:       {zip_wins}/{len(results)}")
    print(f"  LZMA wins:      {lzma_wins}/{len(results)}")

    # Average ratios
    avg_alice = np.mean([r.alice_ratio for r in results])
    avg_zip = np.mean([r.zip_ratio for r in results])
    avg_lzma = np.mean([r.lzma_ratio for r in results])

    print(f"\nAverage Compression Ratios:")
    print(f"  ALICE-Zip: {avg_alice:.2f}x")
    print(f"  ZIP:       {avg_zip:.2f}x")
    print(f"  LZMA:      {avg_lzma:.2f}x")

    # Best cases for ALICE
    print(f"\nBest cases for ALICE-Zip:")
    alice_best = sorted(results, key=lambda r: r.alice_ratio, reverse=True)[:3]
    for r in alice_best:
        print(f"  {r.name}: {r.alice_ratio:.1f}x (vs ZIP {r.zip_ratio:.1f}x, LZMA {r.lzma_ratio:.1f}x)")

    # Worst cases for ALICE
    print(f"\nWorst cases for ALICE-Zip:")
    alice_worst = sorted(results, key=lambda r: r.alice_ratio)[:3]
    for r in alice_worst:
        print(f"  {r.name}: {r.alice_ratio:.1f}x (vs ZIP {r.zip_ratio:.1f}x, LZMA {r.lzma_ratio:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description='ALICE-Zip Benchmark Suite')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of benchmarks')
    args = parser.parse_args()

    results = run_benchmarks(quick=args.quick)

    print_results_table(results)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
