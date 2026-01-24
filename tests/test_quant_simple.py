#!/usr/bin/env python3
"""
Simple 8-bit Quantization Test (1D data only, no Perlin)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ys/ALICE-Zip')

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.generators import CompressionEngine


def psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    rng = float(orig.max() - orig.min())
    return 20 * np.log10(rng / np.sqrt(mse)) if rng > 0 else float('inf')


def run_test(name, data):
    print(f"\n{name}")
    print(f"  Original: {data.nbytes:,} bytes")

    # Force LZMA to avoid slow Perlin fitting
    d32 = ProceduralCompressionDesigner()
    r32 = d32.compress(data, enable_lossless=True, quantize_residual=None,
                       force_engine=CompressionEngine.FUNCTION_FITTER)

    d8 = ProceduralCompressionDesigner()
    r8 = d8.compress(data, enable_lossless=True, quantize_residual=8,
                     force_engine=CompressionEngine.FUNCTION_FITTER)

    rec32 = d32.decompress(r32)
    rec8 = d8.decompress(r8)

    p32 = psnr(data, rec32)
    p8 = psnr(data, rec8)

    print(f"  32-bit: {r32.total_compressed_size:,}B, PSNR={'inf' if p32==float('inf') else f'{p32:.1f}dB'}")
    print(f"  8-bit:  {r8.total_compressed_size:,}B, PSNR={p8:.1f}dB")

    if r32.total_compressed_size > r8.total_compressed_size:
        reduction = (r32.total_compressed_size - r8.total_compressed_size) / r32.total_compressed_size * 100
        print(f"  Savings: {reduction:.0f}%")

    return p8 > 30


if __name__ == "__main__":
    print("=" * 60)
    print("8-bit Quantization Test")
    print("=" * 60)

    np.random.seed(42)
    results = []

    # Test 1: Polynomial + noise
    x = np.linspace(0, 10, 2000)
    data = (x**2 + np.random.randn(2000) * 10).astype(np.float32)
    results.append(run_test("Polynomial + Noise (2000 pts)", data))

    # Test 2: Sine + noise
    t = np.linspace(0, 10*np.pi, 3000)
    data = (np.sin(t) * 100 + np.random.randn(3000) * 5).astype(np.float32)
    results.append(run_test("Sine + Noise (3000 pts)", data))

    # Test 3: Multi-frequency
    t = np.linspace(0, 2*np.pi, 2000)
    data = (50*np.sin(3*t) + 30*np.sin(7*t) + np.random.randn(2000) * 5).astype(np.float32)
    results.append(run_test("Multi-freq + Noise (2000 pts)", data))

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if all(results) else "SOME FAIL")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
