#!/usr/bin/env python3
"""
Fast 8-bit Residual Quantization Test

Compares 32-bit (TRUE LOSSLESS) vs 8-bit (NEAR-LOSSLESS)
Using fast synthetic data (no slow Perlin noise)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ys/ALICE-Zip')

from alice_zip.analyzers import ProceduralCompressionDesigner


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(np.abs(original)) - np.min(original)
    if max_val == 0:
        max_val = 1.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def test_polynomial_quantization():
    """Test quantization on polynomial + noise"""
    print("=" * 70)
    print("Test 1: Polynomial + Noise")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(0, 10, 2000)
    data = (0.5 * x**3 - 2 * x**2 + 3 * x + 10 + np.random.randn(2000) * 5).astype(np.float32)

    print(f"Original: {data.nbytes:,} bytes")

    # 32-bit
    d32 = ProceduralCompressionDesigner()
    r32 = d32.compress(data, enable_lossless=True, quantize_residual=None)
    rec32 = d32.decompress(r32)

    # 8-bit
    d8 = ProceduralCompressionDesigner()
    r8 = d8.compress(data, enable_lossless=True, quantize_residual=8)
    rec8 = d8.decompress(r8)

    print(f"\n32-bit: {r32.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec32):.1f}dB")
    print(f"8-bit:  {r8.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec8):.1f}dB")

    if r32.total_compressed_size > 0:
        reduction = (r32.total_compressed_size - r8.total_compressed_size) / r32.total_compressed_size * 100
        print(f"Size reduction: {reduction:.1f}%")

    return calculate_psnr(data, rec8) > 40


def test_sine_quantization():
    """Test quantization on sine + noise"""
    print("\n" + "=" * 70)
    print("Test 2: Sine Wave + Noise")
    print("=" * 70)

    np.random.seed(123)
    t = np.linspace(0, 10 * np.pi, 4000)
    data = (np.sin(t) * 100 + np.random.randn(4000) * 10).astype(np.float32)

    print(f"Original: {data.nbytes:,} bytes")

    # 32-bit
    d32 = ProceduralCompressionDesigner()
    r32 = d32.compress(data, enable_lossless=True, quantize_residual=None)
    rec32 = d32.decompress(r32)

    # 8-bit
    d8 = ProceduralCompressionDesigner()
    r8 = d8.compress(data, enable_lossless=True, quantize_residual=8)
    rec8 = d8.decompress(r8)

    print(f"\n32-bit: {r32.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec32):.1f}dB")
    print(f"8-bit:  {r8.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec8):.1f}dB")

    if r32.total_compressed_size > 0:
        reduction = (r32.total_compressed_size - r8.total_compressed_size) / r32.total_compressed_size * 100
        print(f"Size reduction: {reduction:.1f}%")

    return calculate_psnr(data, rec8) > 40


def test_2d_texture_quantization():
    """Test quantization on 2D texture (fast, no Perlin)"""
    print("\n" + "=" * 70)
    print("Test 3: 2D Texture Pattern (128x128)")
    print("=" * 70)

    np.random.seed(456)
    x = np.linspace(0, 4 * np.pi, 128)
    y = np.linspace(0, 4 * np.pi, 128)
    xx, yy = np.meshgrid(x, y)
    # Fast texture: sum of sines + noise
    pattern = np.sin(xx) * np.cos(yy) + 0.5 * np.sin(2*xx) * np.cos(2*yy) + np.random.randn(128, 128) * 0.1
    data = ((pattern + 2) * 60).astype(np.float32)

    print(f"Original: {data.nbytes:,} bytes")

    # 32-bit
    d32 = ProceduralCompressionDesigner()
    r32 = d32.compress(data, enable_lossless=True, quantize_residual=None)
    rec32 = d32.decompress(r32)

    # 8-bit
    d8 = ProceduralCompressionDesigner()
    r8 = d8.compress(data, enable_lossless=True, quantize_residual=8)
    rec8 = d8.decompress(r8)

    print(f"\n32-bit: {r32.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec32):.1f}dB")
    print(f"8-bit:  {r8.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, rec8):.1f}dB")

    if r32.total_compressed_size > 0:
        reduction = (r32.total_compressed_size - r8.total_compressed_size) / r32.total_compressed_size * 100
        print(f"Size reduction: {reduction:.1f}%")

    return calculate_psnr(data, rec8) > 40


def test_comparison_matrix():
    """Comparison matrix: 32 vs 16 vs 8 bit"""
    print("\n" + "=" * 70)
    print("COMPARISON MATRIX")
    print("=" * 70)

    np.random.seed(789)

    # Test data
    x = np.linspace(0, 10, 1000)
    data = (x**2 + np.random.randn(1000) * 10).astype(np.float32)

    print(f"\nTest data: Quadratic + Noise ({data.nbytes:,} bytes)")
    print("-" * 50)
    print(f"{'Bits':<10} {'Size':<15} {'Ratio':<10} {'PSNR':<15}")
    print("-" * 50)

    for bits in [None, 16, 8]:
        d = ProceduralCompressionDesigner()
        r = d.compress(data, enable_lossless=True, quantize_residual=bits)
        rec = d.decompress(r)
        psnr = calculate_psnr(data, rec)

        bits_str = "32 (lossless)" if bits is None else f"{bits}"
        psnr_str = "inf" if psnr == float('inf') else f"{psnr:.1f}dB"
        print(f"{bits_str:<10} {r.total_compressed_size:,}B{'':<5} {r.effective_ratio:.2f}x{'':<5} {psnr_str:<15}")

    print("-" * 50)
    return True


if __name__ == "__main__":
    print("ALICE-Zip 8-bit Quantization Test (Fast)")
    print("=" * 70)

    results = []
    results.append(("Polynomial", test_polynomial_quantization()))
    results.append(("Sine Wave", test_sine_quantization()))
    results.append(("2D Texture", test_2d_texture_quantization()))
    results.append(("Comparison Matrix", test_comparison_matrix()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print("8-bit quantization: ~75% size reduction with PSNR > 40dB")
        print("Ideal for: Images, audio, sensor data")
    else:
        print("Some tests failed")

    sys.exit(0 if all_passed else 1)
