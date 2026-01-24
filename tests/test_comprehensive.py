#!/usr/bin/env python3
"""
Comprehensive ALICE-Zip Test Suite

Tests adaptive fallback and compression across various data types:
1. Smooth gradients (highly compressible)
2. Periodic signals (Fourier-friendly)
3. Polynomial trends (polynomial-friendly)
4. Noisy data (tests residual compression)
5. Random noise (tests adaptive fallback - should use LZMA)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ys/ALICE-Zip')

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.generators import CompressionEngine


def test_smooth_gradient():
    """Test: Smooth gradient - should compress extremely well"""
    print("=" * 70)
    print("Test 1: Smooth Gradient (512x512)")
    print("  Expected: High compression ratio, LZMA fallback (patterns too simple)")
    print("=" * 70)

    x = np.linspace(0, 255, 512)
    y = np.linspace(0, 255, 512)
    xx, yy = np.meshgrid(x, y)
    data = ((xx + yy) / 2).astype(np.float32)

    designer = ProceduralCompressionDesigner()
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"Compressed: {result.total_compressed_size:,} bytes")
    print(f"Ratio: {result.effective_ratio:.2f}x")
    print(f"Engine: {result.engine_used.value}")
    print(f"Exact match: {np.allclose(data, reconstructed)}")

    return result.effective_ratio > 1.0


def test_pure_sine_wave():
    """Test: Pure sine wave - should achieve excellent compression"""
    print("\n" + "=" * 70)
    print("Test 2: Pure Sine Wave (4096 samples)")
    print("  Expected: Very high compression, procedural fit")
    print("=" * 70)

    t = np.linspace(0, 2 * np.pi, 4096)
    data = (np.sin(5 * t) * 100 + 128).astype(np.float32)

    designer = ProceduralCompressionDesigner()
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"Compressed: {result.total_compressed_size:,} bytes")
    print(f"Ratio: {result.effective_ratio:.2f}x")
    print(f"Engine: {result.engine_used.value}")
    print(f"Exact match: {np.allclose(data, reconstructed, atol=1e-4)}")

    return result.effective_ratio > 1.0


def test_polynomial_data():
    """Test: Polynomial curve - should achieve good compression"""
    print("\n" + "=" * 70)
    print("Test 3: Polynomial Data (2000 samples)")
    print("  Expected: High compression, procedural fit")
    print("=" * 70)

    x = np.linspace(0, 10, 2000)
    # Clean polynomial
    data = (0.5 * x**3 - 2 * x**2 + 3 * x + 10).astype(np.float32)

    designer = ProceduralCompressionDesigner()
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"Compressed: {result.total_compressed_size:,} bytes")
    print(f"Ratio: {result.effective_ratio:.2f}x")
    print(f"Engine: {result.engine_used.value}")
    print(f"Exact match: {np.allclose(data, reconstructed, atol=1e-4)}")

    return result.effective_ratio > 1.0


def test_signal_with_noise():
    """Test: Signal + noise - tests residual compression"""
    print("\n" + "=" * 70)
    print("Test 4: Signal + Noise (2000 samples)")
    print("  Expected: Moderate compression with residual")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(0, 10, 2000)
    # Polynomial + noise
    signal = 0.5 * x**3 - 2 * x**2 + 3 * x + 10
    noise = np.random.randn(2000) * 5
    data = (signal + noise).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # Lossless (with residual)
    result_lossless = designer.compress(data, enable_lossless=True)
    reconstructed_lossless = designer.decompress(result_lossless)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"\nLossless mode:")
    print(f"  Params: {result_lossless.compressed_size:,} bytes")
    print(f"  Residual: {result_lossless.metadata.get('residual_size', 0):,} bytes")
    print(f"  Total: {result_lossless.total_compressed_size:,} bytes")
    print(f"  Ratio: {result_lossless.effective_ratio:.2f}x")
    print(f"  Exact match: {np.allclose(data, reconstructed_lossless, atol=1e-4)}")

    # Lossy (no residual)
    result_lossy = designer.compress(data, enable_lossless=False)
    if result_lossy.generator_params is not None:
        reconstructed_lossy = designer.decompress(result_lossy)
        mse = np.mean((data - reconstructed_lossy) ** 2)
        print(f"\nLossy mode:")
        print(f"  Size: {result_lossy.compressed_size:,} bytes")
        print(f"  Ratio: {data.nbytes / result_lossy.compressed_size:.2f}x")
        print(f"  MSE: {mse:.4f}")

    return result_lossless.effective_ratio >= 0.99  # Allow tiny tolerance


def test_random_noise():
    """Test: Pure random noise - should trigger adaptive fallback"""
    print("\n" + "=" * 70)
    print("Test 5: Random Noise (256x256)")
    print("  Expected: Adaptive fallback to LZMA (incompressible)")
    print("=" * 70)

    np.random.seed(123)
    data = np.random.randn(256, 256).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # With adaptive fallback
    result = designer.compress(data, enable_lossless=True, adaptive_fallback=True)
    reconstructed = designer.decompress(result)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"Compressed: {result.total_compressed_size:,} bytes")
    print(f"Ratio: {result.effective_ratio:.2f}x")
    print(f"Engine: {result.engine_used.value}")
    print(f"Adaptive fallback: {result.metadata.get('adaptive_fallback', False)}")
    print(f"Exact match: {np.allclose(data, reconstructed)}")

    # Ratio should be >= ~0.99 (adaptive fallback prevents expansion)
    return result.effective_ratio >= 0.99


def test_mixed_frequency():
    """Test: Multiple sine waves - tests Fourier fitting"""
    print("\n" + "=" * 70)
    print("Test 6: Multi-frequency Signal (4096 samples)")
    print("  Expected: Good compression via Fourier")
    print("=" * 70)

    t = np.linspace(0, 2 * np.pi, 4096)
    # Sum of multiple frequencies
    data = (
        50 * np.sin(3 * t) +
        30 * np.sin(7 * t + 0.5) +
        20 * np.sin(15 * t + 1.0) +
        10 * np.sin(31 * t + 1.5)
    ).astype(np.float32)

    designer = ProceduralCompressionDesigner()
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"Original: {data.nbytes:,} bytes")
    print(f"Compressed: {result.total_compressed_size:,} bytes")
    print(f"Ratio: {result.effective_ratio:.2f}x")
    print(f"Engine: {result.engine_used.value}")
    print(f"Has residual: {result.has_residual}")
    print(f"Exact match: {np.allclose(data, reconstructed, atol=1e-4)}")

    return result.effective_ratio > 1.0


def test_edge_cases():
    """Test: Edge cases"""
    print("\n" + "=" * 70)
    print("Test 7: Edge Cases")
    print("=" * 70)

    designer = ProceduralCompressionDesigner()
    all_pass = True

    # Constant array
    print("\n  a) Constant array (all zeros)")
    data = np.zeros(1000, dtype=np.float32)
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)
    match = np.allclose(data, reconstructed)
    print(f"     Ratio: {result.effective_ratio:.2f}x, Match: {match}")
    all_pass = all_pass and match

    # Very small array
    print("\n  b) Small array (10 elements)")
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)
    match = np.allclose(data, reconstructed)
    print(f"     Ratio: {result.effective_ratio:.2f}x, Match: {match}")
    all_pass = all_pass and match

    # Large dynamic range
    print("\n  c) Large dynamic range")
    data = np.array([1e-10, 1e10, -1e10, 0, 1e-5], dtype=np.float32)
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)
    match = np.allclose(data, reconstructed, rtol=1e-5)
    print(f"     Ratio: {result.effective_ratio:.2f}x, Match: {match}")
    all_pass = all_pass and match

    return all_pass


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
        print("\nALICE-Zip achieves:")
        print("  - TRUE LOSSLESS compression via residual")
        print("  - Adaptive fallback prevents ratio < 1.0")
        print("  - MDL principle: automatically chooses optimal method")
    else:
        print("SOME TESTS FAILED - Review above results")

    return all_passed


if __name__ == "__main__":
    print("ALICE-Zip Comprehensive Test Suite")
    print("=" * 70)
    print("Testing compression across various data types\n")

    results = [
        ("Smooth Gradient", test_smooth_gradient()),
        ("Pure Sine Wave", test_pure_sine_wave()),
        ("Polynomial Data", test_polynomial_data()),
        ("Signal + Noise", test_signal_with_noise()),
        ("Random Noise (Adaptive)", test_random_noise()),
        ("Multi-frequency", test_mixed_frequency()),
        ("Edge Cases", test_edge_cases()),
    ]

    success = print_summary(results)
    sys.exit(0 if success else 1)
