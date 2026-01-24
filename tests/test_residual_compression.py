#!/usr/bin/env python3
"""
Test residual compression for TRUE LOSSLESS reconstruction.

Verifies: Original = Generated(params) + Decompress(residual)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ys/ALICE-Zip')

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.residual_compression import ResidualCompressionMethod


def test_perlin_noise_lossless():
    """Test lossless compression on Perlin-like noise"""
    print("=" * 60)
    print("Test 1: Perlin-like Noise (256x256)")
    print("=" * 60)

    # Generate test data (simulated Perlin-like pattern)
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    data = (np.sin(xx) * np.cos(yy) + np.random.randn(256, 256) * 0.1).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # Compress with lossless enabled
    result = designer.compress(data, enable_lossless=True)

    print(f"Engine: {result.engine_used.value}")
    print(f"Original: {result.original_size:,} bytes")
    print(f"Params: {result.compressed_size:,} bytes")
    print(f"Residual: {result.metadata.get('residual_size', 0):,} bytes")
    print(f"Total: {result.total_compressed_size:,} bytes")
    print(f"Effective ratio: {result.effective_ratio:.2f}x")
    print(f"Has residual: {result.has_residual}")
    print(f"Is lossless: {result.is_lossless}")

    # Decompress and verify
    reconstructed = designer.decompress(result)

    # Verify EXACT reconstruction
    max_diff = np.max(np.abs(data - reconstructed))
    mse = np.mean((data - reconstructed) ** 2)

    print(f"\nReconstruction verification:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  MSE: {mse:.2e}")
    print(f"  Exact match: {np.allclose(data, reconstructed, atol=1e-5)}")

    return result.is_lossless and np.allclose(data, reconstructed, atol=1e-5)


def test_polynomial_lossless():
    """Test lossless compression on polynomial data with noise"""
    print("\n" + "=" * 60)
    print("Test 2: Polynomial with Noise (1000 points)")
    print("=" * 60)

    # Generate polynomial with small noise
    np.random.seed(123)
    x = np.linspace(0, 10, 1000)
    data = (2.5 * x**3 - 1.2 * x**2 + 0.8 * x + 5.0 + np.random.randn(1000) * 0.5).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # Compress with lossless enabled
    result = designer.compress(data, enable_lossless=True)

    print(f"Engine: {result.engine_used.value}")
    print(f"Original: {result.original_size:,} bytes")
    print(f"Params: {result.compressed_size:,} bytes")
    print(f"Residual: {result.metadata.get('residual_size', 0):,} bytes")
    print(f"Total: {result.total_compressed_size:,} bytes")
    print(f"Effective ratio: {result.effective_ratio:.2f}x")
    print(f"Has residual: {result.has_residual}")
    print(f"Is lossless: {result.is_lossless}")

    # Decompress and verify
    reconstructed = designer.decompress(result)

    # Verify reconstruction
    max_diff = np.max(np.abs(data - reconstructed))
    mse = np.mean((data - reconstructed) ** 2)

    print(f"\nReconstruction verification:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  MSE: {mse:.2e}")
    print(f"  Exact match: {np.allclose(data, reconstructed, atol=1e-5)}")

    return result.is_lossless and np.allclose(data, reconstructed, atol=1e-5)


def test_fourier_lossless():
    """Test lossless compression on Fourier signal with harmonics"""
    print("\n" + "=" * 60)
    print("Test 3: Fourier Signal with Harmonics (2048 points)")
    print("=" * 60)

    # Generate complex periodic signal
    np.random.seed(456)
    n = 2048
    t = np.linspace(0, 2 * np.pi, n)
    data = (
        3.0 * np.sin(5 * t) +
        1.5 * np.sin(12 * t + 0.5) +
        0.8 * np.sin(23 * t + 1.2) +
        np.random.randn(n) * 0.2
    ).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # Compress with lossless enabled
    result = designer.compress(data, enable_lossless=True)

    print(f"Engine: {result.engine_used.value}")
    print(f"Original: {result.original_size:,} bytes")
    print(f"Params: {result.compressed_size:,} bytes")
    print(f"Residual: {result.metadata.get('residual_size', 0):,} bytes")
    print(f"Total: {result.total_compressed_size:,} bytes")
    print(f"Effective ratio: {result.effective_ratio:.2f}x")
    print(f"Has residual: {result.has_residual}")
    print(f"Is lossless: {result.is_lossless}")

    # Decompress and verify
    reconstructed = designer.decompress(result)

    # Verify reconstruction
    max_diff = np.max(np.abs(data - reconstructed))
    mse = np.mean((data - reconstructed) ** 2)

    print(f"\nReconstruction verification:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  MSE: {mse:.2e}")
    print(f"  Exact match: {np.allclose(data, reconstructed, atol=1e-5)}")

    return result.is_lossless and np.allclose(data, reconstructed, atol=1e-5)


def test_lossy_vs_lossless():
    """Compare lossy vs lossless compression"""
    print("\n" + "=" * 60)
    print("Test 4: Lossy vs Lossless Comparison")
    print("=" * 60)

    # Generate test data
    np.random.seed(789)
    data = (np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.randn(1000) * 0.3).astype(np.float32)

    designer = ProceduralCompressionDesigner()

    # Lossy compression
    result_lossy = designer.compress(data, enable_lossless=False)
    recon_lossy = designer.decompress(result_lossy)

    # Lossless compression
    result_lossless = designer.compress(data, enable_lossless=True)
    recon_lossless = designer.decompress(result_lossless)

    print("Lossy:")
    print(f"  Size: {result_lossy.total_compressed_size:,} bytes")
    print(f"  Ratio: {result_lossy.effective_ratio:.2f}x")
    print(f"  MSE: {np.mean((data - recon_lossy) ** 2):.6f}")

    print("\nLossless:")
    print(f"  Size: {result_lossless.total_compressed_size:,} bytes")
    print(f"  Ratio: {result_lossless.effective_ratio:.2f}x")
    print(f"  MSE: {np.mean((data - recon_lossless) ** 2):.2e}")
    print(f"  Exact: {np.allclose(data, recon_lossless, atol=1e-5)}")

    return result_lossless.is_lossless


if __name__ == "__main__":
    print("ALICE-Zip Residual Compression Test")
    print("===================================")
    print("Testing TRUE LOSSLESS reconstruction:")
    print("  Original = Generated(params) + Decompress(residual)\n")

    results = []
    results.append(("Perlin Noise", test_perlin_noise_lossless()))
    results.append(("Polynomial", test_polynomial_lossless()))
    results.append(("Fourier", test_fourier_lossless()))
    results.append(("Lossy vs Lossless", test_lossy_vs_lossless()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
    sys.exit(0 if all_passed else 1)
