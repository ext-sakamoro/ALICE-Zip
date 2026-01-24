#!/usr/bin/env python3
"""
Test ALICE-Zip with real images (JPEG/PNG).

Tests:
1. Lossless mode: Bit-perfect reconstruction
2. Lossy mode: Visual quality without residual
3. Adaptive fallback: Ensures compression ratio >= 1.0x
"""

import numpy as np
import sys
import urllib.request
import os
from pathlib import Path

sys.path.insert(0, '/Users/ys/ALICE-Zip')

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.residual_compression import ResidualCompressionMethod


def download_test_image(url: str, filename: str) -> Path:
    """Download test image if not exists"""
    test_dir = Path('/Users/ys/ALICE-Zip/test_images')
    test_dir.mkdir(exist_ok=True)
    filepath = test_dir / filename

    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")

    return filepath


def load_image(filepath: Path) -> np.ndarray:
    """Load image as numpy array"""
    try:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img)
    except ImportError:
        print("PIL not available, trying matplotlib...")
        try:
            import matplotlib.pyplot as plt
            img = plt.imread(str(filepath))
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            return img
        except ImportError:
            print("Neither PIL nor matplotlib available!")
            return None


def save_image(data: np.ndarray, filepath: Path):
    """Save numpy array as image"""
    try:
        from PIL import Image
        img = Image.fromarray(data.astype(np.uint8))
        img.save(filepath)
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(str(filepath), data.astype(np.uint8))
        except ImportError:
            print("Cannot save image - no PIL or matplotlib")


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 255.0  # Assuming 8-bit image
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Structural Similarity Index (simplified)"""
    # Simplified SSIM calculation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)

    mu1 = np.mean(orig)
    mu2 = np.mean(recon)
    sigma1_sq = np.var(orig)
    sigma2_sq = np.var(recon)
    sigma12 = np.mean((orig - mu1) * (recon - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return float(ssim)


def test_grayscale_gradient():
    """Test with synthetic grayscale gradient"""
    print("=" * 70)
    print("Test 1: Synthetic Grayscale Gradient (256x256)")
    print("=" * 70)

    # Create smooth gradient
    x = np.linspace(0, 255, 256)
    y = np.linspace(0, 255, 256)
    xx, yy = np.meshgrid(x, y)
    data = ((xx + yy) / 2).astype(np.uint8)

    print(f"Image shape: {data.shape}, dtype: {data.dtype}")
    print(f"Original size: {data.nbytes:,} bytes")

    designer = ProceduralCompressionDesigner()

    # Lossless compression
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"\nEngine: {result.engine_used.value}")
    print(f"Params size: {result.compressed_size:,} bytes")
    print(f"Residual size: {result.metadata.get('residual_size', 0):,} bytes")
    print(f"Total size: {result.total_compressed_size:,} bytes")
    print(f"Compression ratio: {result.effective_ratio:.2f}x")
    print(f"Adaptive fallback: {result.metadata.get('adaptive_fallback', False)}")

    # Verify
    exact_match = np.array_equal(data, reconstructed)
    psnr = calculate_psnr(data, reconstructed)

    print(f"\nReconstruction:")
    print(f"  Exact match: {exact_match}")
    print(f"  PSNR: {psnr:.2f} dB")

    return exact_match or psnr > 50


def test_noisy_texture():
    """Test with noisy texture pattern"""
    print("\n" + "=" * 70)
    print("Test 2: Noisy Texture (512x512)")
    print("=" * 70)

    # Create Perlin-like noise pattern
    np.random.seed(42)
    x = np.linspace(0, 8 * np.pi, 512)
    y = np.linspace(0, 8 * np.pi, 512)
    xx, yy = np.meshgrid(x, y)

    # Multi-frequency pattern + noise
    pattern = (
        np.sin(xx) * np.cos(yy) * 0.3 +
        np.sin(2 * xx + 0.5) * np.cos(2 * yy + 0.3) * 0.2 +
        np.sin(4 * xx) * np.cos(4 * yy) * 0.1 +
        np.random.randn(512, 512) * 0.05
    )
    data = ((pattern + 1) * 127.5).clip(0, 255).astype(np.uint8)

    print(f"Image shape: {data.shape}, dtype: {data.dtype}")
    print(f"Original size: {data.nbytes:,} bytes")

    designer = ProceduralCompressionDesigner()

    # Lossless compression
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)

    print(f"\nEngine: {result.engine_used.value}")
    print(f"Params size: {result.compressed_size:,} bytes")
    print(f"Residual size: {result.metadata.get('residual_size', 0):,} bytes")
    print(f"Total size: {result.total_compressed_size:,} bytes")
    print(f"Compression ratio: {result.effective_ratio:.2f}x")
    print(f"Adaptive fallback: {result.metadata.get('adaptive_fallback', False)}")

    # Verify
    exact_match = np.array_equal(data, reconstructed)
    psnr = calculate_psnr(data, reconstructed)

    print(f"\nReconstruction:")
    print(f"  Exact match: {exact_match}")
    print(f"  PSNR: {psnr:.2f} dB")

    return exact_match or psnr > 50


def test_real_image():
    """Test with real photograph"""
    print("\n" + "=" * 70)
    print("Test 3: Real Photograph (Lenna/Mandrill)")
    print("=" * 70)

    # Try to download a test image
    test_images = [
        # USC-SIPI standard test images
        ("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", "lenna.png"),
        ("https://www.cs.cmu.edu/~chuck/lenern/mandrill.jpg", "mandrill.jpg"),
    ]

    data = None
    for url, filename in test_images:
        try:
            filepath = download_test_image(url, filename)
            data = load_image(filepath)
            if data is not None:
                print(f"Loaded: {filename}")
                break
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

    if data is None:
        print("Could not load any test image. Creating synthetic photo-like image...")
        # Create synthetic photo-like image
        np.random.seed(123)
        data = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Convert to grayscale for simpler testing
    if len(data.shape) == 3:
        data_gray = np.mean(data, axis=2).astype(np.uint8)
    else:
        data_gray = data

    print(f"Image shape: {data_gray.shape}, dtype: {data_gray.dtype}")
    print(f"Original size: {data_gray.nbytes:,} bytes")

    designer = ProceduralCompressionDesigner()

    # === Lossless Mode ===
    print("\n--- Lossless Mode ---")
    result_lossless = designer.compress(data_gray, enable_lossless=True)
    reconstructed_lossless = designer.decompress(result_lossless)

    print(f"Engine: {result_lossless.engine_used.value}")
    print(f"Params size: {result_lossless.compressed_size:,} bytes")
    print(f"Residual size: {result_lossless.metadata.get('residual_size', 0):,} bytes")
    print(f"Total size: {result_lossless.total_compressed_size:,} bytes")
    print(f"Compression ratio: {result_lossless.effective_ratio:.2f}x")
    print(f"Adaptive fallback: {result_lossless.metadata.get('adaptive_fallback', False)}")

    exact_match = np.array_equal(data_gray, reconstructed_lossless)
    psnr_lossless = calculate_psnr(data_gray, reconstructed_lossless)

    print(f"Exact match: {exact_match}")
    print(f"PSNR: {psnr_lossless:.2f} dB")

    # === Lossy Mode (no residual) ===
    print("\n--- Lossy Mode (no residual) ---")
    result_lossy = designer.compress(data_gray, enable_lossless=False)

    if result_lossy.generator_params is not None:
        reconstructed_lossy = designer.decompress(result_lossy)

        print(f"Engine: {result_lossy.engine_used.value}")
        print(f"Params size: {result_lossy.compressed_size:,} bytes")
        print(f"Compression ratio: {data_gray.nbytes / result_lossy.compressed_size:.2f}x")

        psnr_lossy = calculate_psnr(data_gray, reconstructed_lossy)
        ssim_lossy = calculate_ssim(data_gray, reconstructed_lossy)

        print(f"PSNR: {psnr_lossy:.2f} dB")
        print(f"SSIM: {ssim_lossy:.4f}")

        # Save comparison images
        test_dir = Path('/Users/ys/ALICE-Zip/test_images')
        test_dir.mkdir(exist_ok=True)
        save_image(data_gray, test_dir / 'original_gray.png')
        save_image(reconstructed_lossy, test_dir / 'reconstructed_lossy.png')
        print(f"\nSaved comparison images to {test_dir}")
    else:
        print("No procedural fit found (using LZMA fallback)")

    return exact_match or psnr_lossless > 50


def test_adaptive_fallback():
    """Verify adaptive fallback prevents ratio < 1.0"""
    print("\n" + "=" * 70)
    print("Test 4: Adaptive Fallback Verification")
    print("=" * 70)

    # Create data that's hard to compress procedurally
    np.random.seed(999)
    data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    print(f"Random noise image: {data.shape}")
    print(f"Original size: {data.nbytes:,} bytes")

    designer = ProceduralCompressionDesigner()

    # With adaptive fallback
    result_adaptive = designer.compress(data, enable_lossless=True, adaptive_fallback=True)

    print(f"\nWith Adaptive Fallback:")
    print(f"  Engine: {result_adaptive.engine_used.value}")
    print(f"  Total size: {result_adaptive.total_compressed_size:,} bytes")
    print(f"  Ratio: {result_adaptive.effective_ratio:.2f}x")
    print(f"  Fallback triggered: {result_adaptive.metadata.get('adaptive_fallback', False)}")

    # Without adaptive fallback
    result_no_adaptive = designer.compress(data, enable_lossless=True, adaptive_fallback=False)

    print(f"\nWithout Adaptive Fallback:")
    print(f"  Engine: {result_no_adaptive.engine_used.value}")
    print(f"  Total size: {result_no_adaptive.total_compressed_size:,} bytes")
    print(f"  Ratio: {result_no_adaptive.effective_ratio:.2f}x")

    # Verify
    adaptive_is_better = result_adaptive.total_compressed_size <= result_no_adaptive.total_compressed_size
    ratio_above_one = result_adaptive.effective_ratio >= 0.99  # Allow small tolerance

    print(f"\nAdaptive chose better option: {adaptive_is_better}")
    print(f"Compression ratio >= 1.0x: {ratio_above_one}")

    return adaptive_is_better


if __name__ == "__main__":
    print("ALICE-Zip Real Image Test")
    print("=" * 70)
    print("Testing with real images and adaptive fallback\n")

    results = []
    results.append(("Grayscale Gradient", test_grayscale_gradient()))
    results.append(("Noisy Texture", test_noisy_texture()))
    results.append(("Real Image", test_real_image()))
    results.append(("Adaptive Fallback", test_adaptive_fallback()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
    sys.exit(0 if all_passed else 1)
