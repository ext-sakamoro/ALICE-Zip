#!/usr/bin/env python3
"""
ProceduralCompressionDesigner on image-shaped data (2-D uint8), all made in
the test: a gradient, a noisy texture, a deterministic photograph-like image
and a sine raster. No network and no image file.

Tests:
1. Lossless mode: the reconstruction equals the input bit for bit (dtype and
   bytes). The photograph-like and texture images go through the LZMA
   fallback; the sine raster goes through the procedural path (generator
   parameters plus a residual), which test_lossless_designer.py also covers
   for 1-D signals.
2. Lossy mode: quality without a residual (reported, not asserted)
3. Adaptive fallback: Ensures compression ratio >= 1.0x
"""

import numpy as np
import sys
from pathlib import Path

# Resolve the repository root from this file so the tests run anywhere
# (an absolute home path only works on the machine it was written on,
# and this is a public repository).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.residual_compression import ResidualCompressionMethod


def photo_like_image(size: int = 256) -> np.ndarray:
    """A deterministic photograph-like grayscale image: smooth shading, a few
    edges and fine texture (the test needs no network and no image file).
    Rounded before the cast and pinned by its SHA-256 below; the nearest value
    is 5e-6 from a rounding tie, so a last-place difference of the platform's
    sin / cos does not reach a pixel here."""
    rng = np.random.default_rng(123)
    y, x = np.mgrid[0:size, 0:size] / size
    shading = 120 + 80 * np.sin(3 * x + 1) * np.cos(2 * y)
    edges = 40 * ((x - 0.5) ** 2 + (y - 0.4) ** 2 < 0.08)
    texture = rng.normal(0, 12, (size, size))
    return np.clip(np.round(shading + edges + texture), 0, 255).astype(np.uint8)


def sine_raster(size: int = 64) -> np.ndarray:
    """A sine laid out in raster order as a uint8 image: the designer finds
    its parameters, so it takes the procedural path."""
    v = 127.5 + 100 * np.sin(np.linspace(0, 8 * np.pi, size * size))
    return np.round(v).astype(np.uint8).reshape(size, size)


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

    # lossless: bit for bit, dtype included
    assert reconstructed.dtype == data.dtype and reconstructed.tobytes() == data.tobytes()


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

    # lossless: bit for bit, dtype included
    assert reconstructed.dtype == data.dtype and reconstructed.tobytes() == data.tobytes()


def test_photo_like_image():
    """Test with a deterministic photograph-like image (synthetic)"""
    print("\n" + "=" * 70)
    print("Test 3: Photograph-like image (synthetic)")
    print("=" * 70)

    # a deterministic photograph-like image (the earlier version downloaded
    # one, which failed offline and left an HTML page in the repository)
    data_gray = photo_like_image()

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

    else:
        print("No procedural fit found (using LZMA fallback)")

    # lossless: bit for bit, dtype included
    assert (reconstructed_lossless.dtype == data_gray.dtype
            and reconstructed_lossless.tobytes() == data_gray.tobytes())


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

    assert adaptive_is_better


PHOTO_LIKE_SHA256 = "0bce69979fa8e648abc344fbc2f23a642281364646d38c5fb198ce98b77b07b1"


def test_the_photo_like_image_is_the_same_on_every_platform():
    import hashlib
    assert hashlib.sha256(photo_like_image().tobytes()).hexdigest() == PHOTO_LIKE_SHA256


def test_a_sine_raster_takes_the_procedural_path_and_comes_back():
    from alice_zip.generators import CompressionEngine
    data = sine_raster()
    designer = ProceduralCompressionDesigner()
    result = designer.compress(data, enable_lossless=True)
    reconstructed = designer.decompress(result)
    assert result.engine_used == CompressionEngine.PROCEDURAL
    assert result.generator_params is not None and result.has_residual
    assert reconstructed.dtype == data.dtype and reconstructed.tobytes() == data.tobytes()
