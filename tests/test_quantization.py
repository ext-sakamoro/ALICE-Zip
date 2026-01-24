#!/usr/bin/env python3
"""
Test 8-bit Residual Quantization

Compares:
- 32-bit (TRUE LOSSLESS): Full precision residual
- 8-bit (NEAR-LOSSLESS): Quantized residual, ~75% size reduction

Expected outcome:
- 8-bit achieves ~4x smaller residual with PSNR > 50dB (imperceptible difference)
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


def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def test_polynomial_with_noise():
    """Test quantization on polynomial + noise data"""
    print("=" * 70)
    print("Test 1: Polynomial + Noise (2000 samples)")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(0, 10, 2000)
    signal = 0.5 * x**3 - 2 * x**2 + 3 * x + 10
    noise = np.random.randn(2000) * 5
    data = (signal + noise).astype(np.float32)

    print(f"Original size: {data.nbytes:,} bytes")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    # 32-bit (TRUE LOSSLESS)
    print("\n--- 32-bit (TRUE LOSSLESS) ---")
    designer_32 = ProceduralCompressionDesigner()
    result_32 = designer_32.compress(data, enable_lossless=True, quantize_residual=None)
    recon_32 = designer_32.decompress(result_32)

    print(f"Engine: {result_32.engine_used.value}")
    print(f"Params: {result_32.compressed_size:,} bytes")
    print(f"Residual: {result_32.metadata.get('residual_size', 0):,} bytes")
    print(f"Total: {result_32.total_compressed_size:,} bytes")
    print(f"Ratio: {result_32.effective_ratio:.2f}x")
    print(f"Is lossless: {result_32.is_lossless}")
    print(f"MSE: {calculate_mse(data, recon_32):.2e}")
    print(f"PSNR: {calculate_psnr(data, recon_32):.2f} dB")

    # 8-bit (NEAR-LOSSLESS)
    print("\n--- 8-bit (NEAR-LOSSLESS) ---")
    designer_8 = ProceduralCompressionDesigner()
    result_8 = designer_8.compress(data, enable_lossless=True, quantize_residual=8)
    recon_8 = designer_8.decompress(result_8)

    print(f"Engine: {result_8.engine_used.value}")
    print(f"Params: {result_8.compressed_size:,} bytes")
    print(f"Residual: {result_8.metadata.get('residual_size', 0):,} bytes")
    print(f"Total: {result_8.total_compressed_size:,} bytes")
    print(f"Ratio: {result_8.effective_ratio:.2f}x")
    print(f"Is lossless: {result_8.is_lossless}")
    print(f"MSE: {calculate_mse(data, recon_8):.2e}")
    print(f"PSNR: {calculate_psnr(data, recon_8):.2f} dB")

    # Comparison
    print("\n--- COMPARISON ---")
    residual_32 = result_32.metadata.get('residual_size', 0)
    residual_8 = result_8.metadata.get('residual_size', 0)
    if residual_32 > 0:
        reduction = (residual_32 - residual_8) / residual_32 * 100
        print(f"Residual size reduction: {reduction:.1f}%")
    print(f"Total size reduction: {(result_32.total_compressed_size - result_8.total_compressed_size) / result_32.total_compressed_size * 100:.1f}%")
    print(f"Quality loss (PSNR): {calculate_psnr(data, recon_32) - calculate_psnr(data, recon_8):.2f} dB")

    return calculate_psnr(data, recon_8) > 40  # Should be high quality


def test_texture_pattern():
    """Test quantization on 2D texture pattern"""
    print("\n" + "=" * 70)
    print("Test 2: 2D Texture Pattern (256x256)")
    print("=" * 70)

    np.random.seed(123)
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    pattern = np.sin(xx) * np.cos(yy) + np.random.randn(256, 256) * 0.1
    data = ((pattern + 1.5) * 80).astype(np.float32)  # Scale to ~[0, 255]

    print(f"Original size: {data.nbytes:,} bytes")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    # 32-bit
    print("\n--- 32-bit (TRUE LOSSLESS) ---")
    designer_32 = ProceduralCompressionDesigner()
    result_32 = designer_32.compress(data, enable_lossless=True, quantize_residual=None)
    recon_32 = designer_32.decompress(result_32)

    print(f"Total: {result_32.total_compressed_size:,} bytes")
    print(f"Ratio: {result_32.effective_ratio:.2f}x")
    print(f"PSNR: {calculate_psnr(data, recon_32):.2f} dB")

    # 8-bit
    print("\n--- 8-bit (NEAR-LOSSLESS) ---")
    designer_8 = ProceduralCompressionDesigner()
    result_8 = designer_8.compress(data, enable_lossless=True, quantize_residual=8)
    recon_8 = designer_8.decompress(result_8)

    print(f"Total: {result_8.total_compressed_size:,} bytes")
    print(f"Ratio: {result_8.effective_ratio:.2f}x")
    print(f"PSNR: {calculate_psnr(data, recon_8):.2f} dB")

    # Comparison
    print("\n--- COMPARISON ---")
    print(f"Size reduction: {(result_32.total_compressed_size - result_8.total_compressed_size) / result_32.total_compressed_size * 100:.1f}%")
    print(f"Quality: PSNR={calculate_psnr(data, recon_8):.1f}dB (>40dB is excellent)")

    return calculate_psnr(data, recon_8) > 40


def test_audio_signal():
    """Test quantization on audio-like signal"""
    print("\n" + "=" * 70)
    print("Test 3: Audio-like Signal (44100 samples @ 1sec)")
    print("=" * 70)

    np.random.seed(456)
    n = 44100  # 1 second at 44.1kHz
    t = np.linspace(0, 1, n)

    # Complex audio signal: fundamental + harmonics + noise
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 (440Hz)
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 (harmonic)
        0.15 * np.sin(2 * np.pi * 1320 * t) +  # E6 (harmonic)
        np.random.randn(n) * 0.02  # Noise floor
    )
    data = (signal * 32767).astype(np.float32)  # 16-bit audio range

    print(f"Original size: {data.nbytes:,} bytes")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    # 32-bit
    print("\n--- 32-bit (TRUE LOSSLESS) ---")
    designer_32 = ProceduralCompressionDesigner()
    result_32 = designer_32.compress(data, enable_lossless=True, quantize_residual=None)
    recon_32 = designer_32.decompress(result_32)

    print(f"Total: {result_32.total_compressed_size:,} bytes")
    print(f"Ratio: {result_32.effective_ratio:.2f}x")
    print(f"PSNR: {calculate_psnr(data, recon_32):.2f} dB")

    # 8-bit
    print("\n--- 8-bit (NEAR-LOSSLESS) ---")
    designer_8 = ProceduralCompressionDesigner()
    result_8 = designer_8.compress(data, enable_lossless=True, quantize_residual=8)
    recon_8 = designer_8.decompress(result_8)

    print(f"Total: {result_8.total_compressed_size:,} bytes")
    print(f"Ratio: {result_8.effective_ratio:.2f}x")
    print(f"PSNR: {calculate_psnr(data, recon_8):.2f} dB")

    # 16-bit (compromise)
    print("\n--- 16-bit (HIGH-QUALITY) ---")
    designer_16 = ProceduralCompressionDesigner()
    result_16 = designer_16.compress(data, enable_lossless=True, quantize_residual=16)
    recon_16 = designer_16.decompress(result_16)

    print(f"Total: {result_16.total_compressed_size:,} bytes")
    print(f"Ratio: {result_16.effective_ratio:.2f}x")
    print(f"PSNR: {calculate_psnr(data, recon_16):.2f} dB")

    # Summary
    print("\n--- SUMMARY ---")
    print(f"32-bit: {result_32.total_compressed_size:,} bytes, PSNR=inf")
    print(f"16-bit: {result_16.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, recon_16):.1f}dB")
    print(f"8-bit:  {result_8.total_compressed_size:,} bytes, PSNR={calculate_psnr(data, recon_8):.1f}dB")

    return calculate_psnr(data, recon_8) > 30  # Audio needs > 30dB


def test_comparison_table():
    """Generate comprehensive comparison table"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 70)

    np.random.seed(789)

    # Generate test data
    test_cases = [
        ("Polynomial+Noise (1D)", (np.linspace(0, 10, 1000)**2 + np.random.randn(1000) * 5).astype(np.float32)),
        ("Sine+Noise (1D)", (np.sin(np.linspace(0, 20*np.pi, 2000)) * 100 + np.random.randn(2000) * 10).astype(np.float32)),
        ("Texture (2D)", ((np.sin(np.linspace(0, 8*np.pi, 128).reshape(-1,1)) * np.cos(np.linspace(0, 8*np.pi, 128)) + np.random.randn(128, 128) * 0.1) * 100).astype(np.float32)),
    ]

    print(f"\n{'Data Type':<25} {'32bit':<12} {'16bit':<12} {'8bit':<12} {'8bit PSNR':<12}")
    print("-" * 75)

    all_passed = True
    for name, data in test_cases:
        designer = ProceduralCompressionDesigner()

        # 32-bit
        r32 = designer.compress(data, enable_lossless=True, quantize_residual=None)

        # 16-bit
        designer = ProceduralCompressionDesigner()
        r16 = designer.compress(data, enable_lossless=True, quantize_residual=16)
        recon16 = designer.decompress(r16)

        # 8-bit
        designer = ProceduralCompressionDesigner()
        r8 = designer.compress(data, enable_lossless=True, quantize_residual=8)
        recon8 = designer.decompress(r8)
        psnr8 = calculate_psnr(data, recon8)

        size_32 = f"{r32.total_compressed_size:,}B"
        size_16 = f"{r16.total_compressed_size:,}B"
        size_8 = f"{r8.total_compressed_size:,}B"
        psnr_str = f"{psnr8:.1f}dB" if psnr8 < float('inf') else "inf"

        print(f"{name:<25} {size_32:<12} {size_16:<12} {size_8:<12} {psnr_str:<12}")

        if psnr8 < 30:
            all_passed = False

    print("-" * 75)
    print("\nNote: PSNR > 40dB = Excellent, > 30dB = Good, > 20dB = Acceptable")

    return all_passed


if __name__ == "__main__":
    print("ALICE-Zip 8-bit Residual Quantization Test")
    print("=" * 70)
    print("Comparing TRUE LOSSLESS (32-bit) vs NEAR-LOSSLESS (8-bit)\n")

    results = []
    results.append(("Polynomial+Noise", test_polynomial_with_noise()))
    results.append(("Texture Pattern", test_texture_pattern()))
    results.append(("Audio Signal", test_audio_signal()))
    results.append(("Comparison Table", test_comparison_table()))

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
        print("8-bit quantization achieves excellent compression with minimal quality loss!")
        print("Recommended for: Images, audio, sensor data, any signal with noise floor")
    else:
        print("Some quality degradation detected - review PSNR values")

    sys.exit(0 if all_passed else 1)
