#!/usr/bin/env python3
"""
Simple 8-bit Quantization Test (1D data only, no Perlin)
"""

import numpy as np
import pytest
import sys
from pathlib import Path
# Resolve the repository root from this file so the tests run anywhere
# (an absolute home path only works on the machine it was written on,
# and this is a public repository).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alice_zip.analyzers import ProceduralCompressionDesigner
from alice_zip.generators import CompressionEngine


def psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    rng = float(orig.max() - orig.min())
    return 20 * np.log10(rng / np.sqrt(mse)) if rng > 0 else float('inf')


def compress_both(data):
    """The input compressed losslessly and with an 8-bit residual."""
    d32 = ProceduralCompressionDesigner()
    r32 = d32.compress(data, enable_lossless=True)
    d8 = ProceduralCompressionDesigner()
    r8 = d8.compress(data, enable_lossless=True, quantize_residual=8)
    return np.asarray(d32.decompress(r32)), np.asarray(d8.decompress(r8)), r8


def cases():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 2000)
    t1 = np.linspace(0, 10 * np.pi, 3000)
    t2 = np.linspace(0, 2 * np.pi, 2000)
    return {
        "polynomial_noise": (x ** 2 + rng.normal(0, 10, 2000)).astype(np.float32),
        "sine_noise": (np.sin(t1) * 100 + rng.normal(0, 5, 3000)).astype(np.float32),
        "multi_frequency_noise": (50 * np.sin(3 * t2) + 30 * np.sin(7 * t2)
                                  + rng.normal(0, 5, 2000)).astype(np.float32),
    }


# the engine the designer picks for each input: the noisy polynomial and
# multi-frequency signals fall back to LZMA (exact even with quantize_residual)
ENGINE = {
    "polynomial_noise": CompressionEngine.FALLBACK_LZMA,
    "sine_noise": CompressionEngine.PROCEDURAL,
    "multi_frequency_noise": CompressionEngine.FALLBACK_LZMA,
}


@pytest.mark.parametrize("name", list(cases()))
def test_the_lossless_result_is_exact_and_the_8_bit_one_is_close(name):
    data = cases()[name]
    rec32, rec8, r8 = compress_both(data)
    assert rec32.dtype == data.dtype and rec32.tobytes() == data.tobytes()
    assert r8.engine_used == ENGINE[name]
    if r8.engine_used == CompressionEngine.PROCEDURAL:
        # an 8-bit residual is lossy but close
        assert not r8.is_lossless
        assert psnr(data, rec8) > 30
    else:
        assert r8.is_lossless and rec8.tobytes() == data.tobytes()
