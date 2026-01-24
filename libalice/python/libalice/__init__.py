"""
libalice - High-Performance Procedural Generation Library

Native Rust implementation for maximum performance.

Functions:
    perlin_2d: Generate 2D Perlin noise texture
    perlin_advanced: Generate 2D Perlin noise with advanced parameters
    fourier_generate: Generate signal from Fourier coefficients
    sine_wave: Generate a sine wave
    multi_sine: Generate multi-tone signal
    fourier_analyze: Analyze signal and extract Fourier coefficients
    polynomial_generate: Generate signal from polynomial coefficients
    polynomial_fit: Fit polynomial to data

License: MIT
Author: Moroya Sakamoto
"""

from ._libalice import (
    perlin_2d,
    perlin_advanced,
    fourier_generate,
    sine_wave,
    multi_sine,
    fourier_analyze,
    polynomial_generate,
    polynomial_fit,
)

__version__ = "0.1.0"
__all__ = [
    "perlin_2d",
    "perlin_advanced",
    "fourier_generate",
    "sine_wave",
    "multi_sine",
    "fourier_analyze",
    "polynomial_generate",
    "polynomial_fit",
]
