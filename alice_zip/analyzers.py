#!/usr/bin/env python3
"""
ALICE-Zip: Basic Analyzers (MIT License)
=========================================

Basic data analysis and fitting functions for compression.
This module provides free-tier compression capabilities.

Supported patterns:
- Sine wave detection and fitting
- Fourier series fitting (multi-frequency)
- Polynomial fitting (up to degree 5)
- LZMA fallback for non-procedural data

Author: Moroya Sakamoto
License: MIT
"""

import logging
import lzma
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from .generators import (
    GeneratorType,
    GeneratorParameters,
    CompressionResult,
    CompressionEngine,
)
from . import native_accelerator as accel

logger = logging.getLogger(__name__)

# Maximum polynomial degree for free version
MAX_POLYNOMIAL_DEGREE = 5

# Minimum compression ratio to accept procedural fit (otherwise use LZMA)
MIN_PROCEDURAL_RATIO = 1.5

# Maximum acceptable error for procedural compression (normalized)
MAX_ACCEPTABLE_ERROR = 0.01


@dataclass
class FitResult:
    """Result of fitting attempt"""
    success: bool
    generator_type: Optional[GeneratorType]
    parameters: dict
    error: float
    compressed_size: int


def analyze_data(data: np.ndarray) -> CompressionResult:
    """
    Analyze data and find the best compression method.

    Tries procedural generators in order of likelihood, falls back to LZMA.

    Args:
        data: Input data as numpy array

    Returns:
        CompressionResult with best compression method
    """
    original_size = data.nbytes
    data_flat = data.flatten().astype(np.float64)

    best_result = None
    best_ratio = 0.0

    # Try sine wave fitting first (common pattern)
    sine_result = try_sine_fit(data_flat)
    if sine_result.success:
        ratio = original_size / sine_result.compressed_size
        if ratio > best_ratio and sine_result.error < MAX_ACCEPTABLE_ERROR:
            best_ratio = ratio
            best_result = sine_result
            logger.debug(f"Sine fit: ratio={ratio:.2f}, error={sine_result.error:.6f}")

    # Try Fourier fitting (multi-frequency)
    fourier_result = try_fourier_fit(data_flat)
    if fourier_result.success:
        ratio = original_size / fourier_result.compressed_size
        if ratio > best_ratio and fourier_result.error < MAX_ACCEPTABLE_ERROR:
            best_ratio = ratio
            best_result = fourier_result
            logger.debug(f"Fourier fit: ratio={ratio:.2f}, error={fourier_result.error:.6f}")

    # Try polynomial fitting
    poly_result = try_polynomial_fit(data_flat)
    if poly_result.success:
        ratio = original_size / poly_result.compressed_size
        if ratio > best_ratio and poly_result.error < MAX_ACCEPTABLE_ERROR:
            best_ratio = ratio
            best_result = poly_result
            logger.debug(f"Polynomial fit: ratio={ratio:.2f}, error={poly_result.error:.6f}")

    # If procedural compression is good enough, use it
    if best_result and best_ratio >= MIN_PROCEDURAL_RATIO:
        params = GeneratorParameters(
            generator_type=best_result.generator_type,
            seed=0,
            parameters=best_result.parameters,
            output_shape=data.shape,
            dtype=str(data.dtype)
        )
        return CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=original_size,
            compressed_size=best_result.compressed_size,
            compression_ratio=best_ratio,
            generator_params=params,
            error_metric=best_result.error,
            is_lossless=best_result.error < 1e-10
        )

    # Fall back to LZMA
    return compress_with_lzma(data)


def try_sine_fit(data: np.ndarray) -> FitResult:
    """
    Try to fit data to a sine wave pattern.

    Model: y = amplitude * sin(2π * frequency * x + phase) + dc_offset

    Args:
        data: 1D array of data points

    Returns:
        FitResult with fitting results
    """
    n = len(data)
    if n < 4:
        return FitResult(False, None, {}, float('inf'), 0)

    try:
        # Estimate DC offset
        dc_offset = np.mean(data)
        centered = data - dc_offset

        # Estimate amplitude
        amplitude = (np.max(centered) - np.min(centered)) / 2
        if amplitude < 1e-10:
            return FitResult(False, None, {}, float('inf'), 0)

        # Estimate frequency using zero crossings
        crossings = np.where(np.diff(np.signbit(centered)))[0]
        if len(crossings) >= 2:
            avg_half_period = np.mean(np.diff(crossings))
            # frequency = cycles per x∈[0,1] interval = n_samples / samples_per_cycle
            frequency = n / (2 * avg_half_period) if avg_half_period > 0 else 1.0
        else:
            # Use FFT for frequency estimation
            fft = np.fft.rfft(centered)
            freqs = np.fft.rfftfreq(n)
            peak_idx = np.argmax(np.abs(fft[1:])) + 1
            frequency = freqs[peak_idx] * n

        # Estimate phase (use arange/n for FFT compatibility)
        x = np.arange(n) / n

        # Grid search for best phase (360 steps for 1-degree resolution)
        best_phase = 0.0
        best_error = float('inf')
        for phase in np.linspace(0, 2 * np.pi, 360):
            fitted = amplitude * np.sin(2 * np.pi * frequency * x + phase) + dc_offset
            error = np.mean((data - fitted) ** 2)
            if error < best_error:
                best_error = error
                best_phase = phase

        # Refine with normalized error
        fitted = amplitude * np.sin(2 * np.pi * frequency * x + best_phase) + dc_offset
        mse = np.mean((data - fitted) ** 2)
        data_var = np.var(data)
        normalized_error = mse / data_var if data_var > 0 else mse

        # Calculate compressed size (parameters only)
        # 4 floats * 8 bytes = 32 bytes + overhead
        compressed_size = 64

        if normalized_error < MAX_ACCEPTABLE_ERROR:
            return FitResult(
                success=True,
                generator_type=GeneratorType.WAVE_FUNCTION,
                parameters={
                    'frequency': float(frequency),
                    'amplitude': float(amplitude),
                    'phase': float(best_phase),
                    'dc_offset': float(dc_offset),
                    'is_simple_sine': True
                },
                error=normalized_error,
                compressed_size=compressed_size
            )
    except Exception as e:
        logger.debug(f"Sine fit failed: {e}")

    return FitResult(False, None, {}, float('inf'), 0)


# Maximum Fourier coefficients for compression
MAX_FOURIER_COEFFICIENTS = 20

# Energy threshold for Fourier fitting (capture this fraction of total energy)
FOURIER_ENERGY_THRESHOLD = 0.99


def try_fourier_fit(data: np.ndarray,
                    max_coefficients: int = MAX_FOURIER_COEFFICIENTS,
                    energy_threshold: float = FOURIER_ENERGY_THRESHOLD) -> FitResult:
    """
    Try to fit data to a Fourier series (sum of sine waves).

    Model: y = dc_offset + Σ(magnitude_i * cos(2π * freq_i * t + phase_i))

    This handles multi-frequency signals that single sine wave fitting cannot.

    Args:
        data: 1D array of data points
        max_coefficients: Maximum number of Fourier coefficients to use
        energy_threshold: Stop when this fraction of signal energy is captured

    Returns:
        FitResult with fitting results
    """
    n = len(data)
    if n < 8:
        return FitResult(False, None, {}, float('inf'), 0)

    try:
        # Use native accelerator for Fourier analysis
        coefficients, dc_offset = accel.fourier_analyze(
            data, max_coefficients, energy_threshold
        )

        if not coefficients:
            return FitResult(False, None, {}, float('inf'), 0)

        # Reconstruct signal to check error
        reconstructed = accel.fourier_generate(n, coefficients, dc_offset)

        # Calculate normalized error
        mse = np.mean((data - reconstructed) ** 2)
        data_var = np.var(data)
        normalized_error = mse / data_var if data_var > 0 else mse

        # Calculate compressed size
        # Each coefficient: (freq_idx: 4 bytes, magnitude: 8 bytes, phase: 8 bytes) = 20 bytes
        # Plus dc_offset: 8 bytes, plus overhead: ~32 bytes
        compressed_size = len(coefficients) * 20 + 8 + 32

        if normalized_error < MAX_ACCEPTABLE_ERROR:
            return FitResult(
                success=True,
                generator_type=GeneratorType.FOURIER,
                parameters={
                    'coefficients': coefficients,
                    'dc_offset': float(dc_offset)
                },
                error=normalized_error,
                compressed_size=compressed_size
            )

    except Exception as e:
        logger.debug(f"Fourier fit failed: {e}")

    return FitResult(False, None, {}, float('inf'), 0)


def try_polynomial_fit(data: np.ndarray, max_degree: int = MAX_POLYNOMIAL_DEGREE) -> FitResult:
    """
    Try to fit data to a polynomial.

    Model: y = c0 + c1*x + c2*x^2 + ... + cn*x^n

    Args:
        data: 1D array of data points
        max_degree: Maximum polynomial degree to try

    Returns:
        FitResult with fitting results
    """
    n = len(data)
    if n < 2:
        return FitResult(False, None, {}, float('inf'), 0)

    x = np.linspace(0, 1, n)
    data_var = np.var(data)

    best_result = FitResult(False, None, {}, float('inf'), 0)

    # Try increasing degrees, stop when error is good enough
    for degree in range(1, min(max_degree + 1, n)):
        try:
            # Fit polynomial
            coeffs = np.polyfit(x, data, degree)
            fitted = np.polyval(coeffs, x)

            # Calculate error
            mse = np.mean((data - fitted) ** 2)
            normalized_error = mse / data_var if data_var > 0 else mse

            # Compressed size: coefficients as float64
            compressed_size = (degree + 1) * 8 + 32  # coeffs + overhead

            if normalized_error < best_result.error:
                best_result = FitResult(
                    success=True,
                    generator_type=GeneratorType.POLYNOMIAL,
                    parameters={
                        'coefficients': coeffs.tolist()  # Highest degree first (np.polyfit format)
                    },
                    error=normalized_error,
                    compressed_size=compressed_size
                )

            # If error is very small, no need to try higher degrees
            if normalized_error < 1e-10:
                break

        except Exception as e:
            logger.debug(f"Polynomial fit degree {degree} failed: {e}")
            continue

    return best_result


def compress_with_lzma(data: np.ndarray, preset: int = 6) -> CompressionResult:
    """
    Compress data using LZMA (fallback method).

    Args:
        data: Input data as numpy array
        preset: LZMA compression preset (0-9)

    Returns:
        CompressionResult with LZMA compression
    """
    original_size = data.nbytes
    raw_bytes = data.tobytes()

    compressed = lzma.compress(raw_bytes, preset=preset)
    compressed_size = len(compressed)

    return CompressionResult(
        success=True,
        engine_used=CompressionEngine.FALLBACK_LZMA,
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
        compressed_data=compressed,
        error_metric=0.0,
        is_lossless=True,
        metadata={
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    )


# ============================================================================
# ProceduralCompressionDesigner - High-level Compression API
# ============================================================================

class ProceduralCompressionDesigner:
    """
    High-level API for procedural compression.

    This class provides a simple interface for compressing and decompressing
    data using ALICE-Zip's procedural generation approach.

    Example:
        designer = ProceduralCompressionDesigner()
        result = designer.compress(data, enable_lossless=True)
        reconstructed = designer.decompress(result)
    """

    def __init__(self, max_error: float = MAX_ACCEPTABLE_ERROR):
        """
        Initialize the compression designer.

        Args:
            max_error: Maximum acceptable normalized error for procedural fit
        """
        self.max_error = max_error

    def compress(self, data: np.ndarray, enable_lossless: bool = False,
                 adaptive_fallback: bool = True,
                 quantize_residual: Optional[int] = None) -> CompressionResult:
        """
        Compress data using the best available method.

        Args:
            data: Input data as numpy array
            enable_lossless: If True, store residual for exact reconstruction
            adaptive_fallback: If True, automatically fall back to LZMA for
                             incompressible data (default: True)
            quantize_residual: If set, quantize residual to N bits (8 or 16)
                             for better compression at cost of precision

        Returns:
            CompressionResult with compression details
        """
        # Ensure data is numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Store original shape and dtype for reconstruction
        original_shape = data.shape
        original_dtype = data.dtype

        # Analyze and compress
        result = analyze_data(data)

        # Store shape/dtype in metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata['shape'] = original_shape
        result.metadata['dtype'] = str(original_dtype)

        # If lossless is requested and we used procedural compression
        if enable_lossless and result.engine_used == CompressionEngine.PROCEDURAL:
            # Regenerate from parameters
            from .generators import decompress_from_params
            regenerated = decompress_from_params(result.generator_params)
            regenerated = regenerated.reshape(original_shape)

            # Calculate residual
            residual = data.astype(np.float64) - regenerated.astype(np.float64)

            # Check if residual is significant
            if np.max(np.abs(residual)) > 1e-10:
                # Quantize residual if requested
                if quantize_residual is not None:
                    # Normalize residual to [0, 1] range
                    r_min, r_max = residual.min(), residual.max()
                    r_range = r_max - r_min if r_max > r_min else 1.0
                    normalized = (residual - r_min) / r_range

                    if quantize_residual == 8:
                        quantized = (normalized * 255).astype(np.uint8)
                        residual_bytes = quantized.tobytes()
                    elif quantize_residual == 16:
                        quantized = (normalized * 65535).astype(np.uint16)
                        residual_bytes = quantized.tobytes()
                    else:
                        residual_bytes = residual.astype(np.float32).tobytes()

                    # Store quantization metadata
                    result.metadata['residual_quantized'] = quantize_residual
                    result.metadata['residual_min'] = float(r_min)
                    result.metadata['residual_max'] = float(r_max)
                    result.is_lossless = False  # Quantization is lossy
                else:
                    residual_bytes = residual.astype(np.float32).tobytes()
                    result.is_lossless = True

                compressed_residual = lzma.compress(residual_bytes, preset=6)

                # Update result
                result.residual_data = compressed_residual
                result.error_metric = 0.0
            else:
                # No significant residual - already lossless
                result.is_lossless = True
                result.error_metric = 0.0

        return result

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """
        Decompress data from a CompressionResult.

        Args:
            result: CompressionResult from compress()

        Returns:
            Reconstructed numpy array
        """
        from .generators import decompress_from_params, decompress_from_lzma

        # Get shape and dtype from metadata
        shape = result.metadata.get('shape') if result.metadata else None
        dtype = result.metadata.get('dtype', 'float32') if result.metadata else 'float32'

        if result.engine_used == CompressionEngine.PROCEDURAL:
            # Regenerate from parameters
            data = decompress_from_params(result.generator_params)

            # Reshape if needed
            if shape is not None:
                data = data.reshape(shape)

            # Add residual if present (for lossless/near-lossless reconstruction)
            if result.has_residual:
                residual_bytes = lzma.decompress(result.residual_data)

                # Check if residual was quantized
                quant_bits = result.metadata.get('residual_quantized') if result.metadata else None
                if quant_bits == 8:
                    quantized = np.frombuffer(residual_bytes, dtype=np.uint8).reshape(shape)
                    r_min = result.metadata.get('residual_min', 0.0)
                    r_max = result.metadata.get('residual_max', 1.0)
                    r_range = r_max - r_min if r_max > r_min else 1.0
                    residual = (quantized.astype(np.float64) / 255.0) * r_range + r_min
                elif quant_bits == 16:
                    quantized = np.frombuffer(residual_bytes, dtype=np.uint16).reshape(shape)
                    r_min = result.metadata.get('residual_min', 0.0)
                    r_max = result.metadata.get('residual_max', 1.0)
                    r_range = r_max - r_min if r_max > r_min else 1.0
                    residual = (quantized.astype(np.float64) / 65535.0) * r_range + r_min
                else:
                    residual = np.frombuffer(residual_bytes, dtype=np.float32).reshape(shape)

                data = data + residual

        elif result.engine_used == CompressionEngine.FALLBACK_LZMA:
            # Decompress LZMA data
            if shape is None:
                raise ValueError("Shape not found in metadata for LZMA decompression")
            data = decompress_from_lzma(result.compressed_data, shape, dtype)

        else:
            raise ValueError(f"Unknown compression engine: {result.engine_used}")

        # Convert to original dtype
        return data.astype(dtype)
