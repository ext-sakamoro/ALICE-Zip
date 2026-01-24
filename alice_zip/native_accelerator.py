#!/usr/bin/env python3
"""
ALICE-Zip: Native Accelerator (MIT License)
============================================

Optional high-performance native acceleration using libalice (Rust).
Falls back to pure Python if libalice is not installed.

Author: Moroya Sakamoto
License: MIT
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import native library (try both module names for compatibility)
_HAS_LIBALICE = False
_native = None
try:
    import libalice as _native
    _HAS_LIBALICE = True
    logger.info("libalice native acceleration enabled")
except ImportError:
    # Fallback: try alice_zip module name (older builds)
    try:
        import alice_zip as _native
        _HAS_LIBALICE = True
        logger.info("alice_zip native acceleration enabled (legacy module name)")
    except ImportError:
        logger.debug("Native accelerator not available, using pure Python fallback")


def is_available() -> bool:
    """Check if native acceleration is available"""
    return _HAS_LIBALICE


# Track which fallback warnings have been shown (to avoid log flooding)
_fallback_warnings_shown = set()


def _warn_fallback(func_name: str) -> None:
    """
    Warn once per function that Python fallback is being used.

    Production deployments should install libalice for 10-100x performance.
    """
    if func_name not in _fallback_warnings_shown:
        _fallback_warnings_shown.add(func_name)
        logger.warning(
            f"{func_name}(): Using pure Python fallback (slower). "
            f"Install libalice for 10-100x performance: pip install libalice"
        )


def perlin_2d(
    width: int,
    height: int,
    seed: int = 42,
    scale: float = 10.0,
    octaves: int = 4
) -> np.ndarray:
    """
    Generate 2D Perlin noise.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility
        scale: Noise scale (larger = more zoomed out)
        octaves: Number of fBm octaves (more = more detail)

    Returns:
        2D numpy array of shape (height, width) with values in [0, 1]
    """
    if _HAS_LIBALICE and hasattr(_native, 'perlin_2d'):
        return _native.perlin_2d(width, height, seed, scale, octaves)
    else:
        _warn_fallback('perlin_2d')
        return _python_perlin_2d(width, height, seed, scale, octaves)


def perlin_advanced(
    width: int,
    height: int,
    seed: int = 42,
    scale: float = 10.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0
) -> np.ndarray:
    """
    Generate 2D Perlin noise with advanced parameters.

    Args:
        width: Image width
        height: Image height
        seed: Random seed
        scale: Noise scale
        octaves: Number of octaves
        persistence: Amplitude reduction per octave (default 0.5)
        lacunarity: Frequency increase per octave (default 2.0)

    Returns:
        2D numpy array with values in [0, 1]
    """
    if _HAS_LIBALICE and hasattr(_native, 'perlin_advanced'):
        return _native.perlin_advanced(
            width, height, seed, scale, octaves, persistence, lacunarity
        )
    else:
        _warn_fallback('perlin_advanced')
        return _python_perlin_advanced(
            width, height, seed, scale, octaves, persistence, lacunarity
        )


def fourier_generate(
    n: int,
    coefficients: List[Tuple[int, float, float]],
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Generate signal from Fourier coefficients.

    Args:
        n: Number of samples to generate
        coefficients: List of (freq_idx, magnitude, phase) tuples
        dc_offset: DC component (mean value)

    Returns:
        1D numpy array of generated signal
    """
    if _HAS_LIBALICE and hasattr(_native, 'fourier_generate'):
        return _native.fourier_generate(n, coefficients, dc_offset)
    else:
        _warn_fallback('fourier_generate')
        return _python_fourier_generate(n, coefficients, dc_offset)


def sine_wave(
    n: int,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Generate a sine wave.

    Args:
        n: Number of samples
        frequency: Frequency (cycles per n samples)
        amplitude: Wave amplitude
        phase: Initial phase in radians
        dc_offset: DC offset

    Returns:
        1D numpy array
    """
    if _HAS_LIBALICE and hasattr(_native, 'sine_wave'):
        return _native.sine_wave(n, frequency, amplitude, phase, dc_offset)
    else:
        _warn_fallback('sine_wave')
        return _python_sine_wave(n, frequency, amplitude, phase, dc_offset)


def multi_sine(
    n: int,
    components: List[Tuple[float, float, float]],
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Generate multi-tone signal (sum of sine waves).

    Args:
        n: Number of samples
        components: List of (frequency, amplitude, phase) tuples
        dc_offset: DC offset

    Returns:
        1D numpy array
    """
    if _HAS_LIBALICE and hasattr(_native, 'multi_sine'):
        return _native.multi_sine(n, components, dc_offset)
    else:
        _warn_fallback('multi_sine')
        return _python_multi_sine(n, components, dc_offset)


def fourier_analyze(
    signal: np.ndarray,
    max_coefficients: int = 20,
    energy_threshold: float = 0.99
) -> Tuple[List[Tuple[int, float, float]], float]:
    """
    Analyze signal and extract Fourier coefficients.

    Args:
        signal: Input signal as numpy array
        max_coefficients: Maximum number of coefficients to return
        energy_threshold: Stop when this fraction of energy is captured

    Returns:
        Tuple of (coefficients, dc_offset) where coefficients is a list of
        (freq_idx, magnitude, phase) tuples
    """
    if _HAS_LIBALICE and hasattr(_native, 'fourier_analyze'):
        return _native.fourier_analyze(
            signal.tolist() if isinstance(signal, np.ndarray) else list(signal),
            max_coefficients,
            energy_threshold
        )
    else:
        _warn_fallback('fourier_analyze')
        return _python_fourier_analyze(signal, max_coefficients, energy_threshold)


def polynomial_generate(n: int, coefficients: List[float]) -> np.ndarray:
    """
    Generate signal from polynomial coefficients.

    Evaluates: y = c[0] * x^n + c[1] * x^(n-1) + ... + c[n]
    where x is normalized to [0, 1]

    Args:
        n: Number of samples
        coefficients: Polynomial coefficients (highest degree first)

    Returns:
        1D numpy array
    """
    if _HAS_LIBALICE and hasattr(_native, 'polynomial_generate'):
        return _native.polynomial_generate(n, coefficients)
    else:
        _warn_fallback('polynomial_generate')
        return _python_polynomial_generate(n, coefficients)


# Maximum polynomial degree to prevent Runge phenomenon (numerical instability)
# Degrees > 5 can cause oscillation artifacts at sample boundaries
MAX_POLYNOMIAL_DEGREE = 5


def polynomial_fit(
    data: np.ndarray,
    max_degree: int = 5,
    error_threshold: float = 0.001
) -> Optional[Tuple[List[float], int, float]]:
    """
    Fit polynomial to data.

    Args:
        data: Input data as numpy array
        max_degree: Maximum polynomial degree to try (capped at 5 to prevent
                    Runge phenomenon / numerical instability)
        error_threshold: Stop if relative error is below this

    Returns:
        Tuple of (coefficients, degree, relative_error) or None if fitting fails

    Note:
        High polynomial degrees (>5) can cause numerical instability known as
        Runge's phenomenon, resulting in wild oscillations near boundaries.
        This function caps max_degree at 5 for stability.
    """
    # Cap max_degree to prevent numerical instability
    if max_degree > MAX_POLYNOMIAL_DEGREE:
        logger.warning(
            f"max_degree={max_degree} exceeds safe limit. "
            f"Capping at {MAX_POLYNOMIAL_DEGREE} to prevent Runge phenomenon."
        )
        max_degree = MAX_POLYNOMIAL_DEGREE

    if _HAS_LIBALICE and hasattr(_native, 'polynomial_fit'):
        return _native.polynomial_fit(
            data.tolist() if isinstance(data, np.ndarray) else list(data),
            max_degree,
            error_threshold
        )
    else:
        _warn_fallback('polynomial_fit')
        return _python_polynomial_fit(data, max_degree, error_threshold)


# ============================================================================
# Python Fallback Implementations
# ============================================================================

def _python_perlin_2d(
    width: int,
    height: int,
    seed: int,
    scale: float,
    octaves: int
) -> np.ndarray:
    """Pure Python Perlin noise (slow but functional)"""
    return _python_perlin_advanced(width, height, seed, scale, octaves, 0.5, 2.0)


def _python_perlin_advanced(
    width: int,
    height: int,
    seed: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float
) -> np.ndarray:
    """
    Pure Python Perlin noise with advanced parameters.

    Fully vectorized implementation using NumPy broadcasting.
    No Python loops over pixels - all operations are array-based.
    """
    # Use local RandomState to avoid polluting global state
    rng = np.random.RandomState(seed)

    # Generate permutation table (local, thread-safe)
    perm = np.arange(256, dtype=np.int32)
    rng.shuffle(perm)
    perm = np.concatenate([perm, perm])  # Double for overflow handling

    # Gradient vectors (8 directions)
    sqrt2_inv = 0.7071067811865476  # 1/sqrt(2)
    grad2 = np.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [sqrt2_inv, sqrt2_inv], [-sqrt2_inv, sqrt2_inv],
        [sqrt2_inv, -sqrt2_inv], [-sqrt2_inv, -sqrt2_inv]
    ], dtype=np.float32)

    # Create coordinate grids (vectorized)
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords)

    result = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        # Scale coordinates for this octave
        NX = X / scale * frequency
        NY = Y / scale * frequency

        # Integer grid coordinates (wrapped to 0-255)
        XI = NX.astype(np.int32) & 255
        YI = NY.astype(np.int32) & 255

        # Fractional parts
        XF = NX - np.floor(NX)
        YF = NY - np.floor(NY)

        # Fade curves (quintic smoothstep) - vectorized
        U = XF * XF * XF * (XF * (XF * 6.0 - 15.0) + 10.0)
        V = YF * YF * YF * (YF * (YF * 6.0 - 15.0) + 10.0)

        # Hash coordinates to get gradient indices
        # perm[perm[xi] + yi] & 7 for each corner
        XI1 = (XI + 1) & 255
        YI1 = (YI + 1) & 255

        AA = perm[perm[XI] + YI] & 7
        AB = perm[perm[XI] + YI1] & 7
        BA = perm[perm[XI1] + YI] & 7
        BB = perm[perm[XI1] + YI1] & 7

        # Compute dot products with gradients (vectorized)
        # g00 = grad[AA] · (xf, yf)
        G00 = grad2[AA, 0] * XF + grad2[AA, 1] * YF
        # g10 = grad[BA] · (xf-1, yf)
        G10 = grad2[BA, 0] * (XF - 1.0) + grad2[BA, 1] * YF
        # g01 = grad[AB] · (xf, yf-1)
        G01 = grad2[AB, 0] * XF + grad2[AB, 1] * (YF - 1.0)
        # g11 = grad[BB] · (xf-1, yf-1)
        G11 = grad2[BB, 0] * (XF - 1.0) + grad2[BB, 1] * (YF - 1.0)

        # Bilinear interpolation (vectorized)
        X0 = G00 + U * (G10 - G00)  # lerp(g00, g10, u)
        X1 = G01 + U * (G11 - G01)  # lerp(g01, g11, u)
        noise = X0 + V * (X1 - X0)  # lerp(x0, x1, v)

        result += noise * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    result = (result / max_value + 1.0) * 0.5
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _python_fourier_generate(
    n: int,
    coefficients: List[Tuple[int, float, float]],
    dc_offset: float
) -> np.ndarray:
    """Pure Python Fourier generation"""
    fft_buffer = np.zeros(n, dtype=complex)

    for freq_idx, magnitude, phase in coefficients:
        if freq_idx < n:
            fft_buffer[freq_idx] = magnitude * np.exp(1j * phase)
            if freq_idx > 0 and freq_idx < n // 2:
                fft_buffer[n - freq_idx] = magnitude * np.exp(-1j * phase)

    result = np.fft.ifft(fft_buffer).real + dc_offset
    return result.astype(np.float32)


def _python_sine_wave(
    n: int,
    frequency: float,
    amplitude: float,
    phase: float,
    dc_offset: float
) -> np.ndarray:
    """Pure Python sine wave generation"""
    # Use arange/n for FFT compatibility (excludes endpoint)
    x = np.arange(n, dtype=np.float64) / n
    result = dc_offset + amplitude * np.sin(2.0 * np.pi * frequency * x + phase)
    return result.astype(np.float32)


def _python_multi_sine(
    n: int,
    components: List[Tuple[float, float, float]],
    dc_offset: float
) -> np.ndarray:
    """Pure Python multi-sine generation"""
    # Use arange/n for FFT compatibility (excludes endpoint)
    x = np.arange(n, dtype=np.float64) / n
    result = np.full(n, dc_offset, dtype=np.float64)

    for freq, amp, phase in components:
        result += amp * np.sin(2.0 * np.pi * freq * x + phase)

    return result.astype(np.float32)


def _python_fourier_analyze(
    signal: np.ndarray,
    max_coefficients: int,
    energy_threshold: float
) -> Tuple[List[Tuple[int, float, float]], float]:
    """Pure Python Fourier analysis"""
    signal = np.asarray(signal, dtype=np.float32)
    n = len(signal)
    dc_offset = float(np.mean(signal))

    # FFT
    fft_result = np.fft.fft(signal - dc_offset)
    half_n = n // 2

    # Extract magnitudes and phases
    freq_data = []
    for i in range(half_n):
        mag = np.abs(fft_result[i])
        phase = np.angle(fft_result[i])
        freq_data.append((i, float(mag), float(phase)))

    # Sort by magnitude
    freq_data.sort(key=lambda x: -x[1])

    # Calculate total energy
    total_energy = sum(m * m for _, m, _ in freq_data)

    # Select coefficients
    coefficients = []
    captured_energy = 0.0

    for idx, mag, phase in freq_data:
        if len(coefficients) >= max_coefficients:
            break
        if mag < 1e-10:
            continue

        coefficients.append((idx, mag, phase))
        captured_energy += mag * mag

        if total_energy > 0 and captured_energy / total_energy > energy_threshold:
            break

    return coefficients, dc_offset


def _python_polynomial_generate(n: int, coefficients: List[float]) -> np.ndarray:
    """Pure Python polynomial generation"""
    if not coefficients:
        return np.zeros(n, dtype=np.float32)

    x = np.linspace(0, 1, n, dtype=np.float64)
    result = np.polyval(coefficients, x)
    return result.astype(np.float32)


def _python_polynomial_fit(
    data: np.ndarray,
    max_degree: int,
    error_threshold: float
) -> Optional[Tuple[List[float], int, float]]:
    """
    Pure Python polynomial fitting with numerical stability safeguards.

    Uses numpy.polyfit with rcond parameter to handle ill-conditioned matrices.
    Degree is capped to prevent Runge phenomenon.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n < 2:
        return None

    # Apply safety cap to prevent Runge phenomenon
    max_degree = min(max_degree, MAX_POLYNOMIAL_DEGREE, n - 1)

    x = np.linspace(0, 1, n)
    variance = np.var(data)

    if variance < 1e-20:
        return ([float(np.mean(data))], 0, 0.0)

    for degree in range(1, max_degree + 1):
        try:
            # Use rcond to handle ill-conditioned Vandermonde matrices
            # This prevents numerical instability warnings
            coeffs = np.polyfit(x, data, degree, rcond=None)
            fitted = np.polyval(coeffs, x)
            mse = np.mean((data - fitted) ** 2)
            relative_error = mse / variance

            # Check for NaN/Inf in coefficients (numerical instability)
            if not np.all(np.isfinite(coeffs)):
                continue

            if relative_error < error_threshold:
                return (coeffs.tolist(), degree, float(relative_error))
        except (np.linalg.LinAlgError, ValueError):
            # Skip this degree if fitting fails
            continue

    return None
