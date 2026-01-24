#!/usr/bin/env python3
"""
ALICE-Zip Native Accelerator Test Suite
=========================================

Tests for native accelerator functions (both Rust and Python fallback).
These tests verify the Python fallback implementations work correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alice_zip.native_accelerator import (
    is_available,
    perlin_2d,
    perlin_advanced,
    fourier_generate,
    sine_wave,
    multi_sine,
    fourier_analyze,
    polynomial_generate,
    polynomial_fit,
    # Python fallback implementations (for direct testing)
    _python_perlin_2d,
    _python_perlin_advanced,
    _python_fourier_generate,
    _python_sine_wave,
    _python_multi_sine,
    _python_fourier_analyze,
    _python_polynomial_generate,
    _python_polynomial_fit,
)


class TestIsAvailable:
    """Test native library availability check"""

    def test_is_available_returns_bool(self):
        """Test that is_available returns a boolean"""
        result = is_available()
        assert isinstance(result, bool)


class TestPerlin2D:
    """Test 2D Perlin noise generation"""

    def test_basic_generation(self):
        """Test basic Perlin noise generation"""
        result = perlin_2d(64, 64, seed=42, scale=10.0, octaves=4)

        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0.0 <= result.min()
        assert result.max() <= 1.0

    def test_reproducibility(self):
        """Test same seed produces same output"""
        result1 = perlin_2d(32, 32, seed=12345)
        result2 = perlin_2d(32, 32, seed=12345)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds(self):
        """Test different seeds produce different outputs"""
        result1 = perlin_2d(32, 32, seed=1)
        result2 = perlin_2d(32, 32, seed=2)

        assert not np.array_equal(result1, result2)

    def test_different_sizes(self):
        """Test generation with different sizes"""
        result_small = perlin_2d(16, 16)
        result_large = perlin_2d(128, 128)

        assert result_small.shape == (16, 16)
        assert result_large.shape == (128, 128)

    def test_rectangular_dimensions(self):
        """Test non-square dimensions"""
        result = perlin_2d(width=100, height=50)

        assert result.shape == (50, 100)


class TestPerlinAdvanced:
    """Test advanced Perlin noise generation"""

    def test_basic_generation(self):
        """Test advanced Perlin with default parameters"""
        result = perlin_advanced(64, 64, seed=42)

        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_persistence_effect(self):
        """Test that persistence affects output"""
        result_low = perlin_advanced(32, 32, seed=42, persistence=0.3)
        result_high = perlin_advanced(32, 32, seed=42, persistence=0.7)

        # Higher persistence = more detail/contrast
        assert np.std(result_low) != np.std(result_high)

    def test_lacunarity_effect(self):
        """Test that lacunarity affects output"""
        result_low = perlin_advanced(32, 32, seed=42, lacunarity=1.5)
        result_high = perlin_advanced(32, 32, seed=42, lacunarity=3.0)

        assert not np.array_equal(result_low, result_high)

    def test_octaves_effect(self):
        """Test that octaves affect detail level"""
        result_few = perlin_advanced(32, 32, seed=42, octaves=1)
        result_many = perlin_advanced(32, 32, seed=42, octaves=6)

        # More octaves = more fine detail
        assert not np.array_equal(result_few, result_many)


class TestFourierGenerate:
    """Test Fourier series generation"""

    def test_single_frequency(self):
        """Test generation with single frequency"""
        coefficients = [(5, 1.0, 0.0)]  # freq_idx=5, mag=1.0, phase=0
        result = fourier_generate(1000, coefficients, dc_offset=0.0)

        assert result.shape == (1000,)
        assert result.dtype == np.float32

    def test_multiple_frequencies(self):
        """Test generation with multiple frequencies"""
        coefficients = [
            (5, 1.0, 0.0),
            (10, 0.5, np.pi/4),
            (20, 0.25, np.pi/2)
        ]
        result = fourier_generate(1000, coefficients, dc_offset=0.0)

        assert result.shape == (1000,)

    def test_dc_offset(self):
        """Test DC offset application"""
        coefficients = [(10, 1.0, 0.0)]
        result = fourier_generate(1000, coefficients, dc_offset=5.0)

        # Mean should be close to DC offset
        np.testing.assert_almost_equal(np.mean(result), 5.0, decimal=1)

    def test_empty_coefficients(self):
        """Test with empty coefficients (DC only)"""
        result = fourier_generate(100, [], dc_offset=3.0)

        np.testing.assert_array_almost_equal(result, np.full(100, 3.0), decimal=5)


class TestSineWave:
    """Test sine wave generation"""

    def test_basic_sine(self):
        """Test basic sine wave"""
        result = sine_wave(1000, frequency=10.0, amplitude=1.0, phase=0.0)

        assert result.shape == (1000,)
        assert result.dtype == np.float32
        np.testing.assert_almost_equal(np.max(np.abs(result)), 1.0, decimal=2)

    def test_amplitude_scaling(self):
        """Test amplitude parameter"""
        result_small = sine_wave(1000, frequency=10.0, amplitude=0.5)
        result_large = sine_wave(1000, frequency=10.0, amplitude=2.0)

        assert np.max(np.abs(result_small)) < np.max(np.abs(result_large))

    def test_dc_offset(self):
        """Test DC offset"""
        result = sine_wave(1000, frequency=10.0, amplitude=1.0, dc_offset=5.0)

        np.testing.assert_almost_equal(np.mean(result), 5.0, decimal=1)

    def test_phase_shift(self):
        """Test phase parameter"""
        result_0 = sine_wave(1000, frequency=5.0, phase=0.0)
        result_90 = sine_wave(1000, frequency=5.0, phase=np.pi/2)

        # Phase shift should produce different starting values
        assert not np.allclose(result_0[:10], result_90[:10])


class TestMultiSine:
    """Test multi-tone sine generation"""

    def test_single_component(self):
        """Test with single component (same as sine_wave)"""
        components = [(10.0, 1.0, 0.0)]
        result = multi_sine(1000, components, dc_offset=0.0)

        assert result.shape == (1000,)

    def test_multiple_components(self):
        """Test with multiple components"""
        components = [
            (10.0, 1.0, 0.0),
            (20.0, 0.5, 0.0),
            (30.0, 0.25, 0.0)
        ]
        result = multi_sine(1000, components, dc_offset=0.0)

        assert result.shape == (1000,)
        # Sum of amplitudes could be greater than individual
        assert np.max(np.abs(result)) > 0.5

    def test_dc_offset(self):
        """Test DC offset in multi-sine"""
        components = [(10.0, 1.0, 0.0)]
        result = multi_sine(1000, components, dc_offset=3.0)

        np.testing.assert_almost_equal(np.mean(result), 3.0, decimal=1)


class TestFourierAnalyze:
    """Test Fourier analysis"""

    def test_analyze_single_sine(self):
        """Test analysis of single sine wave"""
        # Create a clean sine wave
        t = np.arange(1000)
        signal = np.sin(2 * np.pi * 10 * t / 1000)

        coefficients, dc_offset = fourier_analyze(signal, max_coefficients=5)

        # Should find the dominant frequency
        assert len(coefficients) > 0
        # DC offset should be near zero for pure sine
        assert abs(dc_offset) < 0.1

    def test_analyze_with_dc(self):
        """Test analysis of signal with DC offset"""
        t = np.arange(1000)
        signal = np.sin(2 * np.pi * 5 * t / 1000) + 3.0

        coefficients, dc_offset = fourier_analyze(signal)

        # DC offset should be detected
        np.testing.assert_almost_equal(dc_offset, 3.0, decimal=1)

    def test_analyze_returns_tuples(self):
        """Test that analysis returns correct format"""
        signal = np.random.randn(500)
        coefficients, dc_offset = fourier_analyze(signal, max_coefficients=10)

        assert isinstance(coefficients, list)
        assert isinstance(dc_offset, float)
        for coef in coefficients:
            assert len(coef) == 3  # (freq_idx, magnitude, phase)


class TestPolynomialGenerate:
    """Test polynomial generation"""

    def test_linear(self):
        """Test linear polynomial"""
        # y = 2x + 1
        coefficients = [2.0, 1.0]
        result = polynomial_generate(100, coefficients)

        assert result.shape == (100,)
        # At x=0: y=1, at x=1: y=3
        np.testing.assert_almost_equal(result[0], 1.0, decimal=5)
        np.testing.assert_almost_equal(result[-1], 3.0, decimal=5)

    def test_quadratic(self):
        """Test quadratic polynomial"""
        # y = x^2
        coefficients = [1.0, 0.0, 0.0]
        result = polynomial_generate(100, coefficients)

        assert result.shape == (100,)
        assert result[0] >= -0.01  # x^2 is non-negative

    def test_constant(self):
        """Test constant polynomial"""
        coefficients = [5.0]
        result = polynomial_generate(50, coefficients)

        np.testing.assert_array_almost_equal(result, np.full(50, 5.0), decimal=5)

    def test_empty_coefficients(self):
        """Test with empty coefficients"""
        result = polynomial_generate(100, [])

        np.testing.assert_array_equal(result, np.zeros(100))


class TestPolynomialFit:
    """Test polynomial fitting"""

    def test_fit_linear(self):
        """Test fitting linear data"""
        # Generate linear data: y = 2x + 1
        x = np.linspace(0, 1, 100)
        data = 2 * x + 1

        result = polynomial_fit(data, max_degree=3)

        assert result is not None
        coeffs, degree, error = result
        assert degree >= 1
        assert error < 0.001

    def test_fit_quadratic(self):
        """Test fitting quadratic data"""
        x = np.linspace(0, 1, 100)
        data = x**2 + 0.5*x + 0.1

        result = polynomial_fit(data, max_degree=5)

        assert result is not None
        coeffs, degree, error = result
        assert degree >= 2
        assert error < 0.001

    def test_fit_constant(self):
        """Test fitting constant data"""
        data = np.full(100, 5.0)

        result = polynomial_fit(data, max_degree=5)

        assert result is not None
        coeffs, degree, error = result
        assert degree == 0
        assert error < 0.001

    def test_fit_noise_may_fail(self):
        """Test that pure noise may not fit well"""
        data = np.random.randn(100)

        result = polynomial_fit(data, max_degree=3, error_threshold=0.0001)

        # May return None or high error for random data
        if result is not None:
            coeffs, degree, error = result
            # Random data typically has high error
            assert error > 0.1 or degree == 3


class TestPythonFallbacks:
    """Test Python fallback implementations directly"""

    def test_python_perlin_2d(self):
        """Test Python Perlin 2D implementation"""
        result = _python_perlin_2d(32, 32, 42, 10.0, 4)

        assert result.shape == (32, 32)
        assert result.dtype == np.float32
        assert 0.0 <= result.min()
        assert result.max() <= 1.0

    def test_python_fourier_generate(self):
        """Test Python Fourier generate implementation"""
        coefficients = [(5, 1.0, 0.0)]
        result = _python_fourier_generate(500, coefficients, 0.0)

        assert result.shape == (500,)
        assert result.dtype == np.float32

    def test_python_sine_wave(self):
        """Test Python sine wave implementation"""
        result = _python_sine_wave(1000, 10.0, 1.0, 0.0, 0.0)

        assert result.shape == (1000,)
        assert result.dtype == np.float32

    def test_python_polynomial_generate(self):
        """Test Python polynomial generate implementation"""
        coefficients = [1.0, 2.0, 3.0]
        result = _python_polynomial_generate(100, coefficients)

        assert result.shape == (100,)
        assert result.dtype == np.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
