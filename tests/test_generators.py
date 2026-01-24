#!/usr/bin/env python3
"""
ALICE-Zip Generators Test Suite
================================

Comprehensive tests for procedural generators.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alice_zip.generators import (
    ProceduralGenerator,
    PerlinNoiseGenerator,
    FourierGenerator,
    SineWaveGenerator,
    PolynomialGenerator,
    get_generator,
    decompress_from_params,
    decompress_from_lzma,
    GeneratorType,
    GeneratorParameters,
    DataDomain,
    CompressionEngine,
    CompressionResult,
)


class TestPerlinNoiseGenerator:
    """Test PerlinNoiseGenerator"""

    def setup_method(self):
        self.generator = PerlinNoiseGenerator()

    def test_generate_2d_basic(self):
        """Test basic 2D Perlin noise generation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0, 'octaves': 4},
            output_shape=(64, 64)
        )
        result = self.generator.generate(params)

        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_generate_reproducibility(self):
        """Test that same seed produces same output"""
        params = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=12345,
            parameters={'scale': 5.0, 'octaves': 3},
            output_shape=(32, 32)
        )

        result1 = self.generator.generate(params)
        result2 = self.generator.generate(params)

        np.testing.assert_array_equal(result1, result2)

    def test_generate_different_seeds(self):
        """Test that different seeds produce different outputs"""
        params1 = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=1,
            parameters={'scale': 5.0},
            output_shape=(32, 32)
        )
        params2 = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=2,
            parameters={'scale': 5.0},
            output_shape=(32, 32)
        )

        result1 = self.generator.generate(params1)
        result2 = self.generator.generate(params2)

        assert not np.array_equal(result1, result2)

    def test_generate_different_scales(self):
        """Test effect of scale parameter"""
        params_small = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 5.0, 'octaves': 4},
            output_shape=(64, 64)
        )

        params_large = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 20.0, 'octaves': 4},
            output_shape=(64, 64)
        )

        result_small = self.generator.generate(params_small)
        result_large = self.generator.generate(params_large)

        # Different scales should produce visually different patterns
        assert not np.array_equal(result_small, result_large)

    def test_generate_with_persistence(self):
        """Test effect of persistence parameter"""
        params_low = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0, 'octaves': 4, 'persistence': 0.3},
            output_shape=(32, 32)
        )

        params_high = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0, 'octaves': 4, 'persistence': 0.7},
            output_shape=(32, 32)
        )

        result_low = self.generator.generate(params_low)
        result_high = self.generator.generate(params_high)

        # Different persistence should produce different outputs
        assert not np.array_equal(result_low, result_high)


class TestFourierGenerator:
    """Test FourierGenerator"""

    def setup_method(self):
        self.generator = FourierGenerator()

    def test_generate_single_frequency(self):
        """Test generation with single frequency"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'coefficients': [(10, 1.0, 0.0)],  # freq=10, mag=1.0, phase=0
                'dc_offset': 0.0
            },
            output_shape=(1000,)
        )

        result = self.generator.generate(params)

        assert result.shape == (1000,)
        assert result.dtype == np.float32
        # Single sine wave should be bounded
        assert -1.5 <= result.min() <= result.max() <= 1.5

    def test_generate_multiple_frequencies(self):
        """Test generation with multiple frequencies"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'coefficients': [
                    (5, 1.0, 0.0),
                    (10, 0.5, np.pi/4),
                    (15, 0.25, np.pi/2)
                ],
                'dc_offset': 0.5
            },
            output_shape=(1000,)
        )

        result = self.generator.generate(params)

        assert result.shape == (1000,)
        # Should have DC offset
        assert np.mean(result) > 0.3

    def test_generate_dc_only(self):
        """Test generation with DC offset only"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'coefficients': [],
                'dc_offset': 5.0
            },
            output_shape=(100,)
        )

        result = self.generator.generate(params)

        # All values should be close to DC offset
        np.testing.assert_array_almost_equal(result, np.full(100, 5.0), decimal=5)


class TestSineWaveGenerator:
    """Test SineWaveGenerator"""

    def setup_method(self):
        self.generator = SineWaveGenerator()

    def test_generate_basic_sine(self):
        """Test basic sine wave generation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,  # Sine uses FOURIER type internally
            seed=0,
            parameters={
                'frequency': 10.0,
                'amplitude': 1.0,
                'phase': 0.0,
                'dc_offset': 0.0,
                'is_simple_sine': True
            },
            output_shape=(1000,)
        )

        result = self.generator.generate(params)

        assert result.shape == (1000,)
        assert result.dtype == np.float32
        np.testing.assert_almost_equal(np.max(np.abs(result)), 1.0, decimal=2)

    def test_generate_with_phase(self):
        """Test sine wave with phase offset"""
        params_no_phase = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'frequency': 5.0,
                'amplitude': 1.0,
                'phase': 0.0,
                'dc_offset': 0.0
            },
            output_shape=(1000,)
        )
        params_with_phase = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'frequency': 5.0,
                'amplitude': 1.0,
                'phase': np.pi/2,  # 90 degrees
                'dc_offset': 0.0
            },
            output_shape=(1000,)
        )

        result_no_phase = self.generator.generate(params_no_phase)
        result_with_phase = self.generator.generate(params_with_phase)

        # cos(x) = sin(x + pi/2), so phase-shifted should be different
        assert not np.allclose(result_no_phase, result_with_phase)

    def test_generate_with_dc_offset(self):
        """Test sine wave with DC offset"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'frequency': 10.0,
                'amplitude': 1.0,
                'phase': 0.0,
                'dc_offset': 5.0
            },
            output_shape=(1000,)
        )

        result = self.generator.generate(params)

        # Mean should be close to DC offset
        np.testing.assert_almost_equal(np.mean(result), 5.0, decimal=1)


class TestPolynomialGenerator:
    """Test PolynomialGenerator"""

    def setup_method(self):
        self.generator = PolynomialGenerator()

    def test_generate_linear(self):
        """Test linear polynomial generation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [2.0, 1.0]},  # 2x + 1
            output_shape=(100,)
        )

        result = self.generator.generate(params)

        assert result.shape == (100,)
        # At x=0: y=1, at x=1: y=3
        assert result[0] < result[-1]

    def test_generate_quadratic(self):
        """Test quadratic polynomial generation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [1.0, 0.0, 0.0]},  # x^2
            output_shape=(100,)
        )

        result = self.generator.generate(params)

        assert result.shape == (100,)
        # x^2 is always non-negative
        assert result.min() >= -0.01

    def test_generate_constant(self):
        """Test constant polynomial"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [5.0]},  # constant 5
            output_shape=(50,)
        )

        result = self.generator.generate(params)

        np.testing.assert_array_almost_equal(result, np.full(50, 5.0), decimal=5)

    def test_generate_cubic(self):
        """Test cubic polynomial generation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [1.0, -2.0, 1.0, 0.0]},  # x^3 - 2x^2 + x
            output_shape=(100,)
        )

        result = self.generator.generate(params)

        assert result.shape == (100,)
        assert result.dtype == np.float32


class TestGetGenerator:
    """Test get_generator factory function"""

    def test_get_perlin_generator(self):
        """Test getting Perlin noise generator"""
        gen = get_generator(GeneratorType.PERLIN_NOISE)
        assert isinstance(gen, PerlinNoiseGenerator)

    def test_get_fourier_generator(self):
        """Test getting Fourier generator"""
        gen = get_generator(GeneratorType.FOURIER)
        assert isinstance(gen, FourierGenerator)

    def test_get_polynomial_generator(self):
        """Test getting polynomial generator"""
        gen = get_generator(GeneratorType.POLYNOMIAL)
        assert isinstance(gen, PolynomialGenerator)

    def test_get_unknown_generator(self):
        """Test getting unknown generator raises error"""
        with pytest.raises(ValueError, match="Unknown generator type"):
            get_generator(GeneratorType.L_SYSTEM)


class TestDecompressFromParams:
    """Test decompress_from_params function"""

    def test_decompress_simple_sine(self):
        """Test decompression of simple sine wave params"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'frequency': 20.0,
                'amplitude': 2.0,
                'phase': 0.0,
                'dc_offset': 1.0,
                'is_simple_sine': True
            },
            output_shape=(500,)
        )

        result = decompress_from_params(params)

        assert result.shape == (500,)
        assert isinstance(result, np.ndarray)

    def test_decompress_polynomial(self):
        """Test decompression of polynomial params"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [1.0, -2.0, 1.0]},  # x^2 - 2x + 1
            output_shape=(200,)
        )

        result = decompress_from_params(params)

        assert result.shape == (200,)

    def test_decompress_fourier(self):
        """Test decompression of Fourier params"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'coefficients': [(5, 1.0, 0.0), (10, 0.5, 0.0)],
                'dc_offset': 0.0
            },
            output_shape=(1000,)
        )

        result = decompress_from_params(params)

        assert result.shape == (1000,)

    def test_decompress_perlin(self):
        """Test decompression of Perlin noise params"""
        params = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0, 'octaves': 4},
            output_shape=(64, 64)
        )

        result = decompress_from_params(params)

        assert result.shape == (64, 64)


class TestDecompressFromLzma:
    """Test decompress_from_lzma function"""

    def test_decompress_lzma_float32(self):
        """Test LZMA decompression of float32 data"""
        import lzma

        # Create test data
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        compressed = lzma.compress(original.tobytes())

        # Decompress
        result = decompress_from_lzma(compressed, (5,), dtype='float32')

        np.testing.assert_array_equal(result, original)

    def test_decompress_lzma_2d(self):
        """Test LZMA decompression of 2D data"""
        import lzma

        # Create 2D test data
        original = np.arange(100, dtype=np.float32).reshape(10, 10)
        compressed = lzma.compress(original.tobytes())

        # Decompress
        result = decompress_from_lzma(compressed, (10, 10), dtype='float32')

        np.testing.assert_array_equal(result, original)


class TestGeneratorParameters:
    """Test GeneratorParameters dataclass"""

    def test_creation(self):
        """Test basic parameter creation"""
        params = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0},
            output_shape=(64, 64)
        )

        assert params.generator_type == GeneratorType.PERLIN_NOISE
        assert params.seed == 42
        assert params.parameters == {'scale': 10.0}
        assert params.output_shape == (64, 64)
        assert params.dtype == "float32"  # default

    def test_custom_dtype(self):
        """Test custom dtype"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [1.0]},
            output_shape=(100,),
            dtype="float64"
        )

        assert params.dtype == "float64"


class TestDataDomain:
    """Test DataDomain enum"""

    def test_domain_values(self):
        """Test domain enum values"""
        assert DataDomain.NUMERICAL.value == "numerical_data"
        assert DataDomain.TEXTURE_PATTERN.value == "texture_pattern"
        assert DataDomain.NATURAL_IMAGE.value == "natural_image"
        assert DataDomain.AUDIO_MUSIC.value == "audio_music"
        assert DataDomain.TEXT.value == "text"
        assert DataDomain.GENERIC.value == "generic"


class TestCompressionEngine:
    """Test CompressionEngine enum"""

    def test_engine_values(self):
        """Test compression engine enum values"""
        assert CompressionEngine.PROCEDURAL.value == "procedural_generation"
        assert CompressionEngine.NEURAL_CODEC.value == "neural_codec"
        assert CompressionEngine.FUNCTION_FITTER.value == "function_fitter"
        assert CompressionEngine.FALLBACK_LZMA.value == "fallback_lzma"


class TestCompressionResult:
    """Test CompressionResult dataclass"""

    def test_creation(self):
        """Test basic result creation"""
        result = CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0
        )

        assert result.success is True
        assert result.engine_used == CompressionEngine.PROCEDURAL
        assert result.original_size == 1000
        assert result.compressed_size == 100
        assert result.compression_ratio == 10.0

    def test_has_residual(self):
        """Test has_residual property"""
        result_no_residual = CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            residual_data=None
        )

        result_with_residual = CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            residual_data=b'some residual'
        )

        assert result_no_residual.has_residual is False
        assert result_with_residual.has_residual is True

    def test_total_compressed_size(self):
        """Test total_compressed_size property"""
        result = CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            residual_data=b'1234567890'  # 10 bytes
        )

        assert result.total_compressed_size == 110  # 100 + 10

    def test_effective_ratio(self):
        """Test effective_ratio property"""
        result = CompressionResult(
            success=True,
            engine_used=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            compression_ratio=10.0,
            residual_data=b'1234567890'  # 10 bytes
        )

        # 1000 / (100 + 10) = 9.09...
        assert abs(result.effective_ratio - (1000 / 110)) < 0.01


class TestGeneratorType:
    """Test GeneratorType enum"""

    def test_generator_types(self):
        """Test all generator type values"""
        assert GeneratorType.PERLIN_NOISE.value == "perlin_noise"
        assert GeneratorType.SIMPLEX_NOISE.value == "simplex_noise"
        assert GeneratorType.FRACTAL.value == "fractal"
        assert GeneratorType.L_SYSTEM.value == "l_system"
        assert GeneratorType.CELLULAR_AUTOMATA.value == "cellular_automata"
        assert GeneratorType.WAVE_FUNCTION.value == "wave_function"
        assert GeneratorType.FOURIER.value == "fourier"
        assert GeneratorType.POLYNOMIAL.value == "polynomial"
        assert GeneratorType.SPLINE.value == "spline"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
