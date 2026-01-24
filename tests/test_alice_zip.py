#!/usr/bin/env python3
"""
ALICE-Zip Test Suite
====================

Tests for the ALICE-Zip compression library (MIT/Core version).
Note: Compression features are Pro-only, so we test decompression and utilities.
"""

import pytest
import numpy as np
import sys
import json
import struct
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alice_zip import (
    ALICEZip,
    AliceFileHeader,
    AliceFileType,
    ALICE_MAGIC,
    ALICE_VERSION,
    ALICE_FINGERPRINT,
    CompressionEngine,
    GeneratorType,
    GeneratorParameters,
    decompress_from_params,
    decompress_from_lzma,
)


class TestALICEZip:
    """Test ALICEZip core functionality"""

    def setup_method(self):
        self.zipper = ALICEZip()

    def test_compress_numpy_array(self):
        """Test compression of numpy array (now implemented)"""
        data = np.array([x**2 for x in range(100)], dtype=np.float32)
        compressed = self.zipper.compress(data)

        assert compressed[:9] == ALICE_MAGIC
        assert len(compressed) > 0

    def test_compress_and_decompress_roundtrip(self):
        """Test compress/decompress roundtrip"""
        # Use FFT-compatible time base
        x = np.arange(100) / 100
        data = np.sin(2 * np.pi * 5 * x).astype(np.float32)

        compressed = self.zipper.compress(data)
        restored = self.zipper.decompress(compressed)

        np.testing.assert_array_almost_equal(data, restored.flatten(), decimal=5)

    def test_decompress_invalid_file_too_small(self):
        """Test decompression rejects files that are too small"""
        small_data = b'TOO_SMALL'

        with pytest.raises(ValueError, match="too small"):
            self.zipper.decompress(small_data)

    def test_decompress_invalid_magic(self):
        """Test decompression rejects files with invalid magic"""
        # Create data with wrong magic but correct size
        fake_data = b'WRONG_MAG' + b'\x00' * 100

        with pytest.raises(ValueError, match="Unknown ALICE file format"):
            self.zipper.decompress(fake_data)

    def test_decompress_texture_format_basic(self):
        """Test that ALICE_TEX solid format returns placeholder texture"""
        tex_data = json.dumps({
            "magic": "ALICE_TEX",
            "version": [1, 0],
            "params": {
                "texture_type": "solid",
                "color": [128, 128, 128],
                "width": 32,
                "height": 32
            }
        }).encode('utf-8')

        result = self.zipper.decompress(tex_data)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8


class TestAliceFileHeader:
    """Test AliceFileHeader serialization"""

    def test_header_to_bytes(self):
        """Test header serialization"""
        header = AliceFileHeader(
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
            original_hash=b'\x00' * 32,
            flags=0
        )

        header_bytes = header.to_bytes()

        assert header_bytes[:9] == ALICE_MAGIC
        assert len(header_bytes) == AliceFileHeader.size()

    def test_header_from_bytes(self):
        """Test header deserialization (v2 format)"""
        header = AliceFileHeader(
            version_minor=1,  # v1.1 for v2 format
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
        )

        header_bytes = header.to_bytes()
        restored = AliceFileHeader.from_bytes(header_bytes)

        assert restored.magic == ALICE_MAGIC
        assert restored.version_major == ALICE_VERSION[0]
        assert restored.file_type == AliceFileType.NUMPY_ARRAY
        assert restored.original_size == 1000
        assert restored.compressed_size == 100

    def test_header_size(self):
        """Test header size is correct (v2 format)"""
        assert AliceFileHeader.size() == 66  # v2 format with payload_type
        assert AliceFileHeader.size(version=1) == 65  # v1 format

    def test_header_invalid_magic_raises(self):
        """Test that invalid magic raises error"""
        invalid_data = b'NOT_ALICE' + b'\x00' * 56

        with pytest.raises(ValueError, match="Invalid ALICE_ZIP"):
            AliceFileHeader.from_bytes(invalid_data)


class TestAliceFileType:
    """Test AliceFileType enum"""

    def test_file_types(self):
        """Test file type enum values"""
        assert AliceFileType.NUMPY_ARRAY.value == 0x01
        assert AliceFileType.IMAGE.value == 0x02
        assert AliceFileType.AUDIO.value == 0x03
        assert AliceFileType.TEXT.value == 0x04
        assert AliceFileType.BINARY.value == 0x05


class TestConstants:
    """Test constants and fingerprint"""

    def test_version(self):
        """Test version format"""
        assert isinstance(ALICE_VERSION, tuple)
        assert len(ALICE_VERSION) == 2

    def test_magic(self):
        """Test magic bytes"""
        assert ALICE_MAGIC == b'ALICE_ZIP'

    def test_fingerprint(self):
        """Test fingerprint contents"""
        assert 'project' in ALICE_FINGERPRINT
        assert 'philosophy' in ALICE_FINGERPRINT
        assert 'principle' in ALICE_FINGERPRINT
        assert 'Procedural' in ALICE_FINGERPRINT['philosophy']
        assert 'Kolmogorov' in ALICE_FINGERPRINT['principle']


class TestDecompressFromParams:
    """Test direct decompression from generator parameters"""

    def test_decompress_polynomial(self):
        """Test decompression using polynomial generator"""
        params = GeneratorParameters(
            generator_type=GeneratorType.POLYNOMIAL,
            seed=0,
            parameters={'coefficients': [1.0, 2.0, 3.0]},  # 3 + 2x + x^2
            output_shape=(100,)
        )

        result = decompress_from_params(params)

        assert result.shape == (100,)
        assert result.dtype == np.float32

    def test_decompress_fourier(self):
        """Test decompression using Fourier generator"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'coefficients': [(5, 1.0, 0.0)],
                'dc_offset': 0.0
            },
            output_shape=(1000,)
        )

        result = decompress_from_params(params)

        assert result.shape == (1000,)
        assert result.dtype == np.float32

    def test_decompress_perlin(self):
        """Test decompression using Perlin noise generator"""
        params = GeneratorParameters(
            generator_type=GeneratorType.PERLIN_NOISE,
            seed=42,
            parameters={'scale': 10.0, 'octaves': 4},
            output_shape=(64, 64)
        )

        result = decompress_from_params(params)

        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_decompress_simple_sine(self):
        """Test decompression of simple sine wave"""
        params = GeneratorParameters(
            generator_type=GeneratorType.FOURIER,
            seed=0,
            parameters={
                'frequency': 10.0,
                'amplitude': 1.0,
                'phase': 0.0,
                'dc_offset': 0.0,
                'is_simple_sine': True
            },
            output_shape=(500,)
        )

        result = decompress_from_params(params)

        assert result.shape == (500,)
        np.testing.assert_almost_equal(np.max(np.abs(result)), 1.0, decimal=2)


class TestDecompressFromLzma:
    """Test LZMA decompression fallback"""

    def test_decompress_lzma_1d(self):
        """Test LZMA decompression of 1D data"""
        import lzma

        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        compressed = lzma.compress(original.tobytes())

        result = decompress_from_lzma(compressed, (5,), dtype='float32')

        np.testing.assert_array_equal(result, original)

    def test_decompress_lzma_2d(self):
        """Test LZMA decompression of 2D data"""
        import lzma

        original = np.arange(100, dtype=np.float32).reshape(10, 10)
        compressed = lzma.compress(original.tobytes())

        result = decompress_from_lzma(compressed, (10, 10), dtype='float32')

        np.testing.assert_array_equal(result, original)

    def test_decompress_lzma_different_dtype(self):
        """Test LZMA decompression with different dtype"""
        import lzma

        original = np.arange(50, dtype=np.int32)
        compressed = lzma.compress(original.tobytes())

        result = decompress_from_lzma(compressed, (50,), dtype='int32')

        np.testing.assert_array_equal(result, original)


class TestCompressionEngine:
    """Test CompressionEngine enum"""

    def test_engine_values(self):
        """Test compression engine enum values"""
        assert CompressionEngine.PROCEDURAL.value == "procedural_generation"
        assert CompressionEngine.NEURAL_CODEC.value == "neural_codec"
        assert CompressionEngine.FUNCTION_FITTER.value == "function_fitter"
        assert CompressionEngine.FALLBACK_LZMA.value == "fallback_lzma"


class TestGeneratorType:
    """Test GeneratorType enum"""

    def test_generator_types(self):
        """Test all generator type values"""
        assert GeneratorType.PERLIN_NOISE.value == "perlin_noise"
        assert GeneratorType.FOURIER.value == "fourier"
        assert GeneratorType.POLYNOMIAL.value == "polynomial"


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
