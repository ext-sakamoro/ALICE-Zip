#!/usr/bin/env python3
"""
ALICE-Zip Core Test Suite
==========================

Comprehensive tests for core ALICEZip functionality.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from alice_zip.core import (
    ALICEZip,
    AliceFileHeader,
    AliceFileType,
    ALICE_MAGIC,
    ALICE_VERSION,
    ALICE_FINGERPRINT,
)
from alice_zip.generators import (
    CompressionEngine,
    GeneratorType,
)


class TestAliceFileHeader:
    """Test AliceFileHeader dataclass"""

    def test_header_to_bytes(self):
        """Test header serialization to bytes (v2 format with payload_type)"""
        header = AliceFileHeader(
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=1000,
            compressed_size=100,
        )

        header_bytes = header.to_bytes()

        assert header_bytes[:9] == ALICE_MAGIC
        assert len(header_bytes) == AliceFileHeader.size()
        assert len(header_bytes) == 66  # v2 format includes payload_type

    def test_header_from_bytes(self):
        """Test header deserialization from bytes (v2 format)"""
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

    def test_header_roundtrip_all_types(self):
        """Test header serialization/deserialization for all file types"""
        for file_type in AliceFileType:
            header = AliceFileHeader(
                file_type=file_type,
                engine=CompressionEngine.PROCEDURAL,
                original_size=500,
                compressed_size=50
            )
            header_bytes = header.to_bytes()
            restored = AliceFileHeader.from_bytes(header_bytes)
            assert restored.file_type == file_type

    def test_header_invalid_magic_raises(self):
        """Test that invalid magic raises error"""
        invalid_data = b'NOT_ALICE' + b'\x00' * 56

        with pytest.raises(ValueError, match="Invalid ALICE_ZIP"):
            AliceFileHeader.from_bytes(invalid_data)

    def test_header_size(self):
        """Test header size constant (v2 format)"""
        assert AliceFileHeader.size() == 66  # v2 format with payload_type
        assert AliceFileHeader.size(version=1) == 65  # v1 format without payload_type


class TestAliceFileType:
    """Test AliceFileType enum"""

    def test_file_type_values(self):
        """Test file type enum values"""
        assert AliceFileType.NUMPY_ARRAY.value == 0x01
        assert AliceFileType.IMAGE.value == 0x02
        assert AliceFileType.AUDIO.value == 0x03
        assert AliceFileType.TEXT.value == 0x04
        assert AliceFileType.BINARY.value == 0x05


class TestALICEZipCompress:
    """Test ALICEZip compress functionality (now implemented in Core)"""

    def setup_method(self):
        self.zipper = ALICEZip()

    def test_compress_numpy_array(self):
        """Test compression of numpy array"""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        compressed = self.zipper.compress(data)

        assert compressed[:9] == ALICE_MAGIC
        assert len(compressed) > 0

    def test_compress_and_decompress_roundtrip(self):
        """Test compress/decompress roundtrip"""
        # Use FFT-compatible time base
        x = np.arange(100) / 100
        data = np.sin(2 * np.pi * 5 * x)

        compressed = self.zipper.compress(data)
        restored = self.zipper.decompress(compressed)

        np.testing.assert_array_almost_equal(data, restored.flatten(), decimal=5)


class TestALICEZipDecompress:
    """Test ALICEZip decompression (Core/MIT functionality)"""

    def setup_method(self):
        self.zipper = ALICEZip()

    def test_decompress_rejects_small_file(self):
        """Test decompression rejects files that are too small"""
        small_data = b'TOO_SMALL'

        with pytest.raises(ValueError, match="too small"):
            self.zipper.decompress(small_data)

    def test_decompress_rejects_invalid_magic(self):
        """Test decompression rejects invalid magic"""
        fake_data = b'WRONG_MAG' + b'\x00' * 100

        with pytest.raises(ValueError, match="Unknown ALICE file format"):
            self.zipper.decompress(fake_data)

    def test_decompress_texture_format_basic(self):
        """Test that ALICE_TEX solid format returns placeholder texture"""
        import json
        tex_data = json.dumps({
            "magic": "ALICE_TEX",
            "version": [1, 0],
            "params": {
                "texture_type": "solid",
                "color": [128, 128, 128],
                "width": 64,
                "height": 64
            }
        }).encode('utf-8')

        result = self.zipper.decompress(tex_data)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_decompress_valid_generator_payload(self):
        """Test decompression with valid generator payload"""
        # Create payload
        payload = {
            'type': 'generator',
            'params': {
                'generator_type': 'polynomial',
                'seed': 0,
                'parameters': {'coefficients': [1.0, 2.0]},  # 2x + 1
                'output_shape': [100],
                'dtype': 'float32'
            }
        }
        payload_bytes = json.dumps(payload).encode('utf-8')

        # Create header with correct compressed_size
        header = AliceFileHeader(
            version_minor=1,  # v1.1 for v2 format
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=400,  # 100 float32 = 400 bytes
            compressed_size=len(payload_bytes),
        )

        data = header.to_bytes() + payload_bytes
        result = self.zipper.decompress(data)

        assert result.shape == (100,)
        assert result.dtype == np.float32

    def test_decompress_from_file_path(self):
        """Test decompression from file path"""
        # Create payload
        payload = {
            'type': 'generator',
            'params': {
                'generator_type': 'polynomial',
                'seed': 0,
                'parameters': {'coefficients': [1.0]},  # constant
                'output_shape': [50],
                'dtype': 'float32'
            }
        }
        payload_bytes = json.dumps(payload).encode('utf-8')

        # Create header with correct compressed_size
        header = AliceFileHeader(
            version_minor=1,  # v1.1 for v2 format
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=200,
            compressed_size=len(payload_bytes),
        )

        data = header.to_bytes() + payload_bytes

        with tempfile.NamedTemporaryFile(suffix='.alice', delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            result = self.zipper.decompress(temp_path)
            assert result.shape == (50,)
        finally:
            temp_path.unlink()


class TestALICEZipInfo:
    """Test ALICEZip info functionality"""

    def setup_method(self):
        self.zipper = ALICEZip()

    def test_info_from_file(self):
        """Test getting info from .alice file"""
        # Create payload
        payload = {
            'type': 'generator',
            'params': {
                'generator_type': 'polynomial',
                'seed': 42,
                'parameters': {'coefficients': [1.0, 2.0]},
                'output_shape': [100],
                'dtype': 'float32'
            },
            'metadata': {'test_key': 'test_value'}
        }
        payload_bytes = json.dumps(payload).encode('utf-8')

        # Create header with correct compressed_size
        header = AliceFileHeader(
            version_minor=1,  # v1.1 for v2 format
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=CompressionEngine.PROCEDURAL,
            original_size=400,
            compressed_size=len(payload_bytes),
        )

        data = header.to_bytes() + payload_bytes

        with tempfile.NamedTemporaryFile(suffix='.alice', delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            info = self.zipper.info(temp_path)

            assert info['magic'] == 'ALICE_ZIP'
            assert info['file_type'] == 'NUMPY_ARRAY'
            assert info['engine'] == 'procedural_generation'
            assert info['original_size'] == 400
            assert info['payload_content_type'] == 'generator'
            assert 'alice_fingerprint' in info
        finally:
            temp_path.unlink()

    def test_info_texture_format(self):
        """Test getting info from texture format file"""
        tex_data = {
            'magic': 'ALICE_TEX',
            'version': [1, 0],
            'params': {
                'texture_type': 'perlin_noise',
                'width': 256,
                'height': 256,
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.alice', delete=False) as f:
            f.write(json.dumps(tex_data).encode('utf-8'))
            temp_path = Path(f.name)

        try:
            info = self.zipper.info(temp_path)

            assert info['magic'] == 'ALICE_TEX'
            assert info['file_type'] == 'TEXTURE'
            assert info['texture_type'] == 'perlin_noise'
            assert info['dimensions'] == '256x256'
        finally:
            temp_path.unlink()


class TestConstants:
    """Test module constants"""

    def test_alice_magic(self):
        """Test ALICE magic bytes"""
        assert ALICE_MAGIC == b'ALICE_ZIP'
        assert len(ALICE_MAGIC) == 9

    def test_alice_version(self):
        """Test ALICE version tuple"""
        assert isinstance(ALICE_VERSION, tuple)
        assert len(ALICE_VERSION) == 2
        assert all(isinstance(v, int) for v in ALICE_VERSION)

    def test_alice_fingerprint(self):
        """Test ALICE fingerprint dictionary"""
        assert isinstance(ALICE_FINGERPRINT, dict)
        assert 'project' in ALICE_FINGERPRINT
        assert 'philosophy' in ALICE_FINGERPRINT
        assert 'principle' in ALICE_FINGERPRINT
        assert ALICE_FINGERPRINT['project'] == 'ALICE-Zip'
        assert 'Procedural' in ALICE_FINGERPRINT['philosophy']
        assert 'Kolmogorov' in ALICE_FINGERPRINT['principle']


class TestCompressionEngine:
    """Test CompressionEngine enum"""

    def test_engine_values(self):
        """Test compression engine enum values"""
        assert CompressionEngine.PROCEDURAL.value == "procedural_generation"
        assert CompressionEngine.NEURAL_CODEC.value == "neural_codec"
        assert CompressionEngine.FUNCTION_FITTER.value == "function_fitter"
        assert CompressionEngine.FALLBACK_LZMA.value == "fallback_lzma"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
