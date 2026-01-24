#!/usr/bin/env python3
"""
ALICE-Zip Media Generators Test Suite
=======================================

Comprehensive tests for media generators (Image, Audio, Video).
These tests check the MIT-licensed decompression/generation capabilities.
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if media generators are available
try:
    from alice_zip.media_generators import (
        ImagePattern,
        AudioPattern,
        VideoPattern,
        ImageParams,
        AudioParams,
        VideoParams,
        ImageGenerator,
        AudioGenerator,
        VideoGenerator,
        MediaDecompressor,
        ALICE_IMAGE_MAGIC,
        ALICE_AUDIO_MAGIC,
        ALICE_VIDEO_MAGIC,
        VERSION,
        HAS_PIL,
        HAS_SCIPY,
        HAS_CV2,
    )
    HAS_MEDIA = True
except ImportError:
    HAS_MEDIA = False


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestImagePattern:
    """Test ImagePattern enum"""

    def test_pattern_values(self):
        """Test image pattern enum values"""
        assert ImagePattern.SOLID_COLOR.value == "solid_color"
        assert ImagePattern.GRADIENT_LINEAR.value == "gradient_linear"
        assert ImagePattern.GRADIENT_RADIAL.value == "gradient_radial"
        assert ImagePattern.CHECKERBOARD.value == "checkerboard"
        assert ImagePattern.NOISE_PERLIN.value == "noise_perlin"
        assert ImagePattern.UNKNOWN.value == "unknown"


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestAudioPattern:
    """Test AudioPattern enum"""

    def test_pattern_values(self):
        """Test audio pattern enum values"""
        assert AudioPattern.SILENCE.value == "silence"
        assert AudioPattern.SINE_WAVE.value == "sine_wave"
        assert AudioPattern.MULTI_SINE.value == "multi_sine"
        assert AudioPattern.NOISE_WHITE.value == "noise_white"


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestVideoPattern:
    """Test VideoPattern enum"""

    def test_pattern_values(self):
        """Test video pattern enum values"""
        assert VideoPattern.STATIC.value == "static"
        assert VideoPattern.FADE.value == "fade"
        assert VideoPattern.SLIDE.value == "slide"


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestImageParams:
    """Test ImageParams dataclass"""

    def test_params_creation(self):
        """Test creating image parameters"""
        params = ImageParams(
            pattern=ImagePattern.SOLID_COLOR,
            width=100,
            height=100,
            channels=3,
            color=(255, 0, 0)
        )

        assert params.pattern == ImagePattern.SOLID_COLOR
        assert params.width == 100
        assert params.height == 100
        assert params.channels == 3
        assert params.color == (255, 0, 0)

    def test_params_defaults(self):
        """Test default values"""
        params = ImageParams(
            pattern=ImagePattern.GRADIENT_LINEAR,
            width=64,
            height=64
        )

        assert params.channels == 3
        assert params.gradient_start == (0, 0, 0)
        assert params.gradient_end == (255, 255, 255)


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestAudioParams:
    """Test AudioParams dataclass"""

    def test_params_creation(self):
        """Test creating audio parameters"""
        params = AudioParams(
            pattern=AudioPattern.SINE_WAVE,
            sample_rate=44100,
            duration=1.0,
            channels=1,
            frequencies=[(440.0, 1.0, 0.0)]
        )

        assert params.pattern == AudioPattern.SINE_WAVE
        assert params.sample_rate == 44100
        assert params.duration == 1.0
        assert params.channels == 1
        assert len(params.frequencies) == 1


@pytest.mark.skipif(not HAS_MEDIA or not HAS_PIL, reason="PIL not available")
class TestImageGenerator:
    """Test ImageGenerator"""

    def setup_method(self):
        self.generator = ImageGenerator()

    def test_generate_solid_color(self):
        """Test solid color image generation"""
        params = ImageParams(
            pattern=ImagePattern.SOLID_COLOR,
            width=64,
            height=64,
            channels=3,
            color=(255, 128, 0)
        )

        result = self.generator.generate(params)

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        # All pixels should be the same color
        assert np.all(result[:, :, 0] == 255)
        assert np.all(result[:, :, 1] == 128)
        assert np.all(result[:, :, 2] == 0)

    def test_generate_linear_gradient_horizontal(self):
        """Test horizontal linear gradient generation"""
        params = ImageParams(
            pattern=ImagePattern.GRADIENT_LINEAR,
            width=100,
            height=50,
            channels=3,
            gradient_start=(0, 0, 0),
            gradient_end=(255, 255, 255),
            gradient_angle=0.0
        )

        result = self.generator.generate(params)

        assert result.shape == (50, 100, 3)
        # Left edge should be dark, right edge should be light
        assert np.mean(result[:, 0, :]) < np.mean(result[:, -1, :])

    def test_generate_linear_gradient_vertical(self):
        """Test vertical linear gradient generation"""
        params = ImageParams(
            pattern=ImagePattern.GRADIENT_LINEAR,
            width=50,
            height=100,
            channels=3,
            gradient_start=(0, 0, 0),
            gradient_end=(255, 255, 255),
            gradient_angle=90.0
        )

        result = self.generator.generate(params)

        assert result.shape == (100, 50, 3)
        # Top edge should be dark, bottom edge should be light
        assert np.mean(result[0, :, :]) < np.mean(result[-1, :, :])

    def test_generate_radial_gradient(self):
        """Test radial gradient generation"""
        params = ImageParams(
            pattern=ImagePattern.GRADIENT_RADIAL,
            width=100,
            height=100,
            channels=3,
            gradient_start=(255, 255, 255),
            gradient_end=(0, 0, 0),
            gradient_center=(0.5, 0.5)
        )

        result = self.generator.generate(params)

        assert result.shape == (100, 100, 3)
        # Center should be lighter than corners
        center_val = np.mean(result[50, 50, :])
        corner_val = np.mean(result[0, 0, :])
        assert center_val > corner_val

    def test_generate_checkerboard(self):
        """Test checkerboard pattern generation"""
        params = ImageParams(
            pattern=ImagePattern.CHECKERBOARD,
            width=64,
            height=64,
            channels=3,
            pattern_scale=8,
            color=(0, 0, 0),
            gradient_end=(255, 255, 255)
        )

        result = self.generator.generate(params)

        assert result.shape == (64, 64, 3)
        # Should have alternating pattern
        # Check two adjacent 8x8 cells are different
        cell1 = result[0:8, 0:8, 0].mean()
        cell2 = result[0:8, 8:16, 0].mean()
        assert abs(cell1 - cell2) > 100

    def test_decompress_valid_data(self):
        """Test decompression from valid ALICE format"""
        compressed = {
            'magic': ALICE_IMAGE_MAGIC.decode(),
            'version': list(VERSION),
            'params': {
                'pattern': ImagePattern.SOLID_COLOR.value,
                'width': 32,
                'height': 32,
                'channels': 3,
                'color': [100, 150, 200]
            }
        }

        result = self.generator.decompress(json.dumps(compressed).encode())

        assert result.shape == (32, 32, 3)
        np.testing.assert_array_equal(result[0, 0], [100, 150, 200])


@pytest.mark.skipif(not HAS_MEDIA or not HAS_SCIPY, reason="SciPy not available")
class TestAudioGenerator:
    """Test AudioGenerator"""

    def setup_method(self):
        self.generator = AudioGenerator()

    def test_generate_silence(self):
        """Test silence generation"""
        params = AudioParams(
            pattern=AudioPattern.SILENCE,
            sample_rate=44100,
            duration=0.1,
            channels=1
        )

        result = self.generator.generate(params)

        assert result.shape[0] == int(44100 * 0.1)
        assert result.dtype == np.int16
        # Silence should be all zeros
        assert np.all(result == 0)

    def test_generate_sine_wave(self):
        """Test sine wave generation"""
        params = AudioParams(
            pattern=AudioPattern.SINE_WAVE,
            sample_rate=44100,
            duration=0.1,
            channels=1,
            frequencies=[(440.0, 1.0, 0.0)]
        )

        result = self.generator.generate(params)

        n_samples = int(44100 * 0.1)
        assert result.shape[0] == n_samples
        assert result.dtype == np.int16
        # Should have some amplitude
        assert np.max(np.abs(result)) > 1000

    def test_generate_multi_sine(self):
        """Test multi-sine (chord) generation"""
        params = AudioParams(
            pattern=AudioPattern.MULTI_SINE,
            sample_rate=44100,
            duration=0.1,
            channels=1,
            frequencies=[
                (261.63, 1.0, 0.0),  # C4
                (329.63, 1.0, 0.0),  # E4
                (392.00, 1.0, 0.0),  # G4
            ]
        )

        result = self.generator.generate(params)

        assert result.dtype == np.int16
        assert np.max(np.abs(result)) > 1000

    def test_generate_white_noise(self):
        """Test white noise generation"""
        params = AudioParams(
            pattern=AudioPattern.NOISE_WHITE,
            sample_rate=44100,
            duration=0.1,
            channels=1,
            noise_amplitude=0.5
        )

        result = self.generator.generate(params)

        assert result.dtype == np.int16
        # White noise should have variation
        assert np.std(result) > 100

    def test_generate_stereo(self):
        """Test stereo audio generation"""
        params = AudioParams(
            pattern=AudioPattern.SINE_WAVE,
            sample_rate=44100,
            duration=0.05,
            channels=2,
            frequencies=[(440.0, 1.0, 0.0)]
        )

        result = self.generator.generate(params)

        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_decompress_valid_data(self):
        """Test decompression from valid ALICE format"""
        compressed = {
            'magic': ALICE_AUDIO_MAGIC.decode(),
            'version': list(VERSION),
            'params': {
                'pattern': AudioPattern.SINE_WAVE.value,
                'sample_rate': 22050,
                'duration': 0.05,
                'channels': 1,
                'frequencies': [[440.0, 1.0, 0.0]]
            }
        }

        audio, sr = self.generator.decompress(json.dumps(compressed).encode())

        assert sr == 22050
        assert audio.dtype == np.int16


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestMediaDecompressor:
    """Test MediaDecompressor unified interface"""

    def setup_method(self):
        self.decompressor = MediaDecompressor()

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
    def test_decompress_image(self):
        """Test image decompression via unified interface"""
        compressed = {
            'magic': ALICE_IMAGE_MAGIC.decode(),
            'version': list(VERSION),
            'params': {
                'pattern': ImagePattern.SOLID_COLOR.value,
                'width': 16,
                'height': 16,
                'channels': 3,
                'color': [255, 255, 255]
            }
        }

        result = self.decompressor.decompress(json.dumps(compressed).encode())

        assert result.shape == (16, 16, 3)

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_decompress_audio(self):
        """Test audio decompression via unified interface"""
        compressed = {
            'magic': ALICE_AUDIO_MAGIC.decode(),
            'version': list(VERSION),
            'params': {
                'pattern': AudioPattern.SILENCE.value,
                'sample_rate': 44100,
                'duration': 0.01,
                'channels': 1,
                'frequencies': []
            }
        }

        audio, sr = self.decompressor.decompress(json.dumps(compressed).encode())

        assert sr == 44100

    def test_decompress_unknown_format(self):
        """Test decompression of unknown format raises error"""
        compressed = {
            'magic': 'UNKNOWN',
            'version': list(VERSION),
        }

        with pytest.raises(ValueError, match="Unknown ALICE media format"):
            self.decompressor.decompress(json.dumps(compressed).encode())


@pytest.mark.skipif(not HAS_MEDIA, reason="Media generators not available")
class TestConstants:
    """Test media generator constants"""

    def test_magic_values(self):
        """Test magic byte values"""
        assert ALICE_IMAGE_MAGIC == b'ALICE_IMG'
        assert ALICE_AUDIO_MAGIC == b'ALICE_AUD'
        assert ALICE_VIDEO_MAGIC == b'ALICE_VID'

    def test_version(self):
        """Test version tuple"""
        assert isinstance(VERSION, tuple)
        assert len(VERSION) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
