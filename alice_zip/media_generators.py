#!/usr/bin/env python3
"""
ALICE-Zip Media Generators (MIT License)
=========================================

Generation logic for media decompression.
This module contains the code that regenerates media from parameters.
Required for decompression.

Author: Moroya Sakamoto
License: MIT

Copyright (c) 2026 Moroya Sakamoto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import numpy as np

# Optional imports with graceful fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import scipy.fft as fft
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# Constants
# ============================================================================

ALICE_IMAGE_MAGIC = b'ALICE_IMG'
ALICE_AUDIO_MAGIC = b'ALICE_AUD'
ALICE_VIDEO_MAGIC = b'ALICE_VID'
VERSION = (1, 0)


# ============================================================================
# Binary Encoding Helpers (for DCT coefficients, motion vectors, etc.)
# ============================================================================

def encode_coefficients_b64(coefficients: np.ndarray) -> str:
    """
    Encode numerical coefficients as Base64 string.

    Using binary format reduces:
    - File size: 3-4x smaller than JSON text
    - Parse time: 10-100x faster than json.loads

    Args:
        coefficients: numpy array of shape (N, 3) with (y, x, value) rows

    Returns:
        Base64-encoded string
    """
    import base64
    import logging
    logger = logging.getLogger(__name__)

    # Use explicit little-endian '<f8' for cross-platform binary compatibility
    # (x86/ARM little-endian ↔ big-endian systems like some embedded/network devices)
    arr = np.asarray(coefficients, dtype='<f8')
    result = base64.b64encode(arr.tobytes()).decode('ascii')

    # Warn about large Base64 data (will cause slow JSON parsing on load)
    LARGE_B64_THRESHOLD = 10 * 1024 * 1024  # 10MB
    if len(result) > LARGE_B64_THRESHOLD:
        logger.warning(
            f"Encoding large data as Base64 ({len(result) / (1024**2):.1f} MB). "
            f"This will cause slow JSON parsing when loading. "
            f"Consider using separate binary payload format for data >10MB."
        )

    return result


def decode_coefficients_b64(b64_str: str, shape: Tuple[int, int] = (-1, 3)) -> np.ndarray:
    """
    Decode Base64 string to numpy array.

    Args:
        b64_str: Base64-encoded string
        shape: Expected array shape (default: Nx3 for DCT coefficients)

    Returns:
        numpy array
    """
    import base64
    import logging
    logger = logging.getLogger(__name__)

    # Warn about large Base64 data embedded in JSON (performance concern)
    LARGE_B64_THRESHOLD = 10 * 1024 * 1024  # 10MB
    b64_size = len(b64_str)
    if b64_size > LARGE_B64_THRESHOLD:
        logger.warning(
            f"Large Base64 data in JSON ({b64_size / (1024**2):.1f} MB). "
            f"This may cause slow parsing and high memory usage. "
            f"Consider using separate binary payload format for better performance."
        )

    raw = base64.b64decode(b64_str)
    # Use explicit little-endian '<f8' for cross-platform binary compatibility
    return np.frombuffer(raw, dtype='<f8').reshape(shape)


# ============================================================================
# Enums
# ============================================================================

class ImagePattern(Enum):
    """Detected image patterns"""
    SOLID_COLOR = "solid_color"
    GRADIENT_LINEAR = "gradient_linear"
    GRADIENT_RADIAL = "gradient_radial"
    CHECKERBOARD = "checkerboard"
    STRIPES = "stripes"
    NOISE_PERLIN = "noise_perlin"
    NOISE_GAUSSIAN = "noise_gaussian"
    PERIODIC = "periodic"
    FRACTAL = "fractal"
    UNKNOWN = "unknown"


class AudioPattern(Enum):
    """Detected audio patterns"""
    SILENCE = "silence"
    SINE_WAVE = "sine_wave"
    MULTI_SINE = "multi_sine"
    NOISE_WHITE = "noise_white"
    NOISE_PINK = "noise_pink"
    SPEECH = "speech"
    MUSIC = "music"
    UNKNOWN = "unknown"


class VideoPattern(Enum):
    """Detected video patterns"""
    STATIC = "static"
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"
    ROTATE = "rotate"
    PARTICLE = "particle"
    PROCEDURAL_ANIMATION = "procedural_animation"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ImageParams:
    """Parameters for procedural image generation"""
    pattern: ImagePattern
    width: int
    height: int
    channels: int = 3

    color: Tuple[int, ...] = (0, 0, 0)
    gradient_start: Tuple[int, ...] = (0, 0, 0)
    gradient_end: Tuple[int, ...] = (255, 255, 255)
    gradient_angle: float = 0.0
    gradient_center: Tuple[float, float] = (0.5, 0.5)

    pattern_scale: float = 1.0
    pattern_seed: int = 42

    # DCT coefficients: JSON list (legacy) or Base64-encoded binary (preferred)
    # Binary format: 3 columns (y, x, value) as float64, stored as Base64 string
    # Using binary reduces file size by 3-4x and parsing time by 10-100x
    dct_coefficients: List[Tuple[int, int, float]] = field(default_factory=list)
    dct_coefficients_b64: Optional[str] = None  # Base64-encoded numpy array

    regions: List[Dict[str, Any]] = field(default_factory=list)
    fallback_data: Optional[bytes] = None


@dataclass
class AudioParams:
    """Parameters for procedural audio generation"""
    pattern: AudioPattern
    sample_rate: int
    duration: float
    channels: int = 1

    frequencies: List[Tuple[float, float, float]] = field(default_factory=list)
    envelope: Tuple[float, float, float, float] = (0.01, 0.1, 0.7, 0.2)

    noise_type: str = "white"
    noise_amplitude: float = 0.0

    # Global max amplitude from compression analysis (for normalization)
    # If set, used directly instead of per-chunk normalization
    global_max_amplitude: Optional[float] = None

    fallback_data: Optional[bytes] = None


@dataclass
class VideoParams:
    """Parameters for procedural video generation"""
    pattern: VideoPattern
    width: int
    height: int
    fps: float
    duration: float

    keyframes: List[Tuple[float, ImageParams]] = field(default_factory=list)
    motion_vectors: List[Dict[str, Any]] = field(default_factory=list)
    animation_params: Dict[str, Any] = field(default_factory=dict)

    fallback_data: Optional[bytes] = None


# ============================================================================
# Image Generator (MIT)
# ============================================================================

class ImageGenerator:
    """Generate images from procedural parameters (MIT License)"""

    def __init__(self):
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required: pip install Pillow")

    def generate(self, params: ImageParams) -> np.ndarray:
        """Generate image from procedural parameters"""
        h, w, c = params.height, params.width, params.channels

        if params.pattern == ImagePattern.SOLID_COLOR:
            return np.full((h, w, c), params.color[:c], dtype=np.uint8)

        elif params.pattern == ImagePattern.GRADIENT_LINEAR:
            return self._generate_linear_gradient(params)

        elif params.pattern == ImagePattern.GRADIENT_RADIAL:
            return self._generate_radial_gradient(params)

        elif params.pattern == ImagePattern.CHECKERBOARD:
            return self._generate_checkerboard(params)

        elif params.pattern in [ImagePattern.PERIODIC, ImagePattern.UNKNOWN]:
            return self._generate_from_dct(params)

        else:
            return np.full((h, w, c), 128, dtype=np.uint8)

    def _generate_linear_gradient(self, params: ImageParams) -> np.ndarray:
        """Generate linear gradient (vectorized with NumPy)"""
        h, w, c = params.height, params.width, params.channels

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Calculate interpolation factor t based on gradient angle
        if params.gradient_angle == 0:
            t = x_coords / max(w - 1, 1)
        elif params.gradient_angle == 90:
            t = y_coords / max(h - 1, 1)
        else:
            angle_rad = np.radians(params.gradient_angle)
            nx = np.cos(angle_rad)
            ny = np.sin(angle_rad)
            divisor = w * abs(nx) + h * abs(ny)
            t = (x_coords * nx + y_coords * ny) / max(divisor, 1e-10)

        t = np.clip(t, 0, 1)

        # Build start and end color arrays with proper padding
        start = np.array(params.gradient_start[:c] + (0,) * (c - len(params.gradient_start)), dtype=np.float32)
        end = np.array(params.gradient_end[:c] + (255,) * (c - len(params.gradient_end)), dtype=np.float32)

        # Vectorized interpolation: image[y, x, ch] = start[ch] * (1-t) + end[ch] * t
        # Broadcast t (h, w) with colors (c,) -> (h, w, c)
        t_expanded = t[:, :, np.newaxis]
        image = start * (1 - t_expanded) + end * t_expanded

        return np.clip(image, 0, 255).astype(np.uint8)

    def _generate_radial_gradient(self, params: ImageParams) -> np.ndarray:
        """Generate radial gradient (vectorized with NumPy)"""
        h, w, c = params.height, params.width, params.channels

        # Calculate center coordinates
        cy = params.gradient_center[1] * h
        cx = params.gradient_center[0] * w
        max_dist = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)
        max_dist = max(max_dist, 1e-10)  # Avoid division by zero

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Vectorized distance calculation
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / max_dist
        dist = np.clip(dist, 0, 1)

        # Build start and end color arrays with proper padding
        start = np.array(params.gradient_start[:c] + (0,) * (c - len(params.gradient_start)), dtype=np.float32)
        end = np.array(params.gradient_end[:c] + (255,) * (c - len(params.gradient_end)), dtype=np.float32)

        # Vectorized interpolation
        dist_expanded = dist[:, :, np.newaxis]
        image = start * (1 - dist_expanded) + end * dist_expanded

        return np.clip(image, 0, 255).astype(np.uint8)

    def _generate_checkerboard(self, params: ImageParams) -> np.ndarray:
        """Generate checkerboard (vectorized with NumPy)"""
        h, w, c = params.height, params.width, params.channels
        cell = max(int(params.pattern_scale), 1)  # Ensure positive cell size

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Vectorized checkerboard pattern: boolean mask
        checker_mask = ((x_coords // cell) + (y_coords // cell)) % 2 == 0

        # Prepare colors
        color1 = np.array(params.color[:c] + (0,) * (c - len(params.color)), dtype=np.uint8)
        color2 = np.array(params.gradient_end[:c] + (255,) * (c - len(params.gradient_end)), dtype=np.uint8)

        # Use np.where with broadcasting
        image = np.where(checker_mask[:, :, np.newaxis], color1, color2)

        return image.astype(np.uint8)

    def _generate_from_dct(self, params: ImageParams) -> np.ndarray:
        """
        Generate from DCT coefficients (vectorized with NumPy advanced indexing).

        Supports both:
        - Binary format (dct_coefficients_b64): 10-100x faster parsing
        - JSON list format (dct_coefficients): legacy, slower but human-readable
        """
        # Check for any coefficients
        has_b64 = params.dct_coefficients_b64 is not None
        has_json = params.dct_coefficients and len(params.dct_coefficients) > 0

        if not HAS_SCIPY or (not has_b64 and not has_json):
            return np.full((params.height, params.width, params.channels), 128, dtype=np.uint8)

        h, w = params.height, params.width
        dct = np.zeros((h, w), dtype=np.float64)

        # Prefer binary format (faster)
        if has_b64:
            coeffs = decode_coefficients_b64(params.dct_coefficients_b64)
        else:
            # Legacy JSON list format
            coeffs = np.array(params.dct_coefficients, dtype=np.float64)

        if len(coeffs) > 0:
            y_indices = coeffs[:, 0].astype(int)
            x_indices = coeffs[:, 1].astype(int)
            values = coeffs[:, 2]

            # Filter valid indices (within bounds)
            valid_mask = (y_indices >= 0) & (y_indices < h) & (x_indices >= 0) & (x_indices < w)
            y_valid = y_indices[valid_mask]
            x_valid = x_indices[valid_mask]
            val_valid = values[valid_mask]

            # Vectorized assignment using advanced indexing
            dct[y_valid, x_valid] = val_valid

        gray = fft.idctn(dct, norm='ortho')
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        if params.channels == 1:
            return gray[:, :, np.newaxis]
        else:
            return np.stack([gray] * params.channels, axis=-1)

    def decompress(self, data: bytes) -> np.ndarray:
        """Decompress ALICE format to image"""
        parsed = json.loads(data.decode('utf-8'))

        if parsed.get('magic') != ALICE_IMAGE_MAGIC.decode():
            raise ValueError("Invalid ALICE_IMG file")

        p = parsed['params']
        params = ImageParams(
            pattern=ImagePattern(p['pattern']),
            width=p['width'],
            height=p['height'],
            channels=p['channels'],
            color=tuple(p.get('color', (0, 0, 0))),
            gradient_start=tuple(p.get('gradient_start', (0, 0, 0))),
            gradient_end=tuple(p.get('gradient_end', (255, 255, 255))),
            gradient_angle=p.get('gradient_angle', 0.0),
            gradient_center=tuple(p.get('gradient_center', (0.5, 0.5))),
            pattern_scale=p.get('pattern_scale', 1.0),
            # Prefer binary format (faster, smaller)
            dct_coefficients=p.get('dct_coefficients', []),
            dct_coefficients_b64=p.get('dct_coefficients_b64')
        )

        return self.generate(params)


# ============================================================================
# Audio Generator (MIT)
# ============================================================================

class AudioGenerator:
    """Generate audio from procedural parameters (MIT License)"""

    # Default chunk size: 1 second of audio at 44.1kHz = ~88KB per channel
    DEFAULT_CHUNK_SAMPLES = 44100

    def __init__(self):
        if not HAS_SCIPY:
            raise ImportError("SciPy required: pip install scipy")

    def generate_chunks(self, params: AudioParams, chunk_samples: Optional[int] = None):
        """
        Generate audio in chunks (generator pattern).

        Yields chunks of audio data to avoid allocating entire audio buffer.
        Memory usage is O(chunk_size) instead of O(total_duration).

        Args:
            params: Audio generation parameters
            chunk_samples: Samples per chunk (default: 44100 = 1 second at 44.1kHz)

        Yields:
            np.ndarray: Audio chunk as int16 array
        """
        if chunk_samples is None:
            chunk_samples = self.DEFAULT_CHUNK_SAMPLES

        n_samples = int(params.duration * params.sample_rate)
        sr = params.sample_rate

        # For envelope calculation, we need to track global position
        total_samples = n_samples

        # Process in chunks
        for chunk_start in range(0, n_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, n_samples)
            chunk_len = chunk_end - chunk_start

            # Create time array for this chunk
            t = np.arange(chunk_start, chunk_end) / sr

            # Generate chunk based on pattern
            if params.pattern == AudioPattern.SILENCE:
                chunk = np.zeros(chunk_len, dtype=np.float64)

            elif params.pattern in [AudioPattern.SINE_WAVE, AudioPattern.MULTI_SINE]:
                chunk = np.zeros(chunk_len, dtype=np.float64)
                for freq, amp, phase in params.frequencies:
                    chunk += amp * np.sin(2 * np.pi * freq * t + phase)

            elif params.pattern == AudioPattern.NOISE_WHITE:
                chunk = np.random.randn(chunk_len) * params.noise_amplitude

            else:
                chunk = np.zeros(chunk_len, dtype=np.float64)

            # Apply envelope for this chunk's position
            if params.pattern != AudioPattern.SILENCE:
                chunk = self._apply_envelope_chunk(
                    chunk, params, chunk_start, total_samples
                )

            # Apply normalization using global max (if provided from compression analysis)
            # DO NOT normalize per-chunk - this causes severe audio artifacts!
            if params.global_max_amplitude is not None and params.global_max_amplitude > 0:
                # Use the global max from compression analysis
                chunk = chunk / params.global_max_amplitude * 0.9

            # Convert to int16 (clip to prevent overflow)
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk_int16 = (chunk * 32767).astype(np.int16)

            # Handle multi-channel
            if params.channels > 1:
                chunk_int16 = np.stack([chunk_int16] * params.channels, axis=-1)

            yield chunk_int16

    def generate(self, params: AudioParams) -> np.ndarray:
        """
        Generate complete audio array from procedural parameters.

        For large audio files (>10 seconds), consider using generate_chunks()
        or generate_to_file() to reduce memory usage.

        Args:
            params: Audio generation parameters

        Returns:
            np.ndarray: Complete audio as int16 array
        """
        # For short audio, use direct generation for simplicity
        n_samples = int(params.duration * params.sample_rate)

        # Warn for large audio that would use significant memory
        estimated_bytes = n_samples * 8 * params.channels  # float64 during computation
        if estimated_bytes > 100 * 1024 * 1024:  # >100MB
            import logging
            logging.getLogger(__name__).warning(
                f"Large audio generation ({estimated_bytes / (1024**2):.1f} MB). "
                f"Consider using generate_chunks() or generate_to_file() for streaming."
            )

        # Use chunked generation and concatenate
        chunks = list(self.generate_chunks(params))
        return np.concatenate(chunks, axis=0)

    def generate_to_file(self, params: AudioParams, output_path: Union[str, Path],
                         chunk_samples: Optional[int] = None):
        """
        Generate audio directly to WAV file without holding entire buffer in memory.

        This is the recommended method for long audio files (>10 seconds).
        Memory usage is O(chunk_size) regardless of total duration.

        Args:
            params: Audio generation parameters
            output_path: Output WAV file path
            chunk_samples: Samples per chunk (default: 44100)
        """
        import wave

        output_path = Path(output_path)

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(params.channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(params.sample_rate)

            for chunk in self.generate_chunks(params, chunk_samples):
                wav_file.writeframes(chunk.tobytes())

    def _apply_envelope_chunk(self, chunk: np.ndarray, params: AudioParams,
                              chunk_start: int, total_samples: int) -> np.ndarray:
        """Apply ADSR envelope to a chunk based on its global position"""
        a, d, s, r = params.envelope
        sr = params.sample_rate

        attack_samples = int(a * sr)
        decay_samples = int(d * sr)
        release_samples = int(r * sr)
        sustain_start = attack_samples + decay_samples
        release_start = total_samples - release_samples

        chunk_len = len(chunk)
        envelope = np.ones(chunk_len, dtype=np.float64)

        for i in range(chunk_len):
            global_pos = chunk_start + i

            if global_pos < attack_samples:
                # Attack phase
                envelope[i] = global_pos / max(attack_samples, 1)
            elif global_pos < sustain_start:
                # Decay phase
                decay_pos = global_pos - attack_samples
                envelope[i] = 1.0 - (1.0 - s) * decay_pos / max(decay_samples, 1)
            elif global_pos < release_start:
                # Sustain phase
                envelope[i] = s
            else:
                # Release phase
                release_pos = global_pos - release_start
                envelope[i] = s * (1.0 - release_pos / max(release_samples, 1))

        return chunk * envelope

    def decompress(self, data: bytes) -> Tuple[np.ndarray, int]:
        """Decompress ALICE format to audio"""
        parsed = json.loads(data.decode('utf-8'))

        if parsed.get('magic') != ALICE_AUDIO_MAGIC.decode():
            raise ValueError("Invalid ALICE_AUD file")

        p = parsed['params']
        params = AudioParams(
            pattern=AudioPattern(p['pattern']),
            sample_rate=p['sample_rate'],
            duration=p['duration'],
            channels=p['channels'],
            frequencies=[(f, a, ph) for f, a, ph in p.get('frequencies', [])]
        )

        return self.generate(params), params.sample_rate


# ============================================================================
# Video Generator (MIT)
# ============================================================================

class VideoGenerator:
    """Generate video from procedural parameters (MIT License)"""

    def __init__(self):
        if not HAS_CV2:
            raise ImportError("OpenCV required: pip install opencv-python")
        self.image_generator = ImageGenerator()
        # Frame cache: stores base images for keyframes to avoid regeneration
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._last_keyframe_id: Optional[int] = None

    def _get_keyframe_id(self, kf_params: Optional[ImageParams]) -> int:
        """Generate a hash ID for keyframe parameters for caching"""
        if kf_params is None:
            return 0
        # Use a tuple of key parameters as cache key
        # Convert list to tuple recursively for hashability (JSON decoder returns lists, not tuples)
        dct_coeffs_tuple = tuple(tuple(c) for c in kf_params.dct_coefficients) if kf_params.dct_coefficients else None
        return hash((
            kf_params.pattern,
            kf_params.width,
            kf_params.height,
            kf_params.channels,
            kf_params.color,
            kf_params.gradient_start,
            kf_params.gradient_end,
            kf_params.gradient_angle,
            kf_params.pattern_scale,
            kf_params.pattern_seed,
            kf_params.dct_coefficients_b64,
            dct_coeffs_tuple
        ))

    def _get_base_frame(self, kf_params: Optional[ImageParams],
                        width: int, height: int) -> np.ndarray:
        """
        Get base frame for keyframe, using cache if parameters unchanged.

        This avoids regenerating expensive procedural images (Perlin noise,
        DCT reconstruction) for every frame when the keyframe hasn't changed.
        """
        if kf_params is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        kf_id = self._get_keyframe_id(kf_params)

        # Check cache
        if kf_id in self._frame_cache:
            return self._frame_cache[kf_id].copy()

        # Generate and cache
        frame = self.image_generator.generate(kf_params)
        self._frame_cache[kf_id] = frame

        # Limit cache size (keep only last 10 keyframes)
        if len(self._frame_cache) > 10:
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]

        return frame.copy()

    def generate(self, params: VideoParams, output_path: Union[str, Path]):
        """Generate video from procedural parameters with frame caching"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), fourcc, params.fps,
            (params.width, params.height)
        )

        # Clear cache at start of new video
        self._frame_cache.clear()

        total_frames = int(params.duration * params.fps)

        for frame_idx in range(total_frames):
            t = frame_idx / params.fps

            prev_kf = None
            next_kf = None

            for kf_time, kf_params in params.keyframes:
                if kf_time <= t:
                    prev_kf = (kf_time, kf_params)
                if kf_time > t and next_kf is None:
                    next_kf = (kf_time, kf_params)
                    break

            if prev_kf is None:
                prev_kf = params.keyframes[0] if params.keyframes else (0, None)
            if next_kf is None:
                next_kf = params.keyframes[-1] if params.keyframes else (0, None)

            # Use cached base frame instead of regenerating every time
            frame = self._get_base_frame(prev_kf[1], params.width, params.height)

            # Apply motion transforms (these are cheap compared to regeneration)
            if params.pattern == VideoPattern.SLIDE and params.motion_vectors:
                mv = params.motion_vectors[min(frame_idx, len(params.motion_vectors)-1)]
                dx = int(mv['mean_dx'] * frame_idx)
                dy = int(mv['mean_dy'] * frame_idx)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, M, (params.width, params.height))

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        # Clear cache after video generation
        self._frame_cache.clear()

    def decompress(self, data: bytes, output_path: Union[str, Path]):
        """Decompress ALICE format to video"""
        parsed = json.loads(data.decode('utf-8'))

        if parsed.get('magic') != ALICE_VIDEO_MAGIC.decode():
            raise ValueError("Invalid ALICE_VID file")

        keyframes = []
        for t, kf_dict in parsed.get('keyframes', []):
            img_params = ImageParams(
                pattern=ImagePattern(kf_dict['pattern']),
                width=kf_dict['width'],
                height=kf_dict['height'],
                channels=kf_dict['channels'],
                color=tuple(kf_dict.get('color', (0, 0, 0))),
                gradient_start=tuple(kf_dict.get('gradient_start', (0, 0, 0))),
                gradient_end=tuple(kf_dict.get('gradient_end', (255, 255, 255)))
            )
            keyframes.append((t, img_params))

        params = VideoParams(
            pattern=VideoPattern(parsed['pattern']),
            width=parsed['width'],
            height=parsed['height'],
            fps=parsed['fps'],
            duration=parsed['duration'],
            keyframes=keyframes,
            motion_vectors=parsed.get('motion_vectors', [])
        )

        self.generate(params, output_path)


# ============================================================================
# Unified Media Decompressor (MIT)
# ============================================================================

class MediaDecompressor:
    """Unified interface for decompressing images, audio, and video (MIT License)"""

    def __init__(self):
        self._image_generator = None
        self._audio_generator = None
        self._video_generator = None

    @property
    def image(self) -> ImageGenerator:
        if self._image_generator is None:
            self._image_generator = ImageGenerator()
        return self._image_generator

    @property
    def audio(self) -> AudioGenerator:
        if self._audio_generator is None:
            self._audio_generator = AudioGenerator()
        return self._audio_generator

    @property
    def video(self) -> VideoGenerator:
        if self._video_generator is None:
            self._video_generator = VideoGenerator()
        return self._video_generator

    def decompress(self, data: bytes, output_path: Optional[Path] = None):
        """Decompress media from ALICE format"""
        if data.startswith(b'{"magic": "ALICE_IMG"') or b'"magic": "ALICE_IMG"' in data[:50]:
            return self.image.decompress(data)
        elif data.startswith(b'{"magic": "ALICE_AUD"') or b'"magic": "ALICE_AUD"' in data[:50]:
            return self.audio.decompress(data)
        elif data.startswith(b'{"magic": "ALICE_VID"') or b'"magic": "ALICE_VID"' in data[:50]:
            if output_path is None:
                raise ValueError("output_path required for video decompression")
            return self.video.decompress(data, output_path)
        else:
            raise ValueError("Unknown ALICE media format")
