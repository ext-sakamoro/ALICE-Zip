#!/usr/bin/env python3
"""
ALICE-Zip: Procedural Generators (MIT License)
===============================================

Generation logic for procedural compression.
This module contains the "hands and feet" - the code that regenerates
data from parameters. Required for decompression.

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

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np

from . import native_accelerator as accel

logger = logging.getLogger(__name__)

# Log native acceleration status
if accel.is_available():
    logger.info("Native acceleration (libalice) is available")
else:
    logger.info("Native acceleration not available, using pure Python")


# ============================================================================
# Enums
# ============================================================================

class CompressionEngine(Enum):
    """Available compression engines"""
    PROCEDURAL = "procedural_generation"
    NEURAL_CODEC = "neural_codec"
    FUNCTION_FITTER = "function_fitter"
    FALLBACK_LZMA = "fallback_lzma"


class DataDomain(Enum):
    """Data domain types"""
    TEXTURE_PATTERN = "texture_pattern"
    NATURAL_IMAGE = "natural_image"
    AUDIO_MUSIC = "audio_music"
    MODEL_3D = "3d_model"
    NUMERICAL = "numerical_data"
    TEXT = "text"
    GENERIC = "generic"


class GeneratorType(Enum):
    """Procedural generator types"""
    PERLIN_NOISE = "perlin_noise"
    SIMPLEX_NOISE = "simplex_noise"
    FRACTAL = "fractal"
    L_SYSTEM = "l_system"
    CELLULAR_AUTOMATA = "cellular_automata"
    WAVE_FUNCTION = "wave_function"
    FOURIER = "fourier"
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GeneratorParameters:
    """Parameters for a procedural generator"""
    generator_type: GeneratorType
    seed: int
    parameters: Dict[str, Any]
    output_shape: Tuple[int, ...]
    dtype: str = "float32"


@dataclass
class CompressionResult:
    """Result of compression operation"""
    success: bool
    engine_used: CompressionEngine
    original_size: int
    compressed_size: int
    compression_ratio: float
    generator_params: Optional[GeneratorParameters] = None
    compressed_data: Optional[bytes] = None
    residual_data: Optional[bytes] = None  # Compressed residual for lossless reconstruction
    error_metric: float = 0.0
    is_lossless: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_residual(self) -> bool:
        """Check if residual data is present for lossless reconstruction"""
        return self.residual_data is not None and len(self.residual_data) > 0

    @property
    def total_compressed_size(self) -> int:
        """Total size including residual"""
        base_size = self.compressed_size
        if self.residual_data:
            base_size += len(self.residual_data)
        return base_size

    @property
    def effective_ratio(self) -> float:
        """Compression ratio including residual"""
        total = self.total_compressed_size
        return self.original_size / total if total > 0 else 0.0


# ============================================================================
# Generator Base Class
# ============================================================================

class ProceduralGenerator(ABC):
    """
    Base class for procedural generators.

    Subclasses implement generate() to create data from parameters.
    The fit() method (for finding parameters) is in analyzers.py (Commercial).
    """

    @abstractmethod
    def generate(self, params: GeneratorParameters) -> np.ndarray:
        """Generate data from parameters"""
        pass


# ============================================================================
# Perlin Noise Generator
# ============================================================================

class PerlinNoiseGenerator(ProceduralGenerator):
    """Perlin noise generator for texture/noise patterns (uses native acceleration)"""

    def generate(self, params: GeneratorParameters) -> np.ndarray:
        """Generate Perlin noise from parameters using native acceleration"""
        shape = params.output_shape
        scale = params.parameters.get('scale', 10.0)
        octaves = params.parameters.get('octaves', 4)
        persistence = params.parameters.get('persistence', 0.5)
        lacunarity = params.parameters.get('lacunarity', 2.0)

        width = shape[1] if len(shape) >= 2 else shape[0]
        height = shape[0] if len(shape) >= 2 else 1

        # Use native accelerator (Rust) if available, otherwise pure Python
        result = accel.perlin_advanced(
            width=width,
            height=height,
            seed=params.seed,
            scale=scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )

        return result.astype(np.float32)


# ============================================================================
# Fourier Generator
# ============================================================================

class FourierGenerator(ProceduralGenerator):
    """Fourier series generator for periodic data (uses native acceleration)"""

    def generate(self, params: GeneratorParameters) -> np.ndarray:
        """Generate data from Fourier coefficients using native acceleration"""
        # Calculate total elements for multi-dimensional support
        n = int(np.prod(params.output_shape))
        coefficients = params.parameters.get('coefficients', [])
        dc_offset = params.parameters.get('dc_offset', 0.0)

        # Use native accelerator (Rust) if available
        result = accel.fourier_generate(n, coefficients, dc_offset)
        # Reshape to original shape
        return result.astype(np.float32).reshape(params.output_shape)


# ============================================================================
# Sine Wave Generator
# ============================================================================

class SineWaveGenerator(ProceduralGenerator):
    """Simple sine wave generator (uses native acceleration)"""

    def generate(self, params: GeneratorParameters) -> np.ndarray:
        """Generate sine wave from parameters using native acceleration"""
        # Calculate total elements for multi-dimensional support
        n = int(np.prod(params.output_shape))
        amplitude = params.parameters.get('amplitude', 1.0)
        frequency = params.parameters.get('frequency', 1.0)
        phase = params.parameters.get('phase', 0.0)
        dc_offset = params.parameters.get('dc_offset', 0.0)

        # Use native accelerator (Rust) if available
        result = accel.sine_wave(n, frequency, amplitude, phase, dc_offset)
        # Reshape to original shape
        return result.astype(np.float32).reshape(params.output_shape)


# ============================================================================
# Polynomial Generator
# ============================================================================

class PolynomialGenerator(ProceduralGenerator):
    """Polynomial generator for smooth numerical data (uses native acceleration)"""

    def generate(self, params: GeneratorParameters) -> np.ndarray:
        """Generate data from polynomial coefficients using native acceleration"""
        # Calculate total elements for multi-dimensional support
        n = int(np.prod(params.output_shape))
        coefficients = params.parameters.get('coefficients', [0])

        # Use native accelerator (Rust) if available
        result = accel.polynomial_generate(n, coefficients)
        # Reshape to original shape
        return result.astype(np.float32).reshape(params.output_shape)


# ============================================================================
# Generator Factory (Open-Closed Principle)
# ============================================================================

class GeneratorFactory:
    """
    Factory for creating procedural generators.

    Uses registry pattern for extensibility - new generators can be
    registered without modifying existing code (Open-Closed Principle).
    """
    _registry: Dict[GeneratorType, type] = {}
    _condition_handlers: Dict[str, type] = {}

    @classmethod
    def register(cls, generator_type: GeneratorType):
        """Decorator to register a generator class for a type"""
        def decorator(generator_class: type):
            cls._registry[generator_type] = generator_class
            return generator_class
        return decorator

    @classmethod
    def register_condition(cls, condition_key: str):
        """Decorator to register a generator for a parameter condition"""
        def decorator(generator_class: type):
            cls._condition_handlers[condition_key] = generator_class
            return generator_class
        return decorator

    @classmethod
    def get(cls, params: GeneratorParameters) -> ProceduralGenerator:
        """
        Get appropriate generator for the given parameters.

        First checks condition handlers, then falls back to type registry.
        """
        # Check condition handlers first (e.g., is_simple_sine)
        for condition_key, generator_class in cls._condition_handlers.items():
            if params.parameters.get(condition_key, False):
                return generator_class()

        # Fall back to type-based registry
        generator_type = params.generator_type
        if generator_type in cls._registry:
            return cls._registry[generator_type]()

        raise ValueError(f"Unknown generator type: {generator_type}")

    @classmethod
    def register_defaults(cls):
        """Register all default generators"""
        cls._registry[GeneratorType.PERLIN_NOISE] = PerlinNoiseGenerator
        cls._registry[GeneratorType.FOURIER] = FourierGenerator
        cls._registry[GeneratorType.POLYNOMIAL] = PolynomialGenerator
        cls._registry[GeneratorType.WAVE_FUNCTION] = SineWaveGenerator
        # Condition handlers for special cases
        cls._condition_handlers['is_simple_sine'] = SineWaveGenerator


# Initialize default registrations
GeneratorFactory.register_defaults()


def get_generator(generator_type: GeneratorType) -> ProceduralGenerator:
    """
    Get a generator instance by type.

    DEPRECATED: Use GeneratorFactory.get(params) for full functionality.
    Kept for backward compatibility.
    """
    return GeneratorFactory.get(GeneratorParameters(
        generator_type=generator_type,
        seed=0,
        parameters={},
        output_shape=(1,)
    ))


def decompress_from_params(params: GeneratorParameters) -> np.ndarray:
    """
    Decompress (regenerate) data from generator parameters.

    This is the core decompression function - MIT licensed.
    Uses GeneratorFactory for extensible generator selection.
    """
    generator = GeneratorFactory.get(params)
    return generator.generate(params)


def decompress_from_lzma(compressed_data: bytes, shape: Tuple, dtype: str = 'float32') -> np.ndarray:
    """
    Decompress LZMA-compressed raw data.

    This is the fallback decompression - MIT licensed.
    """
    import lzma
    raw = lzma.decompress(compressed_data)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)
