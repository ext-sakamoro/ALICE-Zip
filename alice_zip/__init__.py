"""
ALICE-Zip: Procedural Generation Compression (Core)
====================================================

Store algorithms, not data.

ALICE-Zip is a revolutionary compression system that uses procedural generation
to achieve extreme compression ratios. Instead of storing raw data, it analyzes
patterns and stores the algorithms that can regenerate the original content.

License: MIT (Open Core)
- Basic compression and full decompression are free under MIT license.
- For advanced Pro/Enterprise features, see: https://github.com/ext-sakamoro/ALICE-Zip

Features:
- Compression: Sine wave fitting, polynomial fitting, LZMA fallback
- Decompression: All .alice formats supported
- Images: JPEG/PNG/BMP procedural reconstruction
- Audio: WAV frequency analysis and sine wave synthesis

Example:
    >>> from alice_zip import ALICEZip
    >>> zipper = ALICEZip()
    >>> # Compress data
    >>> compressed = zipper.compress(data)
    >>> # Decompress .alice files
    >>> restored = zipper.decompress(compressed)

Author: Moroya Sakamoto
"""

from .core import (
    ALICEZip,
    AliceFileHeader,
    AliceFileType,
    AlicePayloadType,
    ALICE_MAGIC,
    ALICE_VERSION,
    ALICE_FINGERPRINT,
    HEADER_V1_SIZE,
    HEADER_V2_SIZE,
)

# Generators (MIT License - Free)
from .generators import (
    CompressionEngine,
    DataDomain,
    GeneratorType,
    GeneratorParameters,
    CompressionResult,
    ProceduralGenerator,
    PerlinNoiseGenerator,
    FourierGenerator,
    SineWaveGenerator,
    PolynomialGenerator,
    get_generator,
    decompress_from_params,
    decompress_from_lzma,
)

# Analyzers (MIT License - Free for basic compression)
from .analyzers import (
    analyze_data,
    try_sine_fit,
    try_fourier_fit,
    try_polynomial_fit,
    compress_with_lzma,
    FitResult,
    ProceduralCompressionDesigner,
)

# Media generators (MIT License - Free for decompression)
try:
    from .media_generators import (
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
    )
    HAS_MEDIA_GENERATORS = True
except ImportError:
    HAS_MEDIA_GENERATORS = False

__all__ = [
    # Core
    'ALICEZip',
    'AliceFileHeader',
    'AliceFileType',
    'AlicePayloadType',
    'ALICE_MAGIC',
    'ALICE_VERSION',
    'ALICE_FINGERPRINT',
    'HEADER_V1_SIZE',
    'HEADER_V2_SIZE',

    # Generators (MIT)
    'CompressionEngine',
    'DataDomain',
    'GeneratorType',
    'GeneratorParameters',
    'CompressionResult',
    'ProceduralGenerator',
    'PerlinNoiseGenerator',
    'FourierGenerator',
    'SineWaveGenerator',
    'PolynomialGenerator',
    'get_generator',
    'decompress_from_params',
    'decompress_from_lzma',

    # Analyzers (MIT)
    'analyze_data',
    'try_sine_fit',
    'try_fourier_fit',
    'try_polynomial_fit',
    'compress_with_lzma',
    'FitResult',
    'ProceduralCompressionDesigner',

    # Media Generators availability flag
    'HAS_MEDIA_GENERATORS',
]

# Add Media Generators to __all__ only if available
if HAS_MEDIA_GENERATORS:
    __all__.extend([
        'ImagePattern',
        'AudioPattern',
        'VideoPattern',
        'ImageParams',
        'AudioParams',
        'VideoParams',
        'ImageGenerator',
        'AudioGenerator',
        'VideoGenerator',
        'MediaDecompressor',
    ])

__version__ = '1.0.0'
__author__ = 'Moroya Sakamoto'
__license__ = 'MIT'
