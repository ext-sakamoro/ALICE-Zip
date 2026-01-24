#!/usr/bin/env python3
"""
ALICE-Zip: Procedural Generation Compression (Core)
====================================================

A revolutionary compression system that stores generation algorithms
instead of raw data, achieving compression ratios beyond traditional limits.

File Format: .alice
- Magic: ALICE_ZIP
- Version: 1.0
- Engine-specific payload

Author: Moroya Sakamoto
License: MIT

This module provides both compression and decompression functionality.
Basic compression (sine wave, polynomial, LZMA) is free under MIT license.
For advanced compression features, see ALICE Optimizer (Pro).
"""

import struct
import hashlib
import json
import logging
import lzma
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Union, BinaryIO, Any, Dict
from enum import Enum
import numpy as np

# Generators (MIT License - Free for decompression)
from .generators import (
    CompressionResult,
    CompressionEngine,
    GeneratorParameters,
    GeneratorType,
    decompress_from_params,
    decompress_from_lzma,
)

# Analyzers (MIT License - Free for basic compression)
from .analyzers import analyze_data, compress_with_lzma

# Media Decompressor (MIT License - Free for decompression)
from .media_generators import (
    MediaDecompressor,
    ALICE_IMAGE_MAGIC,
    ALICE_AUDIO_MAGIC,
    ALICE_VIDEO_MAGIC,
)


# ============================================================================
# File Format Constants
# ============================================================================

ALICE_MAGIC = b'ALICE_ZIP'
ALICE_VERSION = (1, 0)

# ALICE Fingerprint
ALICE_FINGERPRINT = {
    'project': 'ALICE-Zip',
    'full_name': 'Procedural Generation Compression',
    'philosophy': 'Procedural Generation over Dictionary Compression',
    'principle': 'Store algorithms, not data - Kolmogorov Complexity Approximation',
    'signature': 'ALICE-PGC-v1.0',
}


class AliceFileType(Enum):
    """File types supported by ALICE_Zip"""
    NUMPY_ARRAY = 0x01
    IMAGE = 0x02
    AUDIO = 0x03
    TEXT = 0x04
    BINARY = 0x05


class AlicePayloadType(Enum):
    """
    Payload type identifier for unified format handling.

    This enum determines how the payload should be parsed and which
    decompressor should be used.
    """
    # Procedural parameters (original ALICE_ZIP format)
    PROCEDURAL = 0x00

    # Media formats (JSON-based, dispatched to MediaDecompressor)
    MEDIA_IMAGE = 0x10      # ALICE_IMG
    MEDIA_AUDIO = 0x11      # ALICE_AUD
    MEDIA_VIDEO = 0x12      # ALICE_VID

    # Texture format (JSON-based)
    TEXTURE = 0x20          # ALICE_TEX

    # LZMA-compressed fallback
    LZMA_FALLBACK = 0x30


# ============================================================================
# File Header
# ============================================================================
#
# Version Compatibility Notes:
# ---------------------------
# The .alice file format uses a versioned header with automatic detection.
#
# VERSIONING STRATEGY:
#   - Minor version (1.x): Backward compatible changes only
#   - Major version (x.0): May break backward compatibility
#
# FORWARD COMPATIBILITY:
#   - Readers SHOULD ignore unknown fields at the end of headers
#   - New fields MUST be appended, never inserted
#   - New enum values can be added; readers should handle unknown values gracefully
#
# ADDING A NEW HEADER VERSION (v3+):
#   1. Define HEADER_V3_SIZE with the new size
#   2. Update from_bytes() to detect v3 by size and/or version number
#   3. Maintain backward compatibility: v3 readers must parse v1/v2 files
#   4. Update to_bytes() to write in the latest format
#   5. Document the binary layout changes
#
# VERSION DETECTION:
#   - v1 (65 bytes): Original format, payload_type inferred from content
#   - v2 (66 bytes): version >= 1.1, explicit payload_type field
#   - Detection uses combination of data length and version_minor field

# Header format versions
HEADER_V1_SIZE = 65  # Original format without payload_type
HEADER_V2_SIZE = 66  # Extended format with payload_type


@dataclass
class AliceFileHeader:
    """
    Header for .alice files.

    Format v1 (65 bytes): Original format, payload_type inferred from content
    Format v2 (66 bytes): Extended format with explicit payload_type

    Binary layout (v2):
        - magic: 9 bytes ('ALICE_ZIP')
        - version_major: 1 byte
        - version_minor: 1 byte
        - file_type: 1 byte (AliceFileType)
        - engine: 1 byte (CompressionEngine index)
        - payload_type: 1 byte (AlicePayloadType) [v2 only]
        - original_size: 8 bytes (uint64)
        - compressed_size: 8 bytes (uint64)
        - original_hash: 32 bytes (SHA-256)
        - flags: 4 bytes (uint32, reserved for future use)

    Compatibility:
        - All fields use little-endian byte order
        - flags field is reserved; readers should ignore unknown flag bits
        - Unknown payload_type values should fall back to PROCEDURAL handling
        - Future versions may extend the header; always use header_size() method
    """
    magic: bytes = ALICE_MAGIC
    version_major: int = ALICE_VERSION[0]
    version_minor: int = ALICE_VERSION[1]
    file_type: AliceFileType = AliceFileType.NUMPY_ARRAY
    engine: CompressionEngine = CompressionEngine.PROCEDURAL
    payload_type: AlicePayloadType = AlicePayloadType.PROCEDURAL
    original_size: int = 0
    compressed_size: int = 0
    original_hash: bytes = b'\x00' * 32
    flags: int = 0

    # Track which version this header was loaded from
    _header_version: int = 2

    def to_bytes(self) -> bytes:
        """
        Serialize header to bytes (v2 format with payload_type).

        Returns:
            66-byte header in v2 format
        """
        return struct.pack(
            '<9sBBBBB Q Q 32s I',
            self.magic,
            self.version_major,
            self.version_minor,
            self.file_type.value,
            list(CompressionEngine).index(self.engine),
            self.payload_type.value,
            self.original_size,
            self.compressed_size,
            self.original_hash,
            self.flags
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AliceFileHeader':
        """
        Deserialize header from bytes with automatic version detection.

        Supports both v1 (65 bytes) and v2 (66 bytes) formats.
        v1 headers have payload_type inferred as PROCEDURAL.

        Args:
            data: Raw bytes containing the header

        Returns:
            Parsed AliceFileHeader

        Raises:
            ValueError: If magic number is invalid or data is too short
        """
        if len(data) < HEADER_V1_SIZE:
            raise ValueError(f"Data too short for header: {len(data)} bytes")

        # First, check magic number (common to both versions)
        magic = data[:9]
        if magic != ALICE_MAGIC:
            raise ValueError(f"Invalid ALICE_ZIP file (magic: {magic})")

        # Check version to determine format
        ver_maj = data[9]
        ver_min = data[10]

        # v2 format: version >= 1.1 uses 66-byte header with payload_type
        # v1 format: version 1.0 uses 65-byte header without payload_type
        is_v2 = (ver_maj > 1) or (ver_maj == 1 and ver_min >= 1)

        if is_v2 and len(data) >= HEADER_V2_SIZE:
            # Parse v2 format (66 bytes)
            (magic, ver_maj, ver_min, file_type, engine_idx, payload_type,
             orig_size, comp_size, orig_hash, flags) = struct.unpack(
                '<9sBBBBB Q Q 32s I',
                data[:HEADER_V2_SIZE]
            )

            try:
                payload_type_enum = AlicePayloadType(payload_type)
            except ValueError:
                # Unknown payload type, default to PROCEDURAL
                payload_type_enum = AlicePayloadType.PROCEDURAL

            header = cls(
                magic=magic,
                version_major=ver_maj,
                version_minor=ver_min,
                file_type=AliceFileType(file_type),
                engine=list(CompressionEngine)[engine_idx],
                payload_type=payload_type_enum,
                original_size=orig_size,
                compressed_size=comp_size,
                original_hash=orig_hash,
                flags=flags
            )
            header._header_version = 2
            return header
        else:
            # Parse v1 format (65 bytes)
            (magic, ver_maj, ver_min, file_type, engine_idx,
             orig_size, comp_size, orig_hash, flags) = struct.unpack(
                '<9sBBBB Q Q 32s I',
                data[:HEADER_V1_SIZE]
            )

            header = cls(
                magic=magic,
                version_major=ver_maj,
                version_minor=ver_min,
                file_type=AliceFileType(file_type),
                engine=list(CompressionEngine)[engine_idx],
                payload_type=AlicePayloadType.PROCEDURAL,  # Default for v1
                original_size=orig_size,
                compressed_size=comp_size,
                original_hash=orig_hash,
                flags=flags
            )
            header._header_version = 1
            return header

    @classmethod
    def size(cls, version: int = 2) -> int:
        """
        Return header size in bytes for the specified version.

        Args:
            version: Header format version (1 or 2)

        Returns:
            Header size in bytes (65 for v1, 66 for v2)
        """
        return HEADER_V2_SIZE if version == 2 else HEADER_V1_SIZE

    def header_size(self) -> int:
        """Return the header size for this instance's version."""
        return HEADER_V2_SIZE if self._header_version == 2 else HEADER_V1_SIZE


# ============================================================================
# Main ALICE_Zip Class (Core - Decompression Only)
# ============================================================================

class ALICEZip:
    """
    ALICE_Zip compression/decompression engine (Core/MIT version).

    This is the Open Core (free) version that supports:
    - Basic compression (sine wave, polynomial fitting, LZMA fallback)
    - Full decompression of all .alice formats

    For advanced compression features, see ALICE Optimizer (Pro).

    Supports multiple payload types:
    - Procedural parameters (original ALICE_ZIP)
    - Media formats (ALICE_IMG, ALICE_AUD, ALICE_VID)
    - Texture format (ALICE_TEX)

    Example:
        >>> zipper = ALICEZip()
        >>> # Compress
        >>> compressed = zipper.compress(data)
        >>> # Decompress
        >>> restored = zipper.decompress(compressed)
    """

    def __init__(self):
        self._media_decompressor = None

    @property
    def media(self) -> MediaDecompressor:
        """Lazy-loaded media decompressor for image/audio/video formats."""
        if self._media_decompressor is None:
            self._media_decompressor = MediaDecompressor()
        return self._media_decompressor

    def compress(self, data: Union[np.ndarray, bytes, str],
                 output_path: Optional[Path] = None) -> bytes:
        """
        Compress data to .alice format.

        Supports:
        - numpy arrays (numerical data)
        - bytes (raw binary data)

        Compression methods (tried in order):
        - Sine wave fitting (for periodic data)
        - Polynomial fitting (for smooth curves)
        - LZMA fallback (for non-procedural data)

        Args:
            data: Input data to compress
            output_path: Optional path to write compressed file

        Returns:
            Compressed data as bytes

        Example:
            >>> zipper = ALICEZip()
            >>> compressed = zipper.compress(np.sin(np.linspace(0, 10, 1000)))
        """
        # Convert input to numpy array
        if isinstance(data, bytes):
            arr = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, str):
            arr = np.frombuffer(data.encode('utf-8'), dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        original_hash = hashlib.sha256(arr.tobytes()).digest()

        # Analyze and compress
        result = analyze_data(arr)

        if result.engine_used == CompressionEngine.PROCEDURAL:
            # Procedural compression - serialize parameters
            payload = self._serialize_procedural_params(result.generator_params)
            payload_type = AlicePayloadType.PROCEDURAL
        else:
            # LZMA fallback
            payload = self._serialize_lzma_payload(result, arr)
            payload_type = AlicePayloadType.LZMA_FALLBACK

        # Create header
        header = AliceFileHeader(
            magic=ALICE_MAGIC,
            version_major=1,
            version_minor=1,  # v1.1 for v2 header format
            file_type=AliceFileType.NUMPY_ARRAY,
            engine=result.engine_used,
            payload_type=payload_type,
            original_size=result.original_size,
            compressed_size=len(payload),
            original_hash=original_hash,
            flags=0
        )

        compressed = header.to_bytes() + payload

        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'wb') as f:
                f.write(compressed)

        return compressed

    def _serialize_procedural_params(self, params: GeneratorParameters) -> bytes:
        """Serialize procedural generator parameters to JSON bytes."""
        payload = {
            'type': 'generator',
            'params': {
                'generator_type': params.generator_type.value,
                'seed': params.seed,
                'parameters': params.parameters,
                'output_shape': list(params.output_shape),
                'dtype': params.dtype
            }
        }
        return json.dumps(payload, separators=(',', ':')).encode('utf-8')

    def _serialize_lzma_payload(self, result: CompressionResult, data: np.ndarray) -> bytes:
        """Serialize LZMA-compressed data with metadata."""
        metadata = {
            'shape': list(data.shape),
            'dtype': str(data.dtype)
        }
        meta_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')

        # Format: 4-byte meta length + meta JSON + compressed data
        meta_len = struct.pack('<I', len(meta_json))
        return meta_len + meta_json + result.compressed_data

    # Soft limit for warning about large files (1GB)
    LARGE_FILE_WARNING_SIZE = 1024 * 1024 * 1024
    # Maximum JSON payload size (for procedural parameters) - 100MB should be plenty
    MAX_JSON_PAYLOAD_SIZE = 100 * 1024 * 1024
    # Maximum output dimension for generated data (prevents memory exhaustion attacks)
    # 65536 x 65536 x 4 bytes = 16GB max allocation (reasonable upper bound)
    MAX_OUTPUT_DIMENSION = 10_000_000  # 10M elements per dimension (supports 4K images, long audio)

    def decompress(self, data: Union[bytes, Path, str],
                   output_path: Optional[Path] = None) -> Union[np.ndarray, Any]:
        """
        Decompress .alice file with automatic format detection and dispatch.

        Supports:
        - ALICE_ZIP (binary header + procedural/LZMA payload)
        - ALICE_IMG (JSON-based image parameters)
        - ALICE_AUD (JSON-based audio parameters)
        - ALICE_VID (JSON-based video parameters)
        - ALICE_TEX (JSON-based texture parameters)

        Args:
            data: Compressed data (bytes) or path to .alice file
            output_path: Optional output path for video decompression

        Returns:
            Decompressed data (type depends on format):
            - ALICE_ZIP/ALICE_IMG: numpy array
            - ALICE_AUD: tuple of (numpy array, sample_rate)
            - ALICE_VID: None (writes to output_path)

        Raises:
            ValueError: If file is invalid or corrupted
            FileNotFoundError: If file path doesn't exist
        """
        import logging
        import mmap
        logger = logging.getLogger(__name__)

        # Read from file if path - use streaming for binary format
        if isinstance(data, (Path, str)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            file_size = path.stat().st_size

            # Warn about very large files but don't block
            if file_size > self.LARGE_FILE_WARNING_SIZE:
                logger.warning(
                    f"Large file detected ({file_size / (1024**3):.2f} GB). "
                    f"Processing may be slow and memory-intensive."
                )

            # Streaming read: peek at magic to determine format
            with open(path, 'rb') as f:
                magic_peek = f.read(16)  # Read enough to detect format

                # Check for binary ALICE_ZIP format - use streaming
                if magic_peek.startswith(ALICE_MAGIC):
                    return self._decompress_binary_stream(f, magic_peek, file_size, output_path)

                # JSON format - check size limit for JSON payloads
                if file_size > self.MAX_JSON_PAYLOAD_SIZE:
                    raise MemoryError(
                        f"JSON payload too large ({file_size} bytes). "
                        f"Maximum JSON payload: {self.MAX_JSON_PAYLOAD_SIZE} bytes. "
                        f"Consider using binary format for large data."
                    )

                # Read JSON file into memory (size already validated)
                f.seek(0)
                data = f.read()

        # Check minimum file size
        if len(data) < 10:
            raise ValueError(f"Invalid ALICE file: too small ({len(data)} bytes)")

        # =====================================================================
        # Format Detection and Dispatch (for in-memory bytes)
        # =====================================================================

        # Check for JSON-based media formats (ALICE_IMG, ALICE_AUD, ALICE_VID, ALICE_TEX)
        # Use lstrip() to handle leading whitespace (json.loads tolerates it)
        if data.lstrip().startswith(b'{'):
            return self._decompress_media_json(data, output_path)

        # Check for binary ALICE_ZIP format
        if data.startswith(ALICE_MAGIC):
            return self._decompress_binary_format(data, output_path)

        # Unknown format
        raise ValueError("Unknown ALICE file format")

    def _decompress_binary_stream(self, f: BinaryIO, magic_peek: bytes,
                                  file_size: int, output_path: Optional[Path]) -> Union[np.ndarray, Any]:
        """
        Decompress binary ALICE_ZIP format using streaming I/O.

        Reads header first, validates, then reads payload.
        For large payloads (>100MB), uses memory-mapped I/O for efficiency.

        Args:
            f: Open file handle (positioned after magic_peek)
            magic_peek: Already-read bytes for format detection
            file_size: Total file size for validation
            output_path: Output path for video decompression

        Returns:
            Decompressed data
        """
        import mmap

        # Seek back and read header only
        f.seek(0)
        header_data = f.read(HEADER_V2_SIZE)

        if len(header_data) < HEADER_V1_SIZE:
            raise ValueError(f"File too small for header: {len(header_data)} bytes")

        # Parse header
        try:
            header = AliceFileHeader.from_bytes(header_data)
        except struct.error as e:
            raise ValueError(f"Invalid ALICE_ZIP file header: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid ALICE_ZIP file: {e}")

        # Get actual header size and seek to payload
        header_size = header.header_size()
        expected_payload_size = file_size - header_size

        if header.compressed_size > expected_payload_size:
            raise ValueError(
                f"Header claims {header.compressed_size} bytes payload, "
                f"but only {expected_payload_size} bytes available"
            )

        actual_payload_size = header.compressed_size if header.compressed_size > 0 else expected_payload_size

        # For large payloads, use mmap to avoid loading everything into RAM
        # For small payloads, direct read is faster
        MMAP_THRESHOLD = 100 * 1024 * 1024  # 100MB

        if actual_payload_size > MMAP_THRESHOLD:
            # Use memory-mapped I/O for large payloads
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            payload_data = mm[header_size:header_size + actual_payload_size]
            result = self._dispatch_payload(header, bytes(payload_data), output_path)
            mm.close()
            return result
        else:
            # Direct read for small payloads
            f.seek(header_size)
            payload_data = f.read(actual_payload_size)
            return self._dispatch_payload(header, payload_data, output_path)

    def _decompress_media_json(self, data: bytes,
                               output_path: Optional[Path]) -> Union[np.ndarray, Any]:
        """
        Decompress JSON-based media formats (ALICE_IMG, ALICE_AUD, ALICE_VID, ALICE_TEX).

        Args:
            data: Raw bytes containing JSON payload
            output_path: Output path for video decompression

        Returns:
            Decompressed data appropriate to the format
        """
        # Size already validated by caller (MAX_PAYLOAD_SIZE)
        # Decode and parse JSON
        try:
            text = data.decode('utf-8')
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON media format: {e}")

        if not isinstance(parsed, dict):
            raise ValueError("JSON payload must be an object")

        magic = parsed.get('magic', '')

        # Dispatch based on magic string
        if magic == ALICE_IMAGE_MAGIC.decode():
            return self.media.image.decompress(data)

        elif magic == ALICE_AUDIO_MAGIC.decode():
            return self.media.audio.decompress(data)

        elif magic == ALICE_VIDEO_MAGIC.decode():
            if output_path is None:
                raise ValueError("output_path required for video decompression")
            return self.media.video.decompress(data, output_path)

        elif magic == 'ALICE_TEX':
            return self._decompress_texture(parsed)

        else:
            raise ValueError(f"Unknown media format magic: {magic}")

    def _decompress_texture(self, parsed: Dict[str, Any]) -> np.ndarray:
        """
        Decompress ALICE_TEX texture format.

        Args:
            parsed: Parsed JSON data

        Returns:
            numpy array representing the texture
        """
        # ALICE_TEX format requires Commercial version for full support
        # But we can support basic procedural textures
        params = parsed.get('params', {})
        width = params.get('width', 256)
        height = params.get('height', 256)
        texture_type = params.get('texture_type', 'solid')

        # Basic fallback: return gray texture
        # Full procedural texture generation requires Commercial version
        logger = logging.getLogger(__name__)

        if texture_type == 'solid':
            color = params.get('color', [128, 128, 128])
            return np.full((height, width, 3), color, dtype=np.uint8)
        else:
            # For complex textures, return placeholder with warning
            logger.warning(
                f"Unsupported texture type '{texture_type}' - returning gray placeholder. "
                f"Full procedural texture generation requires ALICE-Zip Pro. "
                f"Supported types in Core: 'solid'"
            )
            return np.full((height, width, 3), 128, dtype=np.uint8)

    def _decompress_binary_format(self, data: bytes,
                                  output_path: Optional[Path]) -> Union[np.ndarray, Any]:
        """
        Decompress binary ALICE_ZIP format (header + payload) from in-memory bytes.

        Args:
            data: Raw bytes with ALICE_ZIP header and payload
            output_path: Output path for video decompression

        Returns:
            Decompressed data
        """
        min_size = HEADER_V1_SIZE
        if len(data) < min_size:
            raise ValueError(f"Invalid ALICE_ZIP file: too small ({len(data)} bytes)")

        # Parse header
        try:
            header = AliceFileHeader.from_bytes(data)
        except struct.error as e:
            raise ValueError(f"Invalid ALICE_ZIP file header: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid ALICE_ZIP file: {e}")

        # Get header size for this version
        header_size = header.header_size()

        # Validate header consistency
        if header.compressed_size > len(data) - header_size:
            raise ValueError(
                f"Header claims {header.compressed_size} bytes payload, "
                f"but only {len(data) - header_size} bytes available"
            )

        # Extract payload
        payload_data = data[header_size:]

        # Check payload size before decoding
        if len(payload_data) > self.MAX_JSON_PAYLOAD_SIZE:
            raise MemoryError(f"Payload too large: {len(payload_data)} bytes (max: {self.MAX_JSON_PAYLOAD_SIZE})")

        return self._dispatch_payload(header, payload_data, output_path)

    def _dispatch_payload(self, header: AliceFileHeader, payload_data: bytes,
                          output_path: Optional[Path]) -> Union[np.ndarray, Any]:
        """
        Dispatch decompression based on payload type.

        Args:
            header: Parsed file header
            payload_data: Raw payload bytes
            output_path: Output path for video decompression

        Returns:
            Decompressed data
        """
        # Dispatch based on payload_type (v2) or content detection (v1)
        if header.payload_type == AlicePayloadType.MEDIA_IMAGE:
            return self.media.image.decompress(payload_data)

        elif header.payload_type == AlicePayloadType.MEDIA_AUDIO:
            return self.media.audio.decompress(payload_data)

        elif header.payload_type == AlicePayloadType.MEDIA_VIDEO:
            if output_path is None:
                raise ValueError("output_path required for video decompression")
            return self.media.video.decompress(payload_data, output_path)

        elif header.payload_type == AlicePayloadType.TEXTURE:
            try:
                text = payload_data.decode('utf-8')
                parsed = json.loads(text)
                return self._decompress_texture(parsed)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise ValueError(f"Invalid texture payload: {e}")

        elif header.payload_type == AlicePayloadType.LZMA_FALLBACK:
            # LZMA fallback: binary format (meta_len + meta JSON + LZMA data)
            return self._decompress_lzma_fallback(payload_data)

        else:
            # PROCEDURAL: parse JSON payload
            return self._decompress_procedural_payload(payload_data)

    def _decompress_procedural_payload(self, payload_data: bytes) -> np.ndarray:
        """
        Decompress procedural/LZMA payload (original ALICE_ZIP format).

        Args:
            payload_data: Raw payload bytes (JSON-encoded)

        Returns:
            Decompressed numpy array
        """
        # Decode UTF-8 with error handling
        try:
            payload_text = payload_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid payload encoding: {e}")

        # Parse JSON (size already validated by MAX_PAYLOAD_SIZE)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON payload: {e}")

        # Validate required fields
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")
        if 'type' not in payload:
            raise ValueError("Payload missing 'type' field")

        # Decompress based on type
        if payload['type'] == 'generator':
            return self._decompress_generator(payload)
        else:
            return self._decompress_lzma(payload)

    def _decompress_generator(self, payload: Dict[str, Any]) -> np.ndarray:
        """Decompress generator-based payload."""
        if 'params' not in payload:
            raise ValueError("Generator payload missing 'params' field")

        params_dict = payload['params']

        # Validate required fields
        required_fields = ['generator_type', 'seed', 'parameters', 'output_shape']
        for field in required_fields:
            if field not in params_dict:
                raise ValueError(f"Generator params missing '{field}' field")

        gen_type_str = params_dict['generator_type']
        if gen_type_str.startswith('GeneratorType.'):
            gen_type_str = gen_type_str.split('.')[-1].lower()

        try:
            gen_type = GeneratorType(gen_type_str)
        except ValueError:
            raise ValueError(f"Unknown generator type: {gen_type_str}")

        # Validate output shape
        output_shape = params_dict['output_shape']
        if not isinstance(output_shape, (list, tuple)):
            raise ValueError("output_shape must be a list or tuple")
        if not all(isinstance(d, int) and d > 0 for d in output_shape):
            raise ValueError("output_shape must contain positive integers")
        # Prevent memory exhaustion attacks with excessively large dimensions
        if any(d > self.MAX_OUTPUT_DIMENSION for d in output_shape):
            raise ValueError(
                f"output_shape dimension exceeds maximum ({self.MAX_OUTPUT_DIMENSION}): {output_shape}"
            )

        params = GeneratorParameters(
            generator_type=gen_type,
            seed=int(params_dict['seed']),
            parameters=params_dict['parameters'],
            output_shape=tuple(output_shape),
            dtype=params_dict.get('dtype', 'float32')
        )

        return decompress_from_params(params)

    def _decompress_lzma(self, payload: Dict[str, Any]) -> np.ndarray:
        """Decompress LZMA-based payload."""
        if 'data' not in payload:
            raise ValueError("LZMA payload missing 'data' field")

        compressed_hex = payload['data']
        if not isinstance(compressed_hex, str):
            raise ValueError("LZMA data must be a hex string")

        try:
            compressed_bytes = bytes.fromhex(compressed_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex data: {e}")

        metadata = payload.get('metadata', {})
        dtype = metadata.get('dtype', 'float32')
        shape = metadata.get('shape')

        return decompress_from_lzma(compressed_bytes, shape, dtype)

    def _decompress_lzma_fallback(self, payload_data: bytes) -> np.ndarray:
        """
        Decompress LZMA fallback payload (binary format).

        Format: 4-byte meta_len + meta JSON + LZMA compressed data

        Args:
            payload_data: Raw payload bytes

        Returns:
            Decompressed numpy array
        """
        if len(payload_data) < 4:
            raise ValueError("LZMA fallback payload too short")

        # Read metadata length
        meta_len = struct.unpack('<I', payload_data[:4])[0]

        if len(payload_data) < 4 + meta_len:
            raise ValueError("LZMA fallback payload corrupted (metadata truncated)")

        # Parse metadata
        meta_json = payload_data[4:4 + meta_len]
        try:
            metadata = json.loads(meta_json.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid LZMA fallback metadata: {e}")

        # Get shape and dtype
        shape = tuple(metadata.get('shape', []))
        dtype = metadata.get('dtype', 'uint8')

        # Decompress LZMA data
        lzma_data = payload_data[4 + meta_len:]
        try:
            raw_bytes = lzma.decompress(lzma_data)
        except lzma.LZMAError as e:
            raise ValueError(f"LZMA decompression failed: {e}")

        # Reconstruct array
        return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

    def compress_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Compress a file to .alice format.

        Args:
            input_path: Path to input file
            output_path: Optional path for output .alice file

        Returns:
            Path to the compressed .alice file
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix('.alice')
        else:
            output_path = Path(output_path)

        # Read input file
        with open(input_path, 'rb') as f:
            data = f.read()

        # Compress
        compressed = self.compress(np.frombuffer(data, dtype=np.uint8))

        # Write output
        with open(output_path, 'wb') as f:
            f.write(compressed)

        return output_path

    def decompress_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decompress a .alice file."""
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_suffix('')
            if not output_path.suffix:
                output_path = output_path.with_suffix('.bin')

        data = self.decompress(input_path)

        with open(output_path, 'wb') as f:
            f.write(data.tobytes())

        return output_path

    def info(self, path: Union[Path, str]) -> Dict[str, Any]:
        """
        Get information about a .alice file using streaming I/O.

        Supports all ALICE formats:
        - ALICE_ZIP (binary header + payload)
        - ALICE_IMG, ALICE_AUD, ALICE_VID (JSON media formats)
        - ALICE_TEX (JSON texture format)

        Args:
            path: Path to .alice file

        Returns:
            Dictionary containing file metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size < 10:
            raise ValueError("File too small to be valid")

        with open(path, 'rb') as f:
            # Read just enough to detect format
            magic_peek = f.read(16)

            # Standard ALICE_ZIP binary format - only read header
            if magic_peek.startswith(ALICE_MAGIC):
                f.seek(0)
                return self._info_binary_stream(f, file_size)

            # Check for JSON-based format (use lstrip to handle leading whitespace)
            if magic_peek.lstrip().startswith(b'{'):
                # JSON format - need to read entire file but check size first
                if file_size > self.MAX_JSON_PAYLOAD_SIZE:
                    raise MemoryError(
                        f"JSON file too large ({file_size} bytes). "
                        f"Maximum: {self.MAX_JSON_PAYLOAD_SIZE} bytes"
                    )
                f.seek(0)
                data = f.read()
                return self._info_json_format(data)

        raise ValueError("Unknown ALICE file format")

    def _info_binary_stream(self, f: BinaryIO, file_size: int) -> Dict[str, Any]:
        """Get info for binary ALICE_ZIP format using streaming I/O."""
        # Read only header
        header_data = f.read(HEADER_V2_SIZE)

        if len(header_data) < HEADER_V1_SIZE:
            raise ValueError(f"File too small for ALICE_ZIP format: {len(header_data)} bytes")

        try:
            header = AliceFileHeader.from_bytes(header_data)
        except (struct.error, ValueError) as e:
            raise ValueError(f"Invalid ALICE_ZIP header: {e}")

        header_size = header.header_size()
        payload_size = file_size - header_size

        # For procedural generators, payload is small JSON - read it for metadata
        # For large files, just report basic info without parsing payload
        if payload_size <= self.MAX_JSON_PAYLOAD_SIZE:
            f.seek(header_size)
            payload_data = f.read(payload_size)
            try:
                payload_text = payload_data.decode('utf-8')
                payload = json.loads(payload_text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {'type': 'binary', 'note': 'Could not parse payload'}
        else:
            payload = {
                'type': 'large_binary',
                'note': f'Payload too large to parse ({payload_size} bytes)'
            }

        # Calculate compression ratio safely
        compressed_total = header.compressed_size + header_size
        if compressed_total > 0 and header.original_size > 0:
            compression_ratio = header.original_size / compressed_total
        else:
            compression_ratio = 0.0

        return {
            'magic': header.magic.decode('utf-8', errors='replace'),
            'version': f"{header.version_major}.{header.version_minor}",
            'header_version': header._header_version,
            'file_type': header.file_type.name,
            'engine': header.engine.value,
            'payload_type': header.payload_type.name,
            'payload_content_type': payload.get('type', 'unknown'),
            'original_size': header.original_size,
            'compressed_size': compressed_total,
            'file_size': file_size,
            'compression_ratio': compression_ratio,
            'original_hash': header.original_hash.hex(),
            'metadata': payload.get('metadata', {}),
            'alice_fingerprint': payload.get('alice_fingerprint', ALICE_FINGERPRINT)
        }

    def _info_json_format(self, data: bytes) -> Dict[str, Any]:
        """Get info for JSON-based formats."""
        try:
            text = data.decode('utf-8')
            parsed = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON format: {e}")

        if not isinstance(parsed, dict):
            raise ValueError("JSON payload must be an object")

        magic = parsed.get('magic', 'UNKNOWN')
        version = parsed.get('version', [0, 0])
        if isinstance(version, (list, tuple)) and len(version) >= 2:
            version_str = f"{version[0]}.{version[1]}"
        else:
            version_str = str(version)

        # Determine format type and extract info
        if magic == 'ALICE_TEX':
            params = parsed.get('params', {})
            width = params.get('width', 0)
            height = params.get('height', 0)
            original_size = width * height * 3

            return {
                'magic': magic,
                'version': version_str,
                'file_type': 'TEXTURE',
                'engine': 'procedural_texture',
                'payload_type': AlicePayloadType.TEXTURE.name,
                'texture_type': params.get('texture_type', 'unknown'),
                'dimensions': f"{width}x{height}",
                'original_size': original_size,
                'compressed_size': len(data),
                'compression_ratio': original_size / len(data) if len(data) > 0 else 0,
                'original_hash': 'N/A (procedural)',
                'metadata': params,
                'alice_fingerprint': ALICE_FINGERPRINT
            }

        elif magic == ALICE_IMAGE_MAGIC.decode():
            params = parsed.get('params', {})
            width = params.get('width', 0)
            height = params.get('height', 0)
            channels = params.get('channels', 3)
            original_size = parsed.get('original_size', width * height * channels)

            return {
                'magic': magic,
                'version': version_str,
                'file_type': 'IMAGE',
                'engine': 'media_procedural',
                'payload_type': AlicePayloadType.MEDIA_IMAGE.name,
                'pattern': params.get('pattern', 'unknown'),
                'dimensions': f"{width}x{height}x{channels}",
                'original_size': original_size,
                'compressed_size': len(data),
                'compression_ratio': original_size / len(data) if len(data) > 0 else 0,
                'original_hash': 'N/A (procedural)',
                'metadata': params,
                'alice_fingerprint': ALICE_FINGERPRINT
            }

        elif magic == ALICE_AUDIO_MAGIC.decode():
            params = parsed.get('params', {})
            sample_rate = params.get('sample_rate', 0)
            duration = params.get('duration', 0)
            channels = params.get('channels', 1)
            original_size = parsed.get('original_size', int(sample_rate * duration * channels * 2))

            return {
                'magic': magic,
                'version': version_str,
                'file_type': 'AUDIO',
                'engine': 'media_procedural',
                'payload_type': AlicePayloadType.MEDIA_AUDIO.name,
                'pattern': params.get('pattern', 'unknown'),
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': channels,
                'original_size': original_size,
                'compressed_size': len(data),
                'compression_ratio': original_size / len(data) if len(data) > 0 else 0,
                'original_hash': 'N/A (procedural)',
                'metadata': params,
                'alice_fingerprint': ALICE_FINGERPRINT
            }

        elif magic == ALICE_VIDEO_MAGIC.decode():
            width = parsed.get('width', 0)
            height = parsed.get('height', 0)
            fps = parsed.get('fps', 0)
            duration = parsed.get('duration', 0)
            frame_count = int(fps * duration) if fps > 0 else 0
            original_size = width * height * 3 * frame_count

            return {
                'magic': magic,
                'version': version_str,
                'file_type': 'VIDEO',
                'engine': 'media_procedural',
                'payload_type': AlicePayloadType.MEDIA_VIDEO.name,
                'pattern': parsed.get('pattern', 'unknown'),
                'dimensions': f"{width}x{height}",
                'fps': fps,
                'duration': duration,
                'frame_count': frame_count,
                'original_size': original_size,
                'compressed_size': len(data),
                'compression_ratio': original_size / len(data) if len(data) > 0 else 0,
                'original_hash': 'N/A (procedural)',
                'metadata': parsed,
                'alice_fingerprint': ALICE_FINGERPRINT
            }

        else:
            # Unknown JSON format
            return {
                'magic': magic,
                'version': version_str,
                'file_type': 'UNKNOWN_JSON',
                'engine': 'unknown',
                'payload_type': 'unknown',
                'compressed_size': len(data),
                'metadata': parsed,
                'alice_fingerprint': ALICE_FINGERPRINT
            }
