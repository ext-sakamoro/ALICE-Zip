#!/usr/bin/env python3
"""
ALICE-Zip: Residual Compression (Commercial License)
=====================================================

Implements residual (difference) compression for lossless reconstruction.

The key insight: procedural generation rarely achieves 100% match with real data.
By storing the residual (original - generated), we can achieve true lossless compression.

Formula:
    Data_original = Gen(Params) + Decompress(Residual_compressed)

Author: Moroya Sakamoto
License: ALICE-Zip Commercial License
"""

import json
import logging
import lzma
import struct
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class ResidualCompressionMethod(Enum):
    """Available residual compression methods"""
    NONE = "none"           # No residual (lossy)
    LZMA = "lzma"           # Best ratio, slower
    ZLIB = "zlib"           # Good balance
    ZSTD = "zstd"           # Fast, good ratio (requires zstd)
    DELTA = "delta"         # Delta encoding + compression
    BITDELTA = "bitdelta"   # Differences of f32 bit patterns as wrapping u32 (xz)
    QUANTIZED = "quantized" # Quantize residual before compression


# dtypes the writer records for the original of a residual (the real numeric
# dtypes); anything else is refused when a header is read
_WRITER_DTYPES = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
})


@dataclass
class ResidualData:
    """Encapsulates compressed residual data"""
    method: ResidualCompressionMethod
    compressed_data: bytes
    original_shape: Tuple[int, ...]
    original_dtype: str
    compression_ratio: float
    quantization_bits: Optional[int] = None  # For QUANTIZED method
    # "base_value" of an earlier Rust "delta" file (its first delta is the
    # first value); an earlier Python "delta" file has none and lost it
    base_value: Optional[float] = None
    # the payload is the Rust writer's earlier quantized container (its header
    # records min_val / scale / bits)
    rust_container: bool = False
    # positions (ascending) whose original is kept as it is, and the original
    # elements there (little-endian bytes of the dtype); see compress_original
    exception_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    exception_bytes: bytes = b""
    # the file's layout: 2 (no exceptions, float32 residual), 3 (exceptions as
    # a raw block after the compressed residual) or 4 (residual_dtype, and the
    # exceptions compressed with the residual)
    layout: int = 2
    # "float32" or "float64" (version 4); the residual stream before the
    # method's compression, without the exceptions (version 4, as read)
    residual_dtype: str = "float32"
    residual_stream: Optional[bytes] = None

    # Maximum header size - prevents DoS via malformed header_len
    # 10MB is generous for JSON metadata; larger payloads should use binary format
    MAX_HEADER_SIZE = 10 * 1024 * 1024

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes for storage.

        Format:
        - 4 bytes: header length (little-endian uint32)
        - N bytes: JSON header
        - M bytes: compressed data

        Returns:
            Serialized bytes
        """
        import struct
        import json

        header = {
            'method': self.method.value,
            'shape': list(self.original_shape),  # Ensure list for JSON
            'dtype': self.original_dtype,
            'quant_bits': self.quantization_bits,
            'version': 2  # Version 2 uses 4-byte header length
        }
        k = len(self.exception_positions)
        block = b""
        if self.layout == 4:
            # the residual in the original's precision, the exceptions
            # compressed with it (inside compressed_data)
            header['version'] = 4
            header['residual_dtype'] = self.residual_dtype
            header['exceptions'] = k
        elif k:
            # version 3: a raw block after the compressed residual (positions
            # as u64, then the original elements)
            header['version'] = 3
            header['exceptions'] = k
            block = np.asarray(self.exception_positions, dtype='<u8').tobytes() + \
                self.exception_bytes
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')

        # Use 4-byte integer for header length (supports up to 4GB headers)
        header_len = struct.pack('<I', len(header_json))

        return header_len + header_json + self.compressed_data + block

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ResidualData':
        """
        Deserialize from bytes.

        Supports both v1 (2-byte header length) and v2 (4-byte header length) formats.

        Args:
            data: Serialized bytes

        Returns:
            ResidualData instance

        Raises:
            ValueError: If data is malformed or too large
        """
        import struct
        import json

        if len(data) < 4:
            raise ValueError(f"Data too short: {len(data)} bytes (minimum 4)")

        # Try to detect version by checking if 4-byte interpretation makes sense
        # In v1, header was typically small (< 1000 bytes), so bytes 2-3 would be 0
        # In v2, we use 4 bytes for header length

        # First, try v2 format (4-byte header length)
        header_len_v2 = struct.unpack('<I', data[:4])[0]

        # Security: Validate header length BEFORE any slicing
        # This prevents DoS attacks with malformed header_len values
        data_len = len(data)

        # Check if this looks like v2 format
        if header_len_v2 < cls.MAX_HEADER_SIZE:
            # Validate header_len doesn't exceed available data
            if 4 + header_len_v2 > data_len:
                # This could be v1 format or a corrupted v2 file
                # Fall through to v1 check first
                pass
            else:
                try:
                    # Safe to slice - bounds already validated
                    header_json = data[4:4+header_len_v2].decode('utf-8')
                    # Use standard json.loads - header size already validated
                    header = json.loads(header_json)

                    # v2 format has 'version' field; only versions 1 and 2
                    # were written, so a later one is refused rather than
                    # read as version 2 (same rule as the Rust reader)
                    version = header.get('version', 1)
                    # the writers emit an integer; a float or a string is
                    # refused (the Rust reader takes only a bare integer too)
                    if isinstance(version, bool) or not isinstance(version, int):
                        raise ValueError(
                            f"Residual header version must be a JSON integer, got {version!r}"
                        )
                    if version > 4 or version < 1:
                        raise ValueError(
                            f"Unsupported residual header version {version} (1 to 4 exist)"
                        )
                    if version in (2, 3, 4):
                        compressed_data = data[4+header_len_v2:]
                        return cls._create_from_header(header, compressed_data, version)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass  # Fall through to v1 format
        else:
            # header_len_v2 >= MAX_HEADER_SIZE: definitely not valid v2 format
            # Check if this is an attack (absurdly large value)
            if header_len_v2 > data_len:
                # Could still be v1 format, continue checking
                pass

        # Fall back to v1 format (2-byte header length) for backward compatibility
        if len(data) < 2:
            raise ValueError(f"Data too short for v1 format: {len(data)} bytes")

        header_len_v1 = struct.unpack('<H', data[:2])[0]

        if header_len_v1 > cls.MAX_HEADER_SIZE:
            raise ValueError(f"Header too large: {header_len_v1} bytes (max {cls.MAX_HEADER_SIZE})")

        if 2 + header_len_v1 > len(data):
            raise ValueError(
                f"Header length ({header_len_v1}) exceeds available data "
                f"({len(data) - 2} bytes)"
            )

        try:
            header_json = data[2:2+header_len_v1].decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid header encoding: {e}")

        try:
            # Use standard json.loads - header size already validated via MAX_HEADER_SIZE
            header = json.loads(header_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid header JSON: {e}")

        compressed_data = data[2+header_len_v1:]

        return cls._create_from_header(header, compressed_data)

    @classmethod
    def _create_from_header(cls, header: Dict[str, Any], compressed_data: bytes,
                            version: int = 2) -> 'ResidualData':
        """Create instance from parsed header."""
        # version 3 is a file with exceptions; version 4 records its residual
        # dtype and its number of exceptions (0 or more)
        k = header.get('exceptions')
        if version == 3:
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                raise ValueError(f"version 3 needs a positive integer 'exceptions', got {k!r}")
        elif version == 4:
            if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                raise ValueError(f"version 4 needs an integer 'exceptions' >= 0, got {k!r}")
            if header.get('residual_dtype') not in ('float32', 'float64'):
                raise ValueError(
                    f"version 4 needs residual_dtype float32 or float64, "
                    f"got {header.get('residual_dtype')!r}")
        elif 'exceptions' in header:
            raise ValueError("'exceptions' is only read with versions 3 and 4")
        if version != 4 and 'residual_dtype' in header:
            raise ValueError("'residual_dtype' is only read with version 4")
        # every numeric field is a JSON integer (or a number where the
        # writers write one): a string, a boolean or a float form is refused,
        # as the Rust reader does (bool is an int in Python, so it is named)
        def is_int(v):
            return isinstance(v, int) and not isinstance(v, bool)

        def is_number(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        if 'original_len' in header and not (is_int(header['original_len'])
                                             and header['original_len'] >= 0):
            raise ValueError(f"Invalid original_len: {header['original_len']!r}")
        qb = header.get('quant_bits')
        if qb is not None and not (is_int(qb) and qb in (8, 16, 32)):
            raise ValueError(f"quant_bits must be null, 8, 16 or 32, got {qb!r}")
        if 'base_value' in header and not is_number(header['base_value']):
            raise ValueError(f"base_value must be a number, got {header['base_value']!r}")
        for key in ('min_val', 'scale'):
            if key in header and not is_number(header[key]):
                raise ValueError(f"{key} must be a number, got {header[key]!r}")
        if 'bits' in header and not is_int(header['bits']):
            raise ValueError(f"bits must be an integer, got {header['bits']!r}")

        # the Rust writer before the header keys were aligned recorded only
        # "original_len" (one dimension of float32)
        if 'shape' not in header and 'original_len' in header:
            n = header['original_len']
            header = {**header, 'shape': [n], 'dtype': header.get('dtype', 'float32')}

        if 'shape' in header and 'original_len' in header:
            n = int(np.prod(header['shape'])) if isinstance(header['shape'], (list, tuple)) else None
            if n != header['original_len']:
                raise ValueError(
                    f"shape {header['shape']} and original_len {header['original_len']} disagree"
                )

        # Validate required fields
        required = ['method', 'shape', 'dtype']
        for field in required:
            if field not in header:
                raise ValueError(f"Missing required field: {field}")


        # Validate method
        try:
            method = ResidualCompressionMethod(header['method'])
        except ValueError:
            raise ValueError(f"Unknown compression method: {header['method']}")

        # the writer records the dtype of the original; only the real numeric
        # dtypes it can record are accepted
        if not isinstance(header['dtype'], str) or header['dtype'] not in _WRITER_DTYPES:
            raise ValueError(f"Unsupported dtype {header['dtype']!r} in the residual header")

        # Validate shape
        shape = header['shape']
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"Invalid shape type: {type(shape)}")
        if not all(is_int(d) and d > 0 for d in shape):
            raise ValueError(f"Invalid shape values: {shape}")

        positions = np.zeros(0, dtype=np.uint64)
        exception_bytes = b""
        residual_stream = None
        if version == 4:
            if method not in (ResidualCompressionMethod.NONE, ResidualCompressionMethod.LZMA,
                              ResidualCompressionMethod.ZLIB, ResidualCompressionMethod.BITDELTA) \
                    or header.get('quant_bits') is not None:
                raise ValueError(f"version 4 is lossless: method {method.value} is not read")
            plain = _decompress_plain(method, compressed_data)
            n = int(np.prod(shape))
            rw = 8 if header['residual_dtype'] == 'float64' else 4
            width = np.dtype(header['dtype']).itemsize
            if len(plain) != n * rw + k * (8 + width):
                raise ValueError(
                    f"version 4 payload holds {len(plain)} bytes, expected "
                    f"{n * rw + k * (8 + width)}")
            residual_stream = plain[:n * rw]
            positions = np.frombuffer(plain[n * rw:n * rw + 8 * k], dtype='<u8').astype(np.uint64)
            if k and (np.any(positions[1:] <= positions[:-1]) or int(positions[-1]) >= n):
                raise ValueError(
                    f"exception positions must ascend strictly and stay below {n}")
            exception_bytes = plain[n * rw + 8 * k:]
        elif k:
            if method == ResidualCompressionMethod.QUANTIZED or header.get('quant_bits') is not None:
                raise ValueError("a quantized residual cannot keep exceptions (it is not lossless)")
            width = np.dtype(header['dtype']).itemsize
            need = k * (8 + width)
            if len(compressed_data) < need:
                raise ValueError(f"{k} exceptions need {need} bytes, {len(compressed_data)} follow")
            block = compressed_data[len(compressed_data) - need:]
            compressed_data = compressed_data[:len(compressed_data) - need]
            positions = np.frombuffer(block[:8 * k], dtype='<u8').astype(np.uint64)
            n = int(np.prod(shape))
            if np.any(positions[1:] <= positions[:-1]) or int(positions[-1]) >= n:
                raise ValueError(
                    f"exception positions must ascend strictly and stay below {n}")
            exception_bytes = block[8 * k:]

        return cls(
            method=method,
            compressed_data=compressed_data,
            original_shape=tuple(shape),
            original_dtype=str(header['dtype']),
            compression_ratio=0.0,  # Not stored, recalculated if needed
            quantization_bits=header.get('quant_bits'),
            base_value=header.get('base_value'),
            rust_container=(method == ResidualCompressionMethod.QUANTIZED and 'min_val' in header),
            exception_positions=positions,
            exception_bytes=exception_bytes,
            layout=version if version in (3, 4) else 2,
            residual_dtype=header.get('residual_dtype', 'float32'),
            residual_stream=residual_stream,
        )


class ResidualCompressor:
    """
    Compresses the residual (difference) between original and generated data.

    This enables true lossless compression when combined with procedural generation:
    1. Generate approximation: generated = Gen(params)
    2. Compute residual: residual = original - generated
    3. Compress residual: compressed_residual = Compress(residual)
    4. Store: params + compressed_residual

    Decompression:
    1. Regenerate: generated = Gen(params)
    2. Decompress residual: residual = Decompress(compressed_residual)
    3. Reconstruct: original = generated + residual
    """

    def __init__(
        self,
        method: ResidualCompressionMethod = ResidualCompressionMethod.LZMA,
        lzma_preset: int = 6,
        zlib_level: int = 6,
        quantization_bits: Optional[int] = None
    ):
        """
        Initialize residual compressor.

        Args:
            method: Compression method for residual
            lzma_preset: LZMA compression preset (0-9, higher = better ratio)
            zlib_level: zlib compression level (1-9)
            quantization_bits: If set, quantize residual to this many bits (lossy)
        """
        self.method = method
        self.lzma_preset = lzma_preset
        self.zlib_level = zlib_level
        self.quantization_bits = quantization_bits

    def compute_residual(
        self,
        original: np.ndarray,
        generated: np.ndarray
    ) -> np.ndarray:
        """
        Compute residual between original and generated data.

        Args:
            original: Original data
            generated: Generated approximation

        Returns:
            Residual array (original - generated)
        """
        # Ensure same shape
        if original.shape != generated.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {generated.shape}")

        # Compute difference in float64 for precision
        residual = original.astype(np.float64) - generated.astype(np.float64)

        return residual

    def compress_residual(
        self,
        residual: np.ndarray,
        original_dtype: str = 'float32'
    ) -> ResidualData:
        """
        Compress residual data.

        Args:
            residual: Residual array to compress
            original_dtype: Original data dtype for reconstruction

        Returns:
            ResidualData containing compressed residual
        """
        original_size = residual.nbytes

        # Apply quantization if specified
        if self.quantization_bits is not None or self.method == ResidualCompressionMethod.QUANTIZED:
            bits = self.quantization_bits or 8
            residual_bytes = self._quantize_residual(residual, bits)
            quant_bits = bits
        else:
            # Store as little-endian float32 for cross-platform binary compatibility
            # '<f4' ensures x86 ↔ ARM/MIPS/PowerPC interoperability
            residual_bytes = residual.astype('<f4').tobytes()
            quant_bits = None

        # Compress based on method
        if self.method == ResidualCompressionMethod.NONE:
            compressed = residual_bytes
        elif self.method == ResidualCompressionMethod.LZMA:
            compressed = lzma.compress(residual_bytes, preset=self.lzma_preset)
        elif self.method == ResidualCompressionMethod.ZLIB:
            compressed = zlib.compress(residual_bytes, level=self.zlib_level)
        elif self.method == ResidualCompressionMethod.ZSTD:
            compressed = self._compress_zstd(residual_bytes)
        elif self.method in (ResidualCompressionMethod.DELTA, ResidualCompressionMethod.BITDELTA):
            # always written as bitdelta: integer differences of the f32 bit
            # patterns, so every value comes back bit for bit
            compressed = self._compress_delta(residual)
        elif self.method == ResidualCompressionMethod.QUANTIZED:
            compressed = lzma.compress(residual_bytes, preset=self.lzma_preset)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

        compression_ratio = original_size / len(compressed) if len(compressed) > 0 else 0.0

        logger.debug(
            f"Residual compression: {original_size} -> {len(compressed)} bytes "
            f"({compression_ratio:.2f}x) using {self.method.value}"
        )

        method = self.method
        if method == ResidualCompressionMethod.DELTA:
            method = ResidualCompressionMethod.BITDELTA
        return ResidualData(
            method=method,
            compressed_data=compressed,
            original_shape=residual.shape,
            original_dtype=original_dtype,
            compression_ratio=compression_ratio,
            quantization_bits=quant_bits
        )

    def decompress_residual(self, residual_data: ResidualData) -> np.ndarray:
        """
        Decompress residual data.

        The residual is a float difference (original - generated), returned as
        float32 whatever dtype the header records: that dtype is the
        original's, applied by `reconstruct` (rounded half to even and
        saturated to its range). Casting the residual to an integer dtype here
        would truncate it.

        Args:
            residual_data: Compressed residual data

        Returns:
            Decompressed residual array (float32)
        """
        out = np.asarray(self._decode_residual(residual_data))
        return out.astype(np.dtype(residual_data.residual_dtype), copy=False)

    def _decode_residual(self, residual_data: ResidualData) -> np.ndarray:
        """Decompress residual data (dtype as the method produces it)."""
        if residual_data.layout == 4:
            rdt = np.dtype(residual_data.residual_dtype).newbyteorder('<')
            stream = residual_data.residual_stream
            if residual_data.method == ResidualCompressionMethod.BITDELTA:
                return _bit_delta_decode(stream, rdt).reshape(residual_data.original_shape)
            return np.frombuffer(stream, dtype=rdt).reshape(residual_data.original_shape).copy()
        method = residual_data.method
        compressed = residual_data.compressed_data
        shape = residual_data.original_shape

        # Decompress based on method
        if method == ResidualCompressionMethod.NONE:
            raw_bytes = compressed
        elif method == ResidualCompressionMethod.LZMA:
            raw_bytes = lzma.decompress(compressed)
        elif method == ResidualCompressionMethod.ZLIB:
            raw_bytes = zlib.decompress(compressed)
        elif method == ResidualCompressionMethod.ZSTD:
            raw_bytes = self._decompress_zstd(compressed)
        elif method == ResidualCompressionMethod.BITDELTA:
            return self._decompress_bitdelta(compressed, shape)
        elif method == ResidualCompressionMethod.DELTA:
            # an earlier Rust file recorded base_value and kept the first value
            # as the first delta; an earlier Python file did neither and
            # cannot be reconstructed (decompress_delta_differences returns its
            # differences on request)
            if residual_data.base_value is None:
                raise ValueError(
                    "delta residual without its base value: recompress from the original"
                )
            return self._decompress_delta(compressed, shape)
        elif method == ResidualCompressionMethod.QUANTIZED and residual_data.rust_container:
            return _decode_rust_quantized_container(compressed).reshape(shape)
        elif method == ResidualCompressionMethod.QUANTIZED:
            raw_bytes = lzma.decompress(compressed)
            return self._dequantize_residual(
                raw_bytes, shape, residual_data.quantization_bits or 8
            )
        else:
            raise ValueError(f"Unknown compression method: {method}")

        # Convert back to array
        if residual_data.quantization_bits is not None:
            return self._dequantize_residual(
                raw_bytes, shape, residual_data.quantization_bits
            )
        else:
            # the stored float32 values as they are: a float64 round trip
            # would quiet signaling NaNs (the bits would not be the stored ones)
            return np.frombuffer(raw_bytes, dtype='<f4').reshape(shape).copy()

    def reconstruct(
        self,
        generated: np.ndarray,
        residual_data: ResidualData
    ) -> np.ndarray:
        """
        Reconstruct original data from generated + residual.

        Args:
            generated: Generated approximation
            residual_data: Compressed residual

        Returns:
            Reconstructed original data
        """
        residual = self.decompress_residual(residual_data)

        # Ensure shapes match
        if generated.shape != residual.shape:
            raise ValueError(f"Shape mismatch: {generated.shape} vs {residual.shape}")

        target_dtype = np.dtype(residual_data.original_dtype)
        out, invalid = _rebuild(np.asarray(generated, dtype=np.float64).ravel(),
                                residual.ravel(), target_dtype)
        positions = residual_data.exception_positions
        if len(positions):
            idx = positions.astype(np.intp)
            out[idx] = np.frombuffer(residual_data.exception_bytes,
                                     dtype=target_dtype.newbyteorder('<'))
            invalid[idx] = False
        if invalid.any():
            raise ValueError("NaN where the original is an integer")
        return out.reshape(residual.shape)

    def compress_original(self, original: np.ndarray, generated: np.ndarray) -> ResidualData:
        """
        Residual of `original` against `generated` that rebuilds the original
        bit for bit with `reconstruct`.

        The residual is kept in the original's precision: float64 for
        float64 / int32 / uint32 / int64 / uint64 originals, float32 for the
        others. A position is kept as an exception (the original element
        itself, the residual there 0) when the original or the generated value
        is not finite, or when the reconstruct rule applied to generated and
        the residual does not give the original's bits. A float32 residual
        without exceptions is written as version 2; anything else as version
        4, with the exceptions compressed together with the residual.

        Raises:
            ValueError: for a quantizing compressor (lossy, so it cannot keep
                exceptions), a method that is not lossless, a dtype the writer
                does not record, or shapes that differ
        """
        if self.method == ResidualCompressionMethod.QUANTIZED or self.quantization_bits is not None:
            raise ValueError("a quantized residual is not lossless: use a lossless method")
        method = self.method
        if method == ResidualCompressionMethod.DELTA:
            method = ResidualCompressionMethod.BITDELTA
        if method not in (ResidualCompressionMethod.NONE, ResidualCompressionMethod.LZMA,
                          ResidualCompressionMethod.ZLIB, ResidualCompressionMethod.BITDELTA):
            raise ValueError(f"compress_original writes none, lzma, zlib or bitdelta, not {method.value}")
        original = np.asarray(original)
        dtype = original.dtype
        if dtype.name not in _WRITER_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype.name!r}")
        generated = np.asarray(generated, dtype=np.float64)
        if generated.shape != original.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {generated.shape}")
        rdt = np.dtype(np.float64 if dtype.name in _FLOAT64_RESIDUAL else np.float32)
        o = original.ravel()
        g = generated.ravel()
        with np.errstate(over="ignore", invalid="ignore"):
            # a signaling NaN is quieted here; it is an exception either way
            of = o.astype(np.float64)
            finite = np.isfinite(g) & np.isfinite(of)
            r = np.where(finite, of - np.where(finite, g, 0.0), 0.0).astype(rdt)
        rebuilt, invalid = _rebuild(g, r, dtype)
        uint = np.dtype(f"u{dtype.itemsize}")
        exceptions = ~finite | invalid | (rebuilt.view(uint) != o.view(uint))
        r[exceptions] = 0.0
        positions = np.nonzero(exceptions)[0].astype(np.uint64)
        elements = o[exceptions].astype(dtype.newbyteorder('<')).tobytes()
        if method == ResidualCompressionMethod.BITDELTA:
            stream = _bit_delta_encode(r, rdt)
        else:
            stream = r.astype(rdt.newbyteorder('<')).tobytes()
        plain = stream + positions.astype('<u8').tobytes() + elements
        if method == ResidualCompressionMethod.NONE:
            compressed = plain
        elif method == ResidualCompressionMethod.ZLIB:
            compressed = zlib.compress(plain, level=self.zlib_level)
        else:
            compressed = lzma.compress(plain, preset=self.lzma_preset)
        layout = 2 if rdt == np.float32 and len(positions) == 0 else 4
        return ResidualData(
            method=method,
            compressed_data=compressed,
            original_shape=original.shape,
            original_dtype=dtype.name,
            compression_ratio=original.nbytes / len(compressed) if compressed else 0.0,
            exception_positions=positions if layout == 4 else np.zeros(0, dtype=np.uint64),
            exception_bytes=elements,
            layout=layout,
            residual_dtype=rdt.name,
            residual_stream=stream,
        )

    def _quantize_residual(self, residual: np.ndarray, bits: int) -> bytes:
        """
        Quantize residual to specified bit depth.

        Binary format:
            [min_val: 8 bytes, little-endian double '<d']
            [scale:   8 bytes, little-endian double '<d']
            [quantized_data: N bytes, uint8/uint16/uint32 depending on bits]

        Args:
            residual: Input residual array
            bits: Quantization bit depth (8, 16, or 32)

        Returns:
            Packed bytes: 16-byte header + quantized payload
        """
        # computed in float64 whatever the input dtype: in float32 the top
        # 32-bit code 2^32 - 1 rounds to 2^32, outside uint32
        residual = np.asarray(residual, dtype=np.float64)
        if not np.all(np.isfinite(residual)):
            # the codes span the finite range from the minimum to the maximum
            raise ValueError("values that are not finite cannot be quantized")

        # Normalize to [0, 1]
        min_val = residual.min()
        max_val = residual.max()
        scale = max_val - min_val

        if scale < 1e-10:
            # Constant residual
            scale = 1.0

        normalized = (residual - min_val) / scale

        # Quantize (round half to even, as numpy.round)
        top = float(2 ** bits - 1)
        quantized = np.clip(np.round(normalized * top), 0.0, top)

        # Pack into bytes
        if bits == 8:
            packed = quantized.astype(np.uint8)
        elif bits == 16:
            packed = quantized.astype(np.uint16)
        else:
            packed = quantized.astype(np.uint32)

        # Store min/max for dequantization
        import struct
        header = struct.pack('<dd', min_val, scale)

        return header + packed.tobytes()

    def _dequantize_residual(
        self,
        data: bytes,
        shape: Tuple[int, ...],
        bits: int
    ) -> np.ndarray:
        """Dequantize residual from bytes"""
        import struct

        # Extract header
        min_val, scale = struct.unpack('<dd', data[:16])
        packed_data = data[16:]

        # Unpack
        if bits == 8:
            quantized = np.frombuffer(packed_data, dtype=np.uint8)
        elif bits == 16:
            quantized = np.frombuffer(packed_data, dtype=np.uint16)
        else:
            quantized = np.frombuffer(packed_data, dtype=np.uint32)

        quantized = quantized.reshape(shape)

        # Dequantize
        levels = 2 ** bits
        normalized = quantized.astype(np.float64) / (levels - 1)
        residual = normalized * scale + min_val

        return residual

    def _compress_delta(self, residual: np.ndarray) -> bytes:
        """bitdelta stream of the float32 values, xz-compressed"""
        return lzma.compress(_bit_delta_encode(residual), preset=self.lzma_preset)

    def _decompress_bitdelta(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """float32 values of a bitdelta residual, bit for bit"""
        return _bit_delta_decode(lzma.decompress(data)).reshape(shape)

    def _decompress_delta(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Earlier float-difference delta (as the earlier Rust writer stored it)"""
        raw = lzma.decompress(data)
        # Use little-endian '<f4' for cross-platform binary compatibility
        deltas = np.frombuffer(raw, dtype='<f4')

        # Reconstruct from deltas, accumulating in float32 as that writer did
        flat = np.cumsum(deltas, dtype=np.float32)
        return flat.reshape(shape)

    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress with zstd (if available)"""
        try:
            import zstd
            return zstd.compress(data, 3)
        except ImportError:
            logger.warning("zstd not available, falling back to LZMA")
            return lzma.compress(data, preset=self.lzma_preset)

    def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress zstd data"""
        try:
            import zstd
            return zstd.decompress(data)
        except ImportError:
            # Might be LZMA fallback
            return lzma.decompress(data)


# ============================================================================
# Residual Analysis Utilities
# ============================================================================

def analyze_residual(
    original: np.ndarray,
    generated: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze residual characteristics to determine best compression strategy.

    Args:
        original: Original data
        generated: Generated approximation

    Returns:
        Dict with residual statistics and recommendations
    """
    residual = original.astype(np.float64) - generated.astype(np.float64)

    # Basic statistics
    stats = {
        'shape': residual.shape,
        'min': float(residual.min()),
        'max': float(residual.max()),
        'mean': float(residual.mean()),
        'std': float(residual.std()),
        'mse': float(np.mean(residual ** 2)),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'mae': float(np.mean(np.abs(residual))),
    }

    # Relative error (if original has non-zero values)
    original_range = original.max() - original.min()
    if original_range > 1e-10:
        stats['relative_rmse'] = stats['rmse'] / original_range
        stats['psnr'] = 20 * np.log10(original_range / stats['rmse']) if stats['rmse'] > 0 else float('inf')
    else:
        stats['relative_rmse'] = 0.0
        stats['psnr'] = float('inf')

    # Sparsity (what fraction of residual is near zero)
    threshold = stats['std'] * 0.1
    near_zero = np.sum(np.abs(residual) < threshold)
    stats['sparsity'] = float(near_zero / residual.size)

    # Entropy estimate (compressibility indicator)
    # Use '<f4' for consistency with compress/decompress methods
    residual_bytes = residual.astype('<f4').tobytes()
    compressed_lzma = lzma.compress(residual_bytes, preset=1)  # Fast preset
    stats['raw_size'] = len(residual_bytes)
    stats['estimated_compressed_size'] = len(compressed_lzma)
    stats['estimated_ratio'] = len(residual_bytes) / len(compressed_lzma)

    # Recommendation
    if stats['relative_rmse'] < 0.001:
        stats['recommendation'] = 'excellent_fit'
        stats['recommended_method'] = ResidualCompressionMethod.LZMA
    elif stats['relative_rmse'] < 0.01:
        stats['recommendation'] = 'good_fit'
        stats['recommended_method'] = ResidualCompressionMethod.ZLIB
    elif stats['sparsity'] > 0.5:
        stats['recommendation'] = 'sparse_residual'
        stats['recommended_method'] = ResidualCompressionMethod.DELTA
    else:
        stats['recommendation'] = 'poor_fit'
        stats['recommended_method'] = ResidualCompressionMethod.QUANTIZED

    return stats


def estimate_total_compression(
    original_size: int,
    params_size: int,
    residual_data: ResidualData
) -> Dict[str, float]:
    """
    Estimate total compression including params and residual.

    Args:
        original_size: Original data size in bytes
        params_size: Serialized parameters size in bytes
        residual_data: Compressed residual

    Returns:
        Dict with compression statistics
    """
    residual_size = len(residual_data.compressed_data)
    total_compressed = params_size + residual_size + 50  # 50 bytes overhead estimate

    return {
        'original_size': original_size,
        'params_size': params_size,
        'residual_size': residual_size,
        'total_compressed': total_compressed,
        'total_ratio': original_size / total_compressed if total_compressed > 0 else 0.0,
        'params_fraction': params_size / total_compressed if total_compressed > 0 else 0.0,
        'residual_fraction': residual_size / total_compressed if total_compressed > 0 else 0.0,
    }


def decompress_delta_differences(residual_data: "ResidualData") -> np.ndarray:
    """The stored differences of a delta residual, without reconstructing it."""
    if residual_data.method not in (ResidualCompressionMethod.DELTA, ResidualCompressionMethod.BITDELTA):
        raise ValueError(f"Not a delta residual: {residual_data.method.value}")
    raw = lzma.decompress(residual_data.compressed_data)
    return np.frombuffer(raw, dtype='<f4').reshape(residual_data.original_shape)


_QUIET64 = np.uint64(0x7FF8000000000000)
# originals whose residual is kept in float64 (float32 cannot carry their precision)
_FLOAT64_RESIDUAL = frozenset({"float64", "int32", "uint32", "int64", "uint64"})


def _sum_nan_bits(generated: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """float64 bits of the NaN of generated (float64) + residual (float32).

    Chosen by rule, not left to the hardware (which differs in which NaN it
    keeps, and in the sign of the NaN of inf + -inf): generated's NaN with the
    quiet bit set; else the residual's NaN widened (sign, quiet bit, its 23
    payload bits at the top of the 52); else (inf + -inf) 0x7FF8000000000000."""
    gb = generated.view(np.uint64)
    if residual.dtype == np.float64:
        # a float64 residual's NaN with the quiet bit set
        widened = residual.view(np.uint64) | np.uint64(1 << 51)
    else:
        rb = residual.astype(np.float32, copy=False).view(np.uint32).astype(np.uint64)
        widened = ((rb >> np.uint64(31)) << np.uint64(63)) | _QUIET64 | \
            ((rb & np.uint64(0x7FFFFF)) << np.uint64(29))
    bits = np.where(np.isnan(residual), widened, _QUIET64)
    return np.where(np.isnan(generated), gb | np.uint64(1 << 51), bits)


def _nan_as(bits: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """float64 NaN bits as NaNs of a float dtype: the sign, the quiet bit and
    the top of the payload (what numpy's conversion does on the platforms it
    is measured on, written out so it does not depend on them)."""
    sign = bits >> np.uint64(63)
    if dtype == np.float16:
        h = (sign << np.uint64(15)) | np.uint64(0x7E00) | \
            ((bits >> np.uint64(42)) & np.uint64(0x3FF))
        return h.astype(np.uint16).view(np.float16)
    if dtype == np.float32:
        f = (sign << np.uint64(31)) | np.uint64(0x7FC00000) | \
            ((bits >> np.uint64(29)) & np.uint64(0x3FFFFF))
        return f.astype(np.uint32).view(np.float32)
    return bits.view(np.float64)


def _rebuild(generated: np.ndarray, residual: np.ndarray, dtype: np.dtype):
    """generated (float64) + residual (float32 or float64) as `dtype`, by the reconstruct
    rule; returns the values and a mask of the integer positions that are NaN
    (refused unless an exception covers them)."""
    with np.errstate(invalid="ignore"):
        s = generated + residual.astype(np.float64)
    nan = np.isnan(s)
    if np.issubdtype(dtype, np.integer):
        return _to_integer(np.where(nan, 0.0, s), dtype), nan
    with np.errstate(over="ignore", invalid="ignore"):
        out = s.astype(dtype)
    if nan.any():
        out[nan] = _nan_as(_sum_nan_bits(generated[nan], residual[nan]), dtype)
    return out, np.zeros(s.shape, dtype=bool)


def _to_integer(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """float64 values as `dtype`: NaN is refused, the rest rounded half to even
    and saturated to the dtype's range (infinities too).

    Saturation is decided in the integer domain: the float of a 64-bit maximum
    is 2^63 (2^64), one above it, so clipping to it and casting is outside the
    range and its result depends on the platform."""
    if np.isnan(values).any():
        raise ValueError("NaN where the original is an integer")
    r = np.round(values)
    info = np.iinfo(dtype)
    high = r >= float(info.max)
    low = r <= float(info.min)
    out = np.empty(r.shape, dtype=dtype)
    out[high] = info.max
    out[low] = info.min
    mid = ~(high | low)
    # strictly between the two floats, so inside the range and cast exactly
    out[mid] = r[mid].astype(dtype)
    return out


def _bit_delta_encode(values: np.ndarray, dtype=np.dtype('<f4')) -> bytes:
    """See _bit_delta_encode32 (float32, the default); with a float64 dtype,
    the stream of the float64 bit patterns as wrapping uint64 the same way."""
    if np.dtype(dtype).itemsize == 8:
        bits = np.ascontiguousarray(values.astype('<f8')).view('<u8').ravel()
        out = np.empty_like(bits)
        out[:1] = bits[:1]
        out[1:] = bits[1:] - bits[:-1]  # uint64 arithmetic wraps
        return out.astype('<u8').tobytes()
    return _bit_delta_encode32(values)


def _bit_delta_encode32(values: np.ndarray) -> bytes:
    """The bitdelta stream of float32 values (before compression): the
    difference of consecutive bit patterns as wrapping uint32, little endian,
    the first difference being the first pattern. Integer arithmetic, so every
    value (NaN payloads, subnormals, +-0, infinities) comes back bit for bit.
    Same stream as the Rust writer (tests/data/residual/bitdelta_streams.txt)."""
    bits = np.ascontiguousarray(np.asarray(values, dtype='<f4')).view('<u4').ravel()
    out = np.empty_like(bits)
    out[:1] = bits[:1]
    out[1:] = bits[1:] - bits[:-1]  # uint32 arithmetic wraps
    return out.astype('<u4').tobytes()


def _bit_delta_decode(stream: bytes, dtype=np.dtype('<f4')) -> np.ndarray:
    """float32 (or float64) values from a bitdelta stream (inverts _bit_delta_encode)."""
    if np.dtype(dtype).itemsize == 8:
        d = np.frombuffer(stream, dtype='<u8')
        return np.cumsum(d, dtype=np.uint64).astype('<u8').view('<f8')
    d = np.frombuffer(stream, dtype='<u4')
    return np.cumsum(d, dtype=np.uint32).astype('<u4').view('<f4')


def _decompress_plain(method: 'ResidualCompressionMethod', data: bytes) -> bytes:
    """The payload of a version 4 file before the method's compression."""
    if method == ResidualCompressionMethod.NONE:
        return data
    if method == ResidualCompressionMethod.ZLIB:
        return zlib.decompress(data)
    return lzma.decompress(data)


def _decode_rust_quantized_container(data: bytes) -> np.ndarray:
    """Values of the Rust writer's earlier quantized container, as its reader
    returns them: version 1 `0xFC · codec · bits · min: f64 · scale: f64 ·
    len: u32 · payload` (codec 0 = raw deflate, 1 = LZMA) or version 0
    `bits · min · scale · len · LZMA payload`; codes of 8 or 16 bits (little
    endian), `code / (levels - 1) * scale + min` in float64, then float32."""
    if data[:1] == b"\xfc":
        if len(data) < 23:
            raise ValueError("Rust quantized container too short")
        codec, bits = data[1], data[2]
        min_val, scale = struct.unpack("<dd", data[3:19])
        n = struct.unpack("<I", data[19:23])[0]
        payload = data[23:23 + n]
        if codec == 0:
            codes = zlib.decompress(payload, -15)
        elif codec == 1:
            codes = lzma.decompress(payload)
        else:
            raise ValueError(f"Unknown codec tag {codec} in a Rust quantized container")
    else:
        if len(data) < 21:
            raise ValueError("Rust quantized container too short")
        bits = data[0]
        min_val, scale = struct.unpack("<dd", data[1:17])
        n = struct.unpack("<I", data[17:21])[0]
        payload = data[21:21 + n]
        codes = lzma.decompress(payload)
    if len(payload) != n:
        raise ValueError("Rust quantized container truncated")
    if bits == 8:
        q = np.frombuffer(codes, dtype=np.uint8).astype(np.float64) / 255.0
    elif bits == 16:
        q = np.frombuffer(codes, dtype="<u2").astype(np.float64) / 65535.0
    else:
        raise ValueError(f"Rust quantized container with {bits} bits")
    return (q * scale + min_val).astype(np.float32)

