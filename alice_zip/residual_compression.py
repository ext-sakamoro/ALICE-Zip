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
from dataclasses import dataclass
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
    QUANTIZED = "quantized" # Quantize residual before compression


@dataclass
class ResidualData:
    """Encapsulates compressed residual data"""
    method: ResidualCompressionMethod
    compressed_data: bytes
    original_shape: Tuple[int, ...]
    original_dtype: str
    compression_ratio: float
    quantization_bits: Optional[int] = None  # For QUANTIZED method

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
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')

        # Use 4-byte integer for header length (supports up to 4GB headers)
        header_len = struct.pack('<I', len(header_json))

        return header_len + header_json + self.compressed_data

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

                    # v2 format has 'version' field
                    if header.get('version', 1) >= 2:
                        compressed_data = data[4+header_len_v2:]
                        return cls._create_from_header(header, compressed_data)
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
    def _create_from_header(cls, header: Dict[str, Any], compressed_data: bytes) -> 'ResidualData':
        """Create instance from parsed header."""
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

        # Validate shape
        shape = header['shape']
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"Invalid shape type: {type(shape)}")
        if not all(isinstance(d, int) and d > 0 for d in shape):
            raise ValueError(f"Invalid shape values: {shape}")

        return cls(
            method=method,
            compressed_data=compressed_data,
            original_shape=tuple(shape),
            original_dtype=str(header['dtype']),
            compression_ratio=0.0,  # Not stored, recalculated if needed
            quantization_bits=header.get('quant_bits')
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
        elif self.method == ResidualCompressionMethod.DELTA:
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

        return ResidualData(
            method=self.method,
            compressed_data=compressed,
            original_shape=residual.shape,
            original_dtype=original_dtype,
            compression_ratio=compression_ratio,
            quantization_bits=quant_bits
        )

    def decompress_residual(self, residual_data: ResidualData) -> np.ndarray:
        """
        Decompress residual data.

        Args:
            residual_data: Compressed residual data

        Returns:
            Decompressed residual array
        """
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
        elif method == ResidualCompressionMethod.DELTA:
            return self._decompress_delta(compressed, shape)
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
            # Use little-endian '<f4' for cross-platform binary compatibility
            residual = np.frombuffer(raw_bytes, dtype='<f4').reshape(shape)
            return residual.astype(np.float64)

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

        # Reconstruct
        reconstructed = generated.astype(np.float64) + residual

        # Cast back to original dtype
        target_dtype = np.dtype(residual_data.original_dtype)

        if np.issubdtype(target_dtype, np.integer):
            # Clip and round for integer types
            info = np.iinfo(target_dtype)
            reconstructed = np.clip(reconstructed, info.min, info.max)
            reconstructed = np.round(reconstructed)

        return reconstructed.astype(target_dtype)

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
        # Normalize to [0, 1]
        min_val = residual.min()
        max_val = residual.max()
        scale = max_val - min_val

        if scale < 1e-10:
            # Constant residual
            scale = 1.0

        normalized = (residual - min_val) / scale

        # Quantize
        levels = 2 ** bits
        quantized = np.clip(np.round(normalized * (levels - 1)), 0, levels - 1)

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
        """Delta encoding + LZMA compression"""
        # Flatten and compute deltas (use little-endian for cross-platform compatibility)
        flat = residual.astype('<f4').flatten()
        deltas = np.diff(flat, prepend=flat[0])

        # LZMA compress deltas
        return lzma.compress(deltas.tobytes(), preset=self.lzma_preset)

    def _decompress_delta(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress delta-encoded residual"""
        raw = lzma.decompress(data)
        # Use little-endian '<f4' for cross-platform binary compatibility
        deltas = np.frombuffer(raw, dtype='<f4')

        # Reconstruct from deltas
        flat = np.cumsum(deltas)
        return flat.reshape(shape).astype(np.float64)

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
