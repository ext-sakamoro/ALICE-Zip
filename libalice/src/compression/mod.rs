//! Compression module for ALICE-Zip
//!
//! Provides high-performance residual compression using LZMA and zlib.
//!
//! # Binary Format Notes
//!
//! All multi-byte numeric values in this module use **little-endian** byte order:
//! - `min_val` (f64): 8 bytes, little-endian
//! - `scale` (f64): 8 bytes, little-endian
//! - `compressed_len` (u32): 4 bytes, little-endian
//! - 16-bit quantized values (u16): 2 bytes each, little-endian
//! - 32-bit float values (f32): 4 bytes each, little-endian
//!
//! This ensures consistent behavior across different CPU architectures.

use std::io::{Read, Cursor};
use flate2::read::{ZlibDecoder, ZlibEncoder};
use flate2::Compression;

/// Compress data using LZMA
///
/// # Arguments
/// * `data` - Raw bytes to compress
/// * `_preset` - Compression level (unused, lzma-rs uses fixed settings)
///
/// # Returns
/// Compressed bytes or IO error
pub fn lzma_compress(data: &[u8], _preset: u32) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_compress(&mut Cursor::new(data), &mut output)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("LZMA compress error: {}", e)))?;
    Ok(output)
}

/// Decompress LZMA data
///
/// # Arguments
/// * `data` - LZMA compressed bytes
///
/// # Returns
/// Decompressed bytes or IO error
pub fn lzma_decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_decompress(&mut Cursor::new(data), &mut output)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("LZMA decompress error: {}", e)))?;
    Ok(output)
}

/// Compress data using zlib
///
/// # Arguments
/// * `data` - Raw bytes to compress
/// * `level` - Compression level (1-9)
///
/// # Returns
/// Compressed bytes or IO error
pub fn zlib_compress(data: &[u8], level: u32) -> std::io::Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(data, Compression::new(level));
    let mut output = Vec::new();
    encoder.read_to_end(&mut output)?;
    Ok(output)
}

/// Decompress zlib data
///
/// # Arguments
/// * `data` - zlib compressed bytes
///
/// # Returns
/// Decompressed bytes or IO error
pub fn zlib_decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output)?;
    Ok(output)
}

/// Quantize f32 array to 8-bit
///
/// # Arguments
/// * `data` - Input f32 array to quantize
///
/// # Returns
/// Tuple of (quantized_bytes, min_val, scale)
///
/// # Edge Cases
/// - **Empty array**: Returns `(Vec::new(), 0.0, 1.0)`
/// - **Constant data**: Uses `scale = 1.0` to avoid division by zero
/// - **NaN/Inf**: May produce undefined quantization results
pub fn quantize_8bit(data: &[f32]) -> (Vec<u8>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let scale = max_val - min_val;

    // If data is constant (scale ≈ 0), use scale = 1.0 to avoid division by zero.
    // This results in all values quantizing to 0, which is correct for constant data.
    let scale = if scale < 1e-10 { 1.0 } else { scale };

    let quantized: Vec<u8> = data.iter()
        .map(|&v| {
            let normalized = ((v as f64 - min_val) / scale).clamp(0.0, 1.0);
            (normalized * 255.0).round() as u8
        })
        .collect();

    (quantized, min_val, scale)
}

/// Dequantize 8-bit to f32 array
pub fn dequantize_8bit(data: &[u8], min_val: f64, scale: f64) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            let normalized = v as f64 / 255.0;
            (normalized * scale + min_val) as f32
        })
        .collect()
}

/// Quantize f32 array to 16-bit
///
/// # Arguments
/// * `data` - Input f32 array to quantize
///
/// # Returns
/// Tuple of (quantized_bytes, min_val, scale)
/// - Bytes are stored in **little-endian** order (2 bytes per value)
///
/// # Edge Cases
/// - **Empty array**: Returns `(Vec::new(), 0.0, 1.0)`
/// - **Constant data**: Uses `scale = 1.0` to avoid division by zero
/// - **NaN/Inf**: May produce undefined quantization results
pub fn quantize_16bit(data: &[f32]) -> (Vec<u8>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let scale = max_val - min_val;

    // If data is constant (scale ≈ 0), use scale = 1.0 to avoid division by zero.
    // This results in all values quantizing to 0, which is correct for constant data.
    let scale = if scale < 1e-10 { 1.0 } else { scale };

    let quantized: Vec<u16> = data.iter()
        .map(|&v| {
            let normalized = ((v as f64 - min_val) / scale).clamp(0.0, 1.0);
            (normalized * 65535.0).round() as u16
        })
        .collect();

    // Convert to bytes (little endian)
    let bytes: Vec<u8> = quantized.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    (bytes, min_val, scale)
}

/// Dequantize 16-bit to f32 array
pub fn dequantize_16bit(data: &[u8], min_val: f64, scale: f64) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let v = u16::from_le_bytes([chunk[0], chunk[1]]);
            let normalized = v as f64 / 65535.0;
            (normalized * scale + min_val) as f32
        })
        .collect()
}

/// Compress residual with quantization + LZMA
///
/// This is the main function for near-lossless compression.
///
/// # Arguments
/// * `residual` - Float residual data
/// * `bits` - Quantization bits (8 or 16)
/// * `lzma_preset` - LZMA preset (0-9)
///
/// # Returns
/// Compressed bytes including header (min_val, scale) or IO error
pub fn compress_residual_quantized(residual: &[f32], bits: u8, lzma_preset: u32) -> std::io::Result<Vec<u8>> {
    let (quantized, min_val, scale) = if bits == 16 {
        quantize_16bit(residual)
    } else {
        let (q, m, s) = quantize_8bit(residual);
        (q, m, s)
    };

    // Compress quantized data
    let compressed = lzma_compress(&quantized, lzma_preset)?;

    // Build output: [bits(1)] [min_val(8)] [scale(8)] [compressed_len(4)] [compressed_data]
    let mut output = Vec::with_capacity(1 + 8 + 8 + 4 + compressed.len());
    output.push(bits);
    output.extend_from_slice(&min_val.to_le_bytes());
    output.extend_from_slice(&scale.to_le_bytes());
    output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);

    Ok(output)
}

/// Decompress quantized residual
///
/// # Arguments
/// * `data` - Compressed bytes from compress_residual_quantized
///
/// # Returns
/// Decompressed f32 residual or IO error
pub fn decompress_residual_quantized(data: &[u8]) -> std::io::Result<Vec<f32>> {
    if data.len() < 21 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Data too short for residual header (need at least 21 bytes)"
        ));
    }

    let bits = data[0];
    // Parse header fields with proper error handling
    // Note: length check above guarantees these slices are valid, but we use
    // map_err for robustness and to avoid panics in all circumstances
    let min_val = f64::from_le_bytes(data[1..9].try_into()
        .map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse min_val from header bytes [1..9]"
        ))?);
    let scale = f64::from_le_bytes(data[9..17].try_into()
        .map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse scale from header bytes [9..17]"
        ))?);
    let compressed_len = u32::from_le_bytes(data[17..21].try_into()
        .map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse compressed_len from header bytes [17..21]"
        ))?) as usize;

    if data.len() < 21 + compressed_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Data truncated: expected {} bytes, got {}", 21 + compressed_len, data.len())
        ));
    }

    let compressed = &data[21..21+compressed_len];
    let quantized = lzma_decompress(compressed)?;

    if bits == 16 {
        Ok(dequantize_16bit(&quantized, min_val, scale))
    } else {
        Ok(dequantize_8bit(&quantized, min_val, scale))
    }
}

/// Compress raw f32 residual (lossless) with LZMA
pub fn compress_residual_lossless(residual: &[f32], lzma_preset: u32) -> std::io::Result<Vec<u8>> {
    // Convert f32 to bytes
    let bytes: Vec<u8> = residual.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();

    // Compress
    let compressed = lzma_compress(&bytes, lzma_preset)?;

    // Build output: [marker(1)=0xFF] [len(4)] [compressed]
    let mut output = Vec::with_capacity(1 + 4 + compressed.len());
    output.push(0xFF); // Marker for lossless mode
    output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);

    Ok(output)
}

/// Decompress lossless residual
pub fn decompress_residual_lossless(data: &[u8]) -> std::io::Result<Vec<f32>> {
    if data.len() < 5 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Data too short for lossless header (need at least 5 bytes)"
        ));
    }
    if data[0] != 0xFF {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid lossless marker: expected 0xFF, got 0x{:02X}", data[0])
        ));
    }

    // Parse compressed length with proper error handling
    let compressed_len = u32::from_le_bytes(data[1..5].try_into()
        .map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse compressed_len from header bytes [1..5]"
        ))?) as usize;

    if data.len() < 5 + compressed_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Data truncated: expected {} bytes, got {}", 5 + compressed_len, data.len())
        ));
    }

    let compressed = &data[5..5+compressed_len];
    let bytes = lzma_decompress(compressed)?;

    // Safety: chunks_exact(4) guarantees each chunk is exactly 4 bytes
    Ok(bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into()
            .expect("chunks_exact(4) guarantees 4 bytes")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_8bit_roundtrip() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let (quantized, min_val, scale) = quantize_8bit(&data);
        let restored = dequantize_8bit(&quantized, min_val, scale);

        // Check approximate equality
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.05, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_quantize_16bit_roundtrip() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let (quantized, min_val, scale) = quantize_16bit(&data);
        let restored = dequantize_16bit(&quantized, min_val, scale);

        // Check approximate equality (should be very close)
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.001, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_compress_residual_quantized() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin() * 100.0).collect();

        let compressed = compress_residual_quantized(&data, 8, 6).unwrap();
        let restored = decompress_residual_quantized(&compressed).unwrap();

        assert_eq!(data.len(), restored.len());

        // Check approximate equality
        let max_err: f32 = data.iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        // Max error should be reasonable for 8-bit quantization
        assert!(max_err < 1.0, "Max error too high: {}", max_err);
    }

    #[test]
    fn test_compress_residual_lossless() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin() * 100.0).collect();

        let compressed = compress_residual_lossless(&data, 6).unwrap();
        let restored = decompress_residual_lossless(&compressed).unwrap();

        assert_eq!(data, restored);
    }

    #[test]
    fn test_zlib_roundtrip() {
        let data = b"Hello, World! This is a test of zlib compression.";
        let compressed = zlib_compress(data, 6).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_invalid_lzma_data() {
        // Test that invalid data returns error instead of panicking
        let invalid_data = b"not valid lzma data";
        let result = lzma_decompress(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_zlib_data() {
        // Test that invalid data returns error instead of panicking
        let invalid_data = b"not valid zlib data";
        let result = zlib_decompress(invalid_data);
        assert!(result.is_err());
    }

    // ===== Boundary Value Tests =====

    #[test]
    fn test_quantize_empty_array() {
        let data: Vec<f32> = Vec::new();
        let (quantized, min_val, scale) = quantize_8bit(&data);
        assert!(quantized.is_empty());
        assert_eq!(min_val, 0.0);
        assert_eq!(scale, 1.0);

        let (quantized16, min_val16, scale16) = quantize_16bit(&data);
        assert!(quantized16.is_empty());
        assert_eq!(min_val16, 0.0);
        assert_eq!(scale16, 1.0);
    }

    #[test]
    fn test_quantize_single_element() {
        let data = vec![42.0f32];
        let (quantized, min_val, scale) = quantize_8bit(&data);
        assert_eq!(quantized.len(), 1);
        assert_eq!(min_val, 42.0);
        // scale should be 1.0 (constant data handling)
        assert_eq!(scale, 1.0);

        let restored = dequantize_8bit(&quantized, min_val, scale);
        assert_eq!(restored.len(), 1);
        // Should restore close to original
        assert!((restored[0] - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_constant_array() {
        // All values are the same (edge case for scale calculation)
        let data = vec![3.14f32; 100];
        let (quantized, min_val, scale) = quantize_8bit(&data);
        assert_eq!(quantized.len(), 100);
        assert!((min_val - 3.14).abs() < 0.0001);
        // scale should be 1.0 when all values are identical
        assert_eq!(scale, 1.0);

        let restored = dequantize_8bit(&quantized, min_val, scale);
        for v in restored {
            assert!((v - 3.14).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_with_nan() {
        // NaN values should be handled gracefully (quantize to some value)
        let data = vec![1.0f32, f32::NAN, 3.0];
        let (quantized, _min_val, _scale) = quantize_8bit(&data);
        // Should not panic, length should match
        assert_eq!(quantized.len(), 3);
    }

    #[test]
    fn test_quantize_with_infinity() {
        // Infinity values - tests extreme range handling
        let data = vec![0.0f32, f32::INFINITY];
        let (quantized, _min_val, _scale) = quantize_8bit(&data);
        assert_eq!(quantized.len(), 2);

        let data_neg = vec![f32::NEG_INFINITY, 0.0f32];
        let (quantized_neg, _min_val_neg, _scale_neg) = quantize_8bit(&data_neg);
        assert_eq!(quantized_neg.len(), 2);
    }

    #[test]
    fn test_compress_residual_empty() {
        let data: Vec<f32> = Vec::new();
        let compressed = compress_residual_quantized(&data, 8, 6).unwrap();
        let restored = decompress_residual_quantized(&compressed).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_compress_residual_single_element() {
        let data = vec![123.456f32];
        let compressed = compress_residual_quantized(&data, 8, 6).unwrap();
        let restored = decompress_residual_quantized(&compressed).unwrap();
        assert_eq!(restored.len(), 1);
        assert!((restored[0] - 123.456).abs() < 1.0);
    }

    #[test]
    fn test_decompress_truncated_header() {
        // Header requires at least 21 bytes
        let short_data = vec![8u8; 20]; // 20 bytes, less than required 21
        let result = decompress_residual_quantized(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_lossless_truncated() {
        // Lossless header requires at least 5 bytes
        let short_data = vec![0xFFu8; 4]; // 4 bytes, less than required 5
        let result = decompress_residual_lossless(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_lossless_invalid_marker() {
        // Invalid marker (not 0xFF)
        let mut data = vec![0x00u8; 10];
        data[0] = 0x00; // Wrong marker
        let result = decompress_residual_lossless(&data);
        assert!(result.is_err());
    }
}
