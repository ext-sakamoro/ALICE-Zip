//! Compression for the `.alice` container — thin re-export of the core crate
//!
//! Until 2.3.0 this module carried its own LZMA / zlib wrappers, the 8 / 16-bit
//! quantiser and the residual container codec; a second copy of the zlib
//! wrapper also lived in the core crate. Since 2.4.0 every one of these laws
//! lives once in `alice_zip::compression` (`lzma` feature) and
//! `alice_zip::quantize`, and this module only keeps the names the CLI, the FFI
//! and the Python bindings use. The tests below pin the contract this crate
//! relies on (container layout, edge cases) against the re-exported functions.
//!
//! Binary formats (all little-endian) are documented on
//! `alice_zip::compression`.

pub use alice_zip::compression::{
    compress_residual_lossless, compress_residual_quantized, decompress_residual_lossless,
    decompress_residual_quantized, lzma_compress, lzma_decompress, zlib_compress, zlib_decompress,
};
pub use alice_zip::quantize::{dequantize_16bit, dequantize_8bit, quantize_16bit, quantize_8bit};

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
        let max_err: f32 = data
            .iter()
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
        let data = vec![2.71f32; 100];
        let (quantized, min_val, scale) = quantize_8bit(&data);
        assert_eq!(quantized.len(), 100);
        assert!((min_val - 2.71).abs() < 0.0001);
        // scale should be 1.0 when all values are identical
        assert_eq!(scale, 1.0);

        let restored = dequantize_8bit(&quantized, min_val, scale);
        for v in restored {
            assert!((v - 2.71).abs() < 0.01);
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
