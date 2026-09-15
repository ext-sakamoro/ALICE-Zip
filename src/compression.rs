//! Byte-stream compression wrappers and the `.alice` residual container.
//!
//! Thin adapters over [`flate2`](https://crates.io/crates/flate2) (zlib, always
//! with `std`) and, behind the `lzma` feature, over
//! [`lzma-rs`](https://crates.io/crates/lzma-rs) (LZMA) plus the quantised /
//! lossless residual containers that the `.alice` file format and
//! `alice-edge` coefficient batches persist. Every function here has a single
//! implementation for the whole ecosystem (`libalice` re-exports this module).
//!
//! # Residual container formats (`lzma` feature)
//!
//! All multi-byte fields are **little-endian**.
//!
//! | container | layout |
//! |-----------|--------|
//! | quantised (`compress_residual_quantized`) | `bits: u8` (8 or 16) · `min: f64` · `scale: f64` · `len: u32` · LZMA(quantised bytes) |
//! | lossless (`compress_residual_lossless`) | `0xFF` · `len: u32` · LZMA(`f32` LE samples) |
//!
//! # Example
//!
//! ```
//! use alice_zip::compression::{zlib_compress, zlib_decompress};
//!
//! let payload = b"the quick brown fox jumps over the lazy dog".repeat(4);
//! let compressed = zlib_compress(&payload, 6).unwrap();
//! let decompressed = zlib_decompress(&compressed).unwrap();
//! assert_eq!(payload, decompressed);
//! ```

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};

#[cfg(feature = "lzma")]
pub use crate::quantize::{dequantize_16bit, dequantize_8bit, quantize_16bit, quantize_8bit};

/// Compress `data` with zlib (deflate + zlib wrapper) at the given level.
///
/// `level` follows the standard zlib range `0..=9` (0 = store, 9 = max
/// compression, 6 = default). Values outside the range are clamped.
///
/// # Errors
///
/// Returns [`io::Error`] on internal encoder failure (usually memory).
pub fn zlib_compress(data: &[u8], level: u32) -> io::Result<Vec<u8>> {
    let level = level.min(9);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data)?;
    encoder.finish()
}

/// Decompress zlib-formatted `data` produced by [`zlib_compress`] (or any
/// zlib-compliant encoder).
///
/// # Errors
///
/// Returns [`io::Error`] if the input is not valid zlib format or the payload
/// is truncated.
pub fn zlib_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------- lzma

/// Compress `data` with LZMA (`.lzma` legacy stream, `lzma-rs` fixed settings;
/// `_preset` is accepted for API compatibility and ignored)
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
#[cfg(feature = "lzma")]
pub fn lzma_compress(data: &[u8], _preset: u32) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_compress(&mut io::Cursor::new(data), &mut output)
        .map_err(|e| io::Error::other(format!("LZMA compress error: {e}")))?;
    Ok(output)
}

/// Decompress an LZMA stream produced by [`lzma_compress`]
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when `data` is not a valid LZMA stream
#[cfg(feature = "lzma")]
pub fn lzma_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_decompress(&mut io::Cursor::new(data), &mut output).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("LZMA decompress error: {e}"),
        )
    })?;
    Ok(output)
}

/// Header size of the quantised residual container
#[cfg(feature = "lzma")]
const QUANTIZED_HEADER: usize = 1 + 8 + 8 + 4;
/// Header size of the lossless residual container
#[cfg(feature = "lzma")]
const LOSSLESS_HEADER: usize = 1 + 4;
/// First byte of the lossless container
#[cfg(feature = "lzma")]
const LOSSLESS_MARKER: u8 = 0xFF;

#[cfg(feature = "lzma")]
fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Quantise `residual` to `bits` (16, or 8 for any other value) and LZMA
/// compress it into the quantised residual container
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
#[cfg(feature = "lzma")]
#[allow(clippy::cast_possible_truncation)]
pub fn compress_residual_quantized(
    residual: &[f32],
    bits: u8,
    lzma_preset: u32,
) -> io::Result<Vec<u8>> {
    let (quantized, min_val, scale) = if bits == 16 {
        quantize_16bit(residual)
    } else {
        quantize_8bit(residual)
    };
    let compressed = lzma_compress(&quantized, lzma_preset)?;
    let mut output = Vec::with_capacity(QUANTIZED_HEADER + compressed.len());
    output.push(bits);
    output.extend_from_slice(&min_val.to_le_bytes());
    output.extend_from_slice(&scale.to_le_bytes());
    output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);
    Ok(output)
}

/// Inverse of [`compress_residual_quantized`]
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the header is truncated, the declared
/// payload length exceeds `data`, or the LZMA stream is invalid
#[cfg(feature = "lzma")]
pub fn decompress_residual_quantized(data: &[u8]) -> io::Result<Vec<f32>> {
    if data.len() < QUANTIZED_HEADER {
        return Err(invalid(format!(
            "Data too short for residual header (need at least {QUANTIZED_HEADER} bytes)"
        )));
    }
    let bits = data[0];
    let min_val = f64::from_le_bytes(data[1..9].try_into().map_err(|_| invalid("min_val"))?);
    let scale = f64::from_le_bytes(data[9..17].try_into().map_err(|_| invalid("scale"))?);
    let compressed_len = u32::from_le_bytes(
        data[17..21]
            .try_into()
            .map_err(|_| invalid("compressed_len"))?,
    ) as usize;
    let end = QUANTIZED_HEADER
        .checked_add(compressed_len)
        .ok_or_else(|| invalid("compressed_len overflow"))?;
    if data.len() < end {
        return Err(invalid(format!(
            "Data truncated: expected {end} bytes, got {}",
            data.len()
        )));
    }
    let quantized = lzma_decompress(&data[QUANTIZED_HEADER..end])?;
    Ok(if bits == 16 {
        dequantize_16bit(&quantized, min_val, scale)
    } else {
        dequantize_8bit(&quantized, min_val, scale)
    })
}

/// LZMA compress the raw little-endian `f32` samples (lossless container)
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
#[cfg(feature = "lzma")]
#[allow(clippy::cast_possible_truncation)]
pub fn compress_residual_lossless(residual: &[f32], lzma_preset: u32) -> io::Result<Vec<u8>> {
    let bytes: Vec<u8> = residual.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let compressed = lzma_compress(&bytes, lzma_preset)?;
    let mut output = Vec::with_capacity(LOSSLESS_HEADER + compressed.len());
    output.push(LOSSLESS_MARKER);
    output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    output.extend_from_slice(&compressed);
    Ok(output)
}

/// Inverse of [`compress_residual_lossless`]
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the marker is wrong, the header or
/// payload is truncated, or the LZMA stream is invalid
#[cfg(feature = "lzma")]
pub fn decompress_residual_lossless(data: &[u8]) -> io::Result<Vec<f32>> {
    if data.len() < LOSSLESS_HEADER {
        return Err(invalid(format!(
            "Data too short for lossless header (need at least {LOSSLESS_HEADER} bytes)"
        )));
    }
    if data[0] != LOSSLESS_MARKER {
        return Err(invalid(format!(
            "Invalid lossless marker: expected 0x{LOSSLESS_MARKER:02X}, got 0x{:02X}",
            data[0]
        )));
    }
    let compressed_len = u32::from_le_bytes(
        data[1..5]
            .try_into()
            .map_err(|_| invalid("compressed_len"))?,
    ) as usize;
    let end = LOSSLESS_HEADER
        .checked_add(compressed_len)
        .ok_or_else(|| invalid("compressed_len overflow"))?;
    if data.len() < end {
        return Err(invalid(format!(
            "Data truncated: expected {end} bytes, got {}",
            data.len()
        )));
    }
    let bytes = lzma_decompress(&data[LOSSLESS_HEADER..end])?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(8);
        let compressed = zlib_compress(&payload, 6).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(payload, decompressed);
        assert!(
            compressed.len() < payload.len(),
            "compression should shrink"
        );
    }

    #[test]
    fn roundtrip_empty() {
        let compressed = zlib_compress(&[], 6).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn roundtrip_binary() {
        let payload: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let compressed = zlib_compress(&payload, 9).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(payload, decompressed);
    }

    #[test]
    fn level_clamping() {
        let payload = b"hello world";
        let over_max = zlib_compress(payload, 42).unwrap();
        let normal = zlib_compress(payload, 9).unwrap();
        assert_eq!(over_max, normal, "level > 9 should clamp to 9");
    }

    #[test]
    fn level_zero_stores() {
        let payload = b"abcdef".repeat(16);
        let stored = zlib_compress(&payload, 0).unwrap();
        let decompressed = zlib_decompress(&stored).unwrap();
        assert_eq!(payload, decompressed);
    }

    #[test]
    fn decompress_invalid_returns_error() {
        let garbage = vec![0xff, 0xfe, 0xfd, 0xfc];
        assert!(zlib_decompress(&garbage).is_err());
    }

    #[cfg(feature = "lzma")]
    mod lzma {
        use super::super::*;

        #[test]
        fn lzma_roundtrip_and_invalid() {
            let payload = b"alice alice alice alice".repeat(32);
            let c = lzma_compress(&payload, 6).unwrap();
            assert!(c.len() < payload.len());
            assert_eq!(lzma_decompress(&c).unwrap(), payload);
            assert!(lzma_decompress(b"not lzma").is_err());
            assert_eq!(
                lzma_decompress(&lzma_compress(&[], 6).unwrap()).unwrap(),
                Vec::<u8>::new()
            );
        }

        #[test]
        fn quantized_container_roundtrip_both_widths() {
            let residual: Vec<f32> = (0..300).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
            for bits in [8u8, 16, 0, 255] {
                let c = compress_residual_quantized(&residual, bits, 6).unwrap();
                assert_eq!(c[0], bits);
                let out = decompress_residual_quantized(&c).unwrap();
                assert_eq!(out.len(), residual.len());
                let tol = if bits == 16 {
                    10.0 / 65535.0
                } else {
                    10.0 / 255.0
                };
                for (a, b) in out.iter().zip(&residual) {
                    assert!((a - b).abs() <= tol / 2.0 + 1e-5, "bits={bits}: {a} vs {b}");
                }
            }
            assert!(decompress_residual_quantized(&[16; 20]).is_err());
            let mut truncated = compress_residual_quantized(&residual, 8, 6).unwrap();
            truncated.truncate(30);
            assert!(decompress_residual_quantized(&truncated).is_err());
        }

        #[test]
        fn lossless_container_is_bit_exact() {
            let residual: Vec<f32> = vec![0.0, -0.0, 1.5e-30, f32::MAX, -7.25, 3.0e10];
            let c = compress_residual_lossless(&residual, 6).unwrap();
            assert_eq!(c[0], LOSSLESS_MARKER);
            let out = decompress_residual_lossless(&c).unwrap();
            assert_eq!(out.len(), residual.len());
            for (a, b) in out.iter().zip(&residual) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert!(decompress_residual_lossless(&[0x00, 0, 0, 0, 0]).is_err());
            assert!(decompress_residual_lossless(&[0xFF, 9, 0, 0, 0, 1]).is_err());
        }
    }
}
