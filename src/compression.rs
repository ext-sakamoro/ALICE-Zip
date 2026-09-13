//! zlib compression / decompression wrappers.
//!
//! Thin adapters over [`flate2`](https://crates.io/crates/flate2) that expose
//! a stable API for downstream ALICE-* crates (notably `alice-db`) to compress
//! small binary payloads without pulling in `flate2` directly.
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
}
