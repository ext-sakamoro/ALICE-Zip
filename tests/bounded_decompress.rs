//! Decompression is bounded: a small input that expands without limit (a decompression bomb) is refused at the
//! caller's output limit, with a typed error, instead of being expanded into memory.
//!
//! oracle (from the rules, not read from the code under test): output of exactly the limit decodes and equals the
//! original; one byte (one sample) more than the limit is refused with `DecompressError::TooLarge { limit }`; a
//! small input that expands to far more than the limit is refused the same way; an input that is not a valid stream
//! is an error that is not `TooLarge`; the xor residual is bounded by its model (a payload larger than the model is
//! refused as too large, before any count comparison).

#![cfg(feature = "std")]

use std::io;

use alice_zip::compression::{zlib_compress, zlib_decompress, DecompressError};

fn too_large(e: &io::Error) -> Option<usize> {
    match e
        .get_ref()
        .and_then(|i| i.downcast_ref::<DecompressError>())
    {
        Some(DecompressError::TooLarge { limit }) => Some(*limit),
        _ => None,
    }
}

/// Bytes that do not compress to nothing (so the exact-limit cases decode a real stream).
fn varied(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 3)) as u8)
        .collect()
}

#[test]
fn zlib_output_of_exactly_the_limit_decodes_and_one_byte_more_is_refused() {
    let data = varied(10_000);
    let c = zlib_compress(&data, 6).unwrap();
    assert_eq!(zlib_decompress(&c, data.len()).unwrap(), data);
    let e = zlib_decompress(&c, data.len() - 1).unwrap_err();
    assert_eq!(too_large(&e), Some(data.len() - 1), "{e}");
}

#[test]
fn a_zlib_bomb_is_refused_at_the_limit() {
    // 64 MiB of zeros compresses to about 64 KiB: a small input that expands a thousandfold
    let bomb = zlib_compress(&vec![0u8; 64 << 20], 9).unwrap();
    assert!(
        bomb.len() < 1 << 20,
        "the bomb is small: {} bytes",
        bomb.len()
    );
    let e = zlib_decompress(&bomb, 1 << 20).unwrap_err();
    assert_eq!(too_large(&e), Some(1 << 20), "{e}");
}

#[test]
fn an_invalid_zlib_stream_is_an_error_that_is_not_too_large() {
    let e = zlib_decompress(b"not a zlib stream at all", 1 << 20).unwrap_err();
    assert_eq!(too_large(&e), None, "{e}");
}

#[cfg(feature = "lzma")]
mod lzma {
    use super::{too_large, varied};
    use alice_zip::compression::{
        compress_residual_lossless, compress_residual_quantized, compress_residual_xor,
        decompress_residual_lossless, decompress_residual_quantized, decompress_residual_xor,
        lzma_compress, lzma_decompress,
    };

    #[test]
    fn lzma_output_of_exactly_the_limit_decodes_and_one_byte_more_is_refused() {
        let data = varied(10_000);
        let c = lzma_compress(&data, 6).unwrap();
        assert_eq!(lzma_decompress(&c, data.len()).unwrap(), data);
        let e = lzma_decompress(&c, data.len() - 1).unwrap_err();
        assert_eq!(too_large(&e), Some(data.len() - 1), "{e}");
    }

    #[test]
    fn an_lzma_bomb_is_refused_at_the_limit() {
        let bomb = lzma_compress(&vec![0u8; 32 << 20], 6).unwrap();
        assert!(
            bomb.len() < 1 << 20,
            "the bomb is small: {} bytes",
            bomb.len()
        );
        let e = lzma_decompress(&bomb, 1 << 20).unwrap_err();
        assert_eq!(too_large(&e), Some(1 << 20), "{e}");
    }

    #[test]
    fn an_invalid_lzma_stream_is_an_error_that_is_not_too_large() {
        let e = lzma_decompress(&[0x5d, 0, 0, 0x80, 0, 1, 2, 3], 1 << 20).unwrap_err();
        assert_eq!(too_large(&e), None, "{e}");
    }

    fn samples(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| alice_det_math::sin(i as f32 * 0.37))
            .collect()
    }

    #[test]
    fn the_lossless_residual_takes_exactly_its_sample_limit_and_refuses_one_less() {
        let s = samples(1_000);
        let c = compress_residual_lossless(&s, 6).unwrap();
        let back = decompress_residual_lossless(&c, s.len()).unwrap();
        assert_eq!(
            back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            s.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let e = decompress_residual_lossless(&c, s.len() - 1).unwrap_err();
        assert_eq!(
            too_large(&e),
            Some((s.len() - 1) * 4),
            "the limit in bytes: {e}"
        );
    }

    #[test]
    fn the_quantized_residual_takes_exactly_its_sample_limit_and_refuses_one_less() {
        let s = samples(1_000);
        let c = compress_residual_quantized(&s, 16, 6).unwrap();
        assert_eq!(
            decompress_residual_quantized(&c, s.len()).unwrap().len(),
            s.len()
        );
        let e = decompress_residual_quantized(&c, s.len() - 1).unwrap_err();
        assert!(too_large(&e).is_some(), "{e}");
    }

    #[test]
    fn a_lossless_residual_bomb_is_refused_at_the_sample_limit() {
        let bomb = compress_residual_lossless(&vec![0.0f32; 8 << 20], 9).unwrap();
        assert!(bomb.len() < 1 << 20);
        let e = decompress_residual_lossless(&bomb, 1_000).unwrap_err();
        assert_eq!(too_large(&e), Some(4_000), "{e}");
    }

    #[test]
    fn the_xor_residual_is_bounded_by_its_model() {
        // a container of 8 Mi samples against a model of 1,000: refused as too large, before any count comparison
        let original = vec![0.0f32; 8 << 20];
        let big_model = vec![0.0f32; 8 << 20];
        let c = compress_residual_xor(&original, &big_model, 9).unwrap();
        let small_model = vec![0.0f32; 1_000];
        let e = decompress_residual_xor(&c, &small_model).unwrap_err();
        assert_eq!(too_large(&e), Some(4_000), "{e}");
        // the matching model decodes
        let ok = samples(500);
        let m = samples(500);
        let c = compress_residual_xor(&ok, &m, 6).unwrap();
        assert_eq!(decompress_residual_xor(&c, &m).unwrap().len(), 500);
    }
}
