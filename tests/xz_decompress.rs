//! xz streams of each check type decode to their data, and a stream whose check does not match is refused
//!
//! oracle (from the xz format, not read from the code): the same 192 KiB written by liblzma with a CRC-64, a
//! CRC-32 and no check all decode to it; a CRC-64 field with one bit changed is refused although every length
//! and header agrees; a stream with one bit of its CRC-32 field changed is refused too. Fixtures:
//! tests/data/bomb/make_bombs.py

#![cfg(feature = "lzma")]

use std::io;

use alice_zip::compression::{xz_decompress, DecompressError};

const XZ_ZEROS: &[u8] = include_bytes!("data/bomb/xz_zeros_64m.xz");
const CRC32: &[u8] = include_bytes!("data/bomb/xz_varied_crc32.xz");
const NONE: &[u8] = include_bytes!("data/bomb/xz_varied_none.xz");

fn varied() -> Vec<u8> {
    (0..192 * 1024_u64)
        .map(|i| ((i * 2_654_435_761) >> 13) as u8)
        .collect()
}

fn too_large(e: &io::Error) -> bool {
    matches!(
        e.get_ref()
            .and_then(|i| i.downcast_ref::<DecompressError>()),
        Some(DecompressError::TooLarge { .. })
    )
}

#[test]
fn each_check_type_decodes_to_the_data() {
    let data = varied();
    assert_eq!(xz_decompress(CRC32, data.len()).unwrap(), data);
    assert_eq!(xz_decompress(NONE, data.len()).unwrap(), data);
    assert_eq!(
        xz_decompress(XZ_ZEROS, 64 << 20).unwrap(),
        vec![0; 64 << 20]
    );
    // one byte short of what the chunk headers state is refused before decoding
    assert!(too_large(
        &xz_decompress(CRC32, data.len() - 1).unwrap_err()
    ));
}

/// The check field sits just before the index; its offset is found from the footer's backward size
fn flip_check_bit(stream: &[u8], check_len: usize) -> Vec<u8> {
    let mut s = stream.to_vec();
    let backward = u32::from_le_bytes(s[s.len() - 8..s.len() - 4].try_into().unwrap());
    let index_at = s.len() - 12 - (backward as usize + 1) * 4;
    s[index_at - check_len] ^= 1;
    s
}

#[test]
fn a_check_that_does_not_match_is_refused() {
    let e = xz_decompress(&flip_check_bit(CRC32, 4), 1 << 20).unwrap_err();
    assert!(!too_large(&e), "{e}");
    assert!(e.to_string().contains("check"), "{e}");
    let e = xz_decompress(&flip_check_bit(XZ_ZEROS, 8), 64 << 20).unwrap_err();
    assert!(e.to_string().contains("check"), "{e}");
}
