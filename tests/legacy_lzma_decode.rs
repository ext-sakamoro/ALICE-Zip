//! `container::decompress_legacy_alice_zip` against files written by the
//! Python writer (`ALICEZip.compress`, `tests/data/container/make_legacy_lzma.py`).
//!
//! oracle: the `.orig` file next to each fixture holds the raw bytes of the
//! array the Python writer compressed; nothing here derives an expected value
//! from the reader itself.

#![cfg(feature = "lzma")]

use alice_zip::container::{decompress_legacy_alice_zip, ContainerError};

const U8: &[u8] = include_bytes!("data/container/legacy_lzma_u8.alice");
const U8_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_u8.orig");
const F64: &[u8] = include_bytes!("data/container/legacy_lzma_f64.alice");
const F64_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_f64.orig");
const SCALAR: &[u8] = include_bytes!("data/container/legacy_lzma_scalar.alice");
const SCALAR_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_scalar.orig");
const EMPTY: &[u8] = include_bytes!("data/container/legacy_lzma_empty.alice");
/// A version 1.1 header with `payload_type` 0x30 and a payload that is not
/// an LZMA fallback payload (the container fixture's placeholder bytes).
const V2: &[u8] = include_bytes!("data/container/legacy_v2.alice");
/// A version 1.0 file, which has no `payload_type`.
const V1: &[u8] = include_bytes!("data/container/legacy_v1.alice");

/// Header layout of version 1.1 (`<9sBBBBB Q Q 32s I`, 66 bytes).
const ORIGINAL_SIZE_AT: usize = 14;
const COMPRESSED_SIZE_AT: usize = 22;
const HASH_AT: usize = 30;
const PAYLOAD_AT: usize = 66;

fn with_original_size(file: &[u8], size: u64) -> Vec<u8> {
    let mut v = file.to_vec();
    v[ORIGINAL_SIZE_AT..ORIGINAL_SIZE_AT + 8].copy_from_slice(&size.to_le_bytes());
    v
}

/// Offset of the xz stream: payload = `meta_len` (u32 LE) ‖ meta JSON ‖ xz.
fn xz_at(file: &[u8]) -> usize {
    let meta_len = u32::from_le_bytes(file[PAYLOAD_AT..PAYLOAD_AT + 4].try_into().unwrap());
    PAYLOAD_AT + 4 + meta_len as usize
}

/// Replaces `from` by `to` inside the meta JSON, rewriting `meta_len` and
/// the header's `compressed_size` so that only the JSON differs.
fn with_meta(file: &[u8], from: &str, to: &str) -> Vec<u8> {
    let xz = xz_at(file);
    let text = std::str::from_utf8(&file[PAYLOAD_AT + 4..xz]).unwrap();
    assert!(text.contains(from), "{text}");
    let meta = text.replace(from, to);
    let mut v = file[..PAYLOAD_AT].to_vec();
    v.extend_from_slice(&u32::try_from(meta.len()).unwrap().to_le_bytes());
    v.extend_from_slice(meta.as_bytes());
    v.extend_from_slice(&file[xz..]);
    let payload = (v.len() - PAYLOAD_AT) as u64;
    v[COMPRESSED_SIZE_AT..COMPRESSED_SIZE_AT + 8].copy_from_slice(&payload.to_le_bytes());
    v
}

#[test]
fn rewriting_the_meta_unchanged_still_decodes() {
    // The helper itself: an identity edit keeps a valid file.
    let v = with_meta(F64, "float64", "float64");
    assert_eq!(v, F64);
    assert_eq!(decompress_legacy_alice_zip(&v).unwrap().data, F64_ORIG);
}

#[test]
fn python_lzma_fallback_files_decode_to_the_original_bytes() {
    let a = decompress_legacy_alice_zip(U8).unwrap();
    assert_eq!(a.data, U8_ORIG);
    assert_eq!(a.shape, vec![4096]);
    assert_eq!(a.dtype, "uint8");

    let a = decompress_legacy_alice_zip(F64).unwrap();
    assert_eq!(a.data, F64_ORIG);
    assert_eq!(a.shape, vec![16, 8]);
    assert_eq!(a.dtype, "float64");
}

#[test]
fn zero_dimensional_and_empty_arrays_decode() {
    let a = decompress_legacy_alice_zip(SCALAR).unwrap();
    assert_eq!(a.data, SCALAR_ORIG);
    assert_eq!(a.data, 3.5_f64.to_le_bytes());
    assert!(a.shape.is_empty());
    assert_eq!(a.dtype, "float64");

    let a = decompress_legacy_alice_zip(EMPTY).unwrap();
    assert!(a.data.is_empty());
    assert_eq!(a.shape, vec![0]);
    assert_eq!(a.dtype, "int32");
}

#[test]
fn a_changed_stored_hash_is_refused() {
    for file in [U8, F64] {
        let mut v = file.to_vec();
        v[HASH_AT + 5] ^= 0x01;
        assert_eq!(
            decompress_legacy_alice_zip(&v),
            Err(ContainerError::OriginalHash)
        );
    }
}

#[test]
fn an_all_zero_hash_means_no_hash_recorded() {
    let mut v = F64.to_vec();
    v[HASH_AT..HASH_AT + 32].fill(0);
    assert_eq!(decompress_legacy_alice_zip(&v).unwrap().data, F64_ORIG);
}

#[test]
fn a_stated_size_larger_than_the_result_is_refused() {
    assert_eq!(
        decompress_legacy_alice_zip(&with_original_size(F64, 1025)),
        Err(ContainerError::OriginalSize {
            stated: 1025,
            actual: 1024
        })
    );
}

#[test]
fn a_result_longer_than_the_stated_size_is_refused_without_finishing() {
    assert_eq!(
        decompress_legacy_alice_zip(&with_original_size(F64, 1023)),
        Err(ContainerError::LegacyPayload)
    );
    assert_eq!(
        decompress_legacy_alice_zip(&with_original_size(U8, 0)),
        Err(ContainerError::LegacyPayload)
    );
}

#[test]
fn payloads_that_are_not_the_lzma_fallback_are_not_decoded_here() {
    // payload_type is the byte after engine (offset 13).
    for t in [0x00, 0x10, 0x11, 0x12, 0x20] {
        let mut v = V2.to_vec();
        v[13] = t;
        assert_eq!(
            decompress_legacy_alice_zip(&v),
            Err(ContainerError::UnsupportedPayload {
                payload_type: Some(t)
            }),
            "{t:#04x}"
        );
    }
    assert_eq!(
        decompress_legacy_alice_zip(V2),
        Err(ContainerError::LegacyPayload)
    );
    assert_eq!(
        decompress_legacy_alice_zip(V1),
        Err(ContainerError::UnsupportedPayload { payload_type: None })
    );
}

#[test]
fn shape_and_dtype_must_account_for_every_byte() {
    // Same JSON length, half the bytes.
    let v = with_meta(F64, "float64", "float32");
    assert_eq!(
        decompress_legacy_alice_zip(&v),
        Err(ContainerError::LegacyPayload)
    );
    let v = with_meta(F64, "[16,8]", "[16,4]");
    assert_eq!(
        decompress_legacy_alice_zip(&v),
        Err(ContainerError::LegacyPayload)
    );
    // A dtype name the writer never produced.
    let v = with_meta(U8, "\"uint8\"", "\"uint9\"");
    assert_eq!(
        decompress_legacy_alice_zip(&v),
        Err(ContainerError::LegacyPayload)
    );
}

#[test]
fn meta_that_is_not_the_writer_form_is_refused() {
    for (from, to) in [
        ("{\"shape\"", "{\"Shape\""),
        ("[16,8]", "[16 8]"),
        ("[16,8]", "[16,08"),
        ("\"dtype\"", "\"dtypf\""),
        ("4\"}", "4\" "),
        ("[16,8]", "[016,8]"),
        ("[16,8]", "[16, 8]"),
        ("[16,8]", "[16,8,]"),
        ("{", " {"),
        ("\"}", "\"} "),
        ("\"}", "\",\"x\":1}"),
        ("float64", "float_64"),
    ] {
        let v = with_meta(F64, from, to);
        assert_eq!(
            decompress_legacy_alice_zip(&v),
            Err(ContainerError::LegacyPayload),
            "{from} -> {to}"
        );
    }
}

#[test]
fn a_damaged_payload_is_refused() {
    // A byte inside the xz stream (its check catches it).
    let mut v = F64.to_vec();
    let at = xz_at(F64) + 40;
    v[at] ^= 0x10;
    assert_eq!(
        decompress_legacy_alice_zip(&v),
        Err(ContainerError::LegacyPayload)
    );

    // meta_len past the payload.
    let mut v = F64.to_vec();
    v[PAYLOAD_AT..PAYLOAD_AT + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        decompress_legacy_alice_zip(&v),
        Err(ContainerError::LegacyPayload)
    );

    // Payload shorter than the header states: the container layout check.
    assert_eq!(
        decompress_legacy_alice_zip(&F64[..F64.len() - 1]),
        Err(ContainerError::Layout)
    );

    // Every truncation and every single bit flip is refused or decodes to
    // the original; nothing panics.
    for len in 0..F64.len() {
        assert!(
            decompress_legacy_alice_zip(&F64[..len]).is_err(),
            "len {len}"
        );
    }
    for i in 0..F64.len() * 8 {
        let mut v = F64.to_vec();
        v[i / 8] ^= 1 << (i % 8);
        if let Ok(a) = decompress_legacy_alice_zip(&v) {
            assert_eq!(a.data, F64_ORIG, "bit {i}");
        }
    }
}
