//! `container::decompress_legacy_alice_zip` against files written by the
//! Python writer (`ALICEZip.compress`, `tests/data/container/make_legacy_lzma.py`).
//!
//! oracle: the `.orig` file next to each fixture holds the raw bytes of the
//! array the Python writer compressed; nothing here derives an expected value
//! from the reader itself.

#![cfg(feature = "lzma")]

use alice_zip::container::{decompress_legacy_alice_zip, ContainerError, LegacyArray};

/// Larger than every fixture, so the limit never decides these tests.
const LIMIT: u64 = 1 << 20;

fn decode(file: &[u8]) -> Result<LegacyArray, ContainerError> {
    decompress_legacy_alice_zip(file, LIMIT)
}

const U8: &[u8] = include_bytes!("data/container/legacy_lzma_u8.alice");
const U8_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_u8.orig");
const F64: &[u8] = include_bytes!("data/container/legacy_lzma_f64.alice");
const F64_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_f64.orig");
const SCALAR: &[u8] = include_bytes!("data/container/legacy_lzma_scalar.alice");
const SCALAR_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_scalar.orig");
const EMPTY: &[u8] = include_bytes!("data/container/legacy_lzma_empty.alice");
const SMALL: &[u8] = include_bytes!("data/container/legacy_lzma_small.alice");
const SMALL_ORIG: &[u8] = include_bytes!("data/container/legacy_lzma_small.orig");
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
    assert_eq!(decode(&v).unwrap().data, F64_ORIG);
}

#[test]
fn python_lzma_fallback_files_decode_to_the_original_bytes() {
    let a = decode(U8).unwrap();
    assert_eq!(a.data, U8_ORIG);
    assert_eq!(a.shape, vec![4096]);
    assert_eq!(a.dtype, "uint8");

    let a = decode(F64).unwrap();
    assert_eq!(a.data, F64_ORIG);
    assert_eq!(a.shape, vec![16, 8]);
    assert_eq!(a.dtype, "float64");
}

#[test]
fn zero_dimensional_and_empty_arrays_decode() {
    let a = decode(SCALAR).unwrap();
    assert_eq!(a.data, SCALAR_ORIG);
    assert_eq!(a.data, 3.5_f64.to_le_bytes());
    assert!(a.shape.is_empty());
    assert_eq!(a.dtype, "float64");

    let a = decode(SMALL).unwrap();
    assert_eq!(a.data, SMALL_ORIG);
    assert_eq!(a.data, [148, 111, 74, 37, 0]);

    let a = decode(EMPTY).unwrap();
    assert!(a.data.is_empty());
    assert_eq!(a.shape, vec![0]);
    assert_eq!(a.dtype, "int32");
}

#[test]
fn a_changed_stored_hash_is_refused() {
    for file in [U8, F64] {
        let mut v = file.to_vec();
        v[HASH_AT + 5] ^= 0x01;
        assert_eq!(decode(&v), Err(ContainerError::OriginalHash));
    }
}

#[test]
fn an_all_zero_hash_means_no_hash_recorded() {
    let mut v = F64.to_vec();
    v[HASH_AT..HASH_AT + 32].fill(0);
    assert_eq!(decode(&v).unwrap().data, F64_ORIG);
}

#[test]
fn a_stated_size_larger_than_the_result_is_refused() {
    // The LZMA2 chunk headers state 1024 bytes in total.
    assert_eq!(
        decode(&with_original_size(F64, 1025)),
        Err(ContainerError::LegacyPayload)
    );
}

#[test]
fn the_caller_limit_bounds_the_stated_size() {
    assert_eq!(
        decompress_legacy_alice_zip(U8, 4095),
        Err(ContainerError::LegacyTooLarge {
            stated: 4096,
            limit: 4095
        })
    );
    assert_eq!(decompress_legacy_alice_zip(U8, 4096).unwrap().data, U8_ORIG);
    // Checked before the payload is read at all.
    let huge = with_original_size(U8, u64::MAX);
    assert_eq!(
        decompress_legacy_alice_zip(&huge, LIMIT),
        Err(ContainerError::LegacyTooLarge {
            stated: u64::MAX,
            limit: LIMIT
        })
    );
}

#[test]
fn a_result_longer_than_the_stated_size_is_refused_without_finishing() {
    assert_eq!(
        decode(&with_original_size(F64, 1023)),
        Err(ContainerError::LegacyPayload)
    );
    assert_eq!(
        decode(&with_original_size(U8, 0)),
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
            decode(&v),
            Err(ContainerError::UnsupportedPayload {
                payload_type: Some(t)
            }),
            "{t:#04x}"
        );
    }
    assert_eq!(decode(V2), Err(ContainerError::LegacyPayload));
    assert_eq!(
        decode(V1),
        Err(ContainerError::UnsupportedPayload { payload_type: None })
    );
}

#[test]
fn shape_and_dtype_must_account_for_every_byte() {
    // Same JSON length, half the bytes.
    let v = with_meta(F64, "float64", "float32");
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
    let v = with_meta(F64, "[16,8]", "[16,4]");
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
    // A dtype name the writer never produced.
    let v = with_meta(U8, "\"uint8\"", "\"uint9\"");
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
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
            decode(&v),
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
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));

    // meta_len past the payload.
    let mut v = F64.to_vec();
    v[PAYLOAD_AT..PAYLOAD_AT + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));

    // Payload shorter than the header states: the container layout check.
    assert_eq!(decode(&F64[..F64.len() - 1]), Err(ContainerError::Layout));

    // Every truncation and every single bit flip is refused or decodes to
    // the original; nothing panics.
    for len in 0..F64.len() {
        assert!(decode(&F64[..len]).is_err(), "len {len}");
    }
    for i in 0..F64.len() * 8 {
        let mut v = F64.to_vec();
        v[i / 8] ^= 1 << (i % 8);
        if let Ok(a) = decode(&v) {
            assert_eq!(a.data, F64_ORIG, "bit {i}");
        }
    }
}

#[test]
fn every_bit_of_the_xz_container_is_covered_by_a_check() {
    // The container fields (stream header, block header, block padding, the
    // CRC-64 of the data, index, footer) are each under a CRC-32, the CRC-64
    // or a zero-padding rule, so any flipped bit there is refused. Inside the
    // LZMA2 data a flip is refused or, where it only touches range coder bits
    // past the end of the data (no xz check covers those), decodes to the
    // exact original; the CRC-64 and `original_hash` guarantee the result.
    for file in [U8, F64, SCALAR, SMALL] {
        let x = xz_at(file);
        let data = x + 12 + (usize::from(file[x + 12]) + 1) * 4;
        let backward = u32::from_le_bytes(file[file.len() - 8..file.len() - 4].try_into().unwrap());
        let index_size = (backward as usize + 1) * 4;
        let check = file.len() - 12 - index_size - 8;
        for i in x * 8..file.len() * 8 {
            let mut v = file.to_vec();
            v[i / 8] ^= 1 << (i % 8);
            let in_data = (data..check).contains(&(i / 8));
            match decode(&v) {
                Err(_) => {}
                Ok(a) if in_data => assert_eq!(a.data, decode(file).unwrap().data, "bit {i}"),
                Ok(_) => panic!(
                    "bit {i} (byte {}) of a container field was not checked",
                    i / 8
                ),
            }
        }
    }
    for i in xz_at(EMPTY) * 8..EMPTY.len() * 8 {
        let mut v = EMPTY.to_vec();
        v[i / 8] ^= 1 << (i % 8);
        assert!(decode(&v).is_err(), "bit {i} of the empty stream");
    }
}

/// CRC-32 (IEEE), bit by bit, for rebuilding xz fields in the tests.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// `file` with `extra` appended to the xz stream (`compressed_size` fixed).
fn with_trailing(file: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut v = file.to_vec();
    v.extend_from_slice(extra);
    let payload = (v.len() - PAYLOAD_AT) as u64;
    v[COMPRESSED_SIZE_AT..COMPRESSED_SIZE_AT + 8].copy_from_slice(&payload.to_le_bytes());
    v
}

#[test]
fn only_the_writers_xz_layout_is_read() {
    // Stream padding, a second stream, and junk after the footer.
    for extra in [&[0_u8; 4][..], &EMPTY[xz_at(EMPTY)..], b"junk"] {
        assert_eq!(
            decode(&with_trailing(F64, extra)),
            Err(ContainerError::LegacyPayload)
        );
    }
    // Check type CRC-32 (flags 0x01) instead of CRC-64, with the stream
    // header and footer CRCs recomputed so that only the type differs.
    let mut v = F64.to_vec();
    let x = xz_at(F64);
    v[x + 7] = 0x01;
    let c = crc32(&v[x + 6..x + 8]);
    v[x + 8..x + 12].copy_from_slice(&c.to_le_bytes());
    let f = v.len() - 12;
    v[f + 9] = 0x01;
    let c = crc32(&v[f + 4..f + 10]);
    v[f..f + 4].copy_from_slice(&c.to_le_bytes());
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
}

/// `file` with byte `at` of the xz block header set to `value` and the block
/// header CRC-32 recomputed.
fn with_block_header_byte(file: &[u8], at: usize, value: u8) -> Vec<u8> {
    let mut v = file.to_vec();
    let h = xz_at(file) + 12;
    let len = (usize::from(v[h]) + 1) * 4;
    v[h + at] = value;
    let c = crc32(&v[h..h + len - 4]);
    v[h + len - 4..h + len].copy_from_slice(&c.to_le_bytes());
    v
}

#[test]
fn block_and_chunk_rules_the_checks_do_not_cover_are_enforced() {
    // Block flags: two filters, or a reserved bit, with a valid CRC.
    for flags in [0x01, 0x04] {
        assert_eq!(
            decode(&with_block_header_byte(F64, 1, flags)),
            Err(ContainerError::LegacyPayload),
            "flags {flags:#04x}"
        );
    }
    // A filter other than LZMA2 (0x03 = delta).
    assert_eq!(
        decode(&with_block_header_byte(F64, 2, 0x03)),
        Err(ContainerError::LegacyPayload)
    );
    let data = xz_at(F64) + 12 + (usize::from(F64[xz_at(F64) + 12]) + 1) * 4;
    // The first chunk without a dictionary reset (0xE0 -> 0xC0) decodes the
    // same bytes, but the format requires the reset.
    assert_eq!(F64[data] & 0xE0, 0xE0);
    let mut v = F64.to_vec();
    v[data] &= !0x20;
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
    // The range coder's first byte is always 0.
    let mut v = F64.to_vec();
    v[data + 6] = 0x01;
    assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
}

/// Writes the CRC-32 of `v[start..end]` at `v[at..at + 4]`.
fn fix_crc(v: &mut [u8], start: usize, end: usize, at: usize) {
    let c = crc32(&v[start..end]);
    v[at..at + 4].copy_from_slice(&c.to_le_bytes());
}

/// Offset of the index (from the footer's backward size).
fn index_at(file: &[u8]) -> usize {
    let backward = u32::from_le_bytes(file[file.len() - 8..file.len() - 4].try_into().unwrap());
    file.len() - 12 - (backward as usize + 1) * 4
}

/// End of the LZMA2 chunks (after the end marker), walking the chunk headers.
fn chunks_end(file: &[u8]) -> (usize, usize) {
    let data = xz_at(file) + 12 + (usize::from(file[xz_at(file) + 12]) + 1) * 4;
    let mut at = data;
    loop {
        let c = file[at];
        let be = |i: usize| usize::from(u16::from_be_bytes([file[i], file[i + 1]]));
        at += match c {
            0x00 => return (data, at + 1),
            0x01 | 0x02 => 3 + be(at + 1) + 1,
            _ => 5 + usize::from(c >= 0xC0) + be(at + 3) + 1,
        };
    }
}

#[test]
fn fields_under_a_crc_are_checked_for_their_own_rules_too() {
    let x = xz_at(F64);
    let f = F64.len() - 12;
    let i = index_at(F64);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("stream header flags", {
            let mut v = F64.to_vec();
            v[x + 7] = 0x01;
            fix_crc(&mut v, x + 6, x + 8, x + 8);
            v
        }),
        ("stream footer flags", {
            let mut v = F64.to_vec();
            v[f + 9] = 0x01;
            fix_crc(&mut v, f + 4, f + 10, f);
            v
        }),
        ("backward size", {
            let mut v = F64.to_vec();
            v[f + 4] += 1;
            fix_crc(&mut v, f + 4, f + 10, f);
            v
        }),
        ("index record", {
            // Records: count 1, unpadded size, unpacked size 1024 (80 08).
            let mut v = F64.to_vec();
            assert_eq!(&v[i + 4..i + 6], [0x80, 0x08]);
            v[i + 4] = 0x81;
            fix_crc(&mut v, i, f - 4, f - 4);
            v
        }),
        ("index padding", {
            let mut v = F64.to_vec();
            assert_eq!(&v[i + 6..i + 8], [0, 0]);
            v[i + 6] = 1;
            fix_crc(&mut v, i, f - 4, f - 4);
            v
        }),
        ("block header padding", with_block_header_byte(F64, 7, 1)),
    ];
    for (name, v) in cases {
        assert_eq!(decode(&v), Err(ContainerError::LegacyPayload), "{name}");
    }

    // Block padding after the chunks (at least one fixture has some).
    let mut padded = 0;
    for file in [U8, F64, SCALAR, SMALL] {
        let (data, end) = chunks_end(file);
        if (end - data) % 4 != 0 {
            padded += 1;
            let mut v = file.to_vec();
            assert_eq!(v[end], 0);
            v[end] = 1;
            assert_eq!(decode(&v), Err(ContainerError::LegacyPayload));
        }
    }
    assert!(padded > 0, "no fixture has block padding");
}

#[test]
fn a_check_type_other_than_crc64_is_refused() {
    // Valid xz streams of the same data with another check type
    // (make_legacy_lzma.py); only the check type differs from the writer's.
    for file in [
        &include_bytes!("data/container/legacy_lzma_f64_crc32.alice")[..],
        &include_bytes!("data/container/legacy_lzma_f64_nocheck.alice")[..],
        &include_bytes!("data/container/legacy_lzma_f64_sha256.alice")[..],
    ] {
        assert_eq!(decode(file), Err(ContainerError::LegacyPayload));
    }
}

#[test]
fn the_dictionary_size_is_bounded_at_64_mib() {
    // Dictionary size byte (block header offset 4): the writer uses 22
    // (8 MiB); 28 is 64 MiB, 29 is 96 MiB, 40 is 4 GiB.
    assert_eq!(F64[xz_at(F64) + 12 + 4], 22);
    assert_eq!(
        decode(&with_block_header_byte(F64, 4, 28)).unwrap().data,
        F64_ORIG
    );
    for b in [29, 40] {
        assert_eq!(
            decode(&with_block_header_byte(F64, 4, b)),
            Err(ContainerError::LegacyPayload),
            "dict byte {b}"
        );
    }
}
