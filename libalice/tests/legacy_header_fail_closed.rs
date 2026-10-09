//! The ALICE_ZIP header reader refuses what the writer never produced
//!
//! Fixtures come from `tests/data/container/container_ref.py` (same `struct`
//! layout as `alice_zip/core.py`). Before this change the reader read a major
//! version of 2 as version 1.1 and an unknown `payload_type` as `Procedural`.

use alice_core::format::{AliceFileHeader, AlicePayloadType, FormatError};

const V1: &[u8] = include_bytes!("../../tests/data/container/legacy_v1.alice");
const V2: &[u8] = include_bytes!("../../tests/data/container/legacy_v2.alice");
const MAJOR2: &[u8] = include_bytes!("../../tests/data/container/legacy_major2.alice");
const UNKNOWN_PAYLOAD: &[u8] =
    include_bytes!("../../tests/data/container/legacy_unknown_payload.alice");

#[test]
fn files_the_writer_produced_are_still_read() {
    let h1 = AliceFileHeader::from_bytes(V1).expect("version 1.0");
    assert_eq!((h1.version_major, h1.version_minor), (1, 0));
    assert_eq!(h1.payload_type, AlicePayloadType::Procedural);
    let h2 = AliceFileHeader::from_bytes(V2).expect("version 1.1");
    assert_eq!((h2.version_major, h2.version_minor), (1, 1));
    assert_eq!(h2.payload_type, AlicePayloadType::LzmaFallback);
    assert_eq!(h2.engine_index, 3);
    assert_eq!(h2.compressed_size, (V2.len() - 66) as u64);
}

#[test]
fn a_version_that_was_never_written_is_refused() {
    assert_eq!(
        AliceFileHeader::from_bytes(MAJOR2),
        Err(FormatError::UnsupportedVersion { major: 2, minor: 0 })
    );
    let mut v = V1.to_vec();
    v[10] = 2;
    assert_eq!(
        AliceFileHeader::from_bytes(&v),
        Err(FormatError::UnsupportedVersion { major: 1, minor: 2 })
    );
}

#[test]
fn an_unknown_payload_type_is_refused() {
    assert_eq!(
        AliceFileHeader::from_bytes(UNKNOWN_PAYLOAD),
        Err(FormatError::InvalidPayloadType(0x7F))
    );
}

#[test]
fn an_engine_index_past_the_four_engines_is_refused() {
    let mut v = V2.to_vec();
    v[12] = 4;
    assert_eq!(
        AliceFileHeader::from_bytes(&v),
        Err(FormatError::InvalidEngine(4))
    );
}

#[test]
fn a_version_1_1_header_cut_to_65_bytes_is_refused() {
    assert_eq!(
        AliceFileHeader::from_bytes(&V2[..65]),
        Err(FormatError::TooShort {
            expected: 66,
            got: 65
        })
    );
}

#[test]
fn the_container_reader_and_this_reader_agree() {
    for (name, bytes) in [
        ("v1", V1),
        ("v2", V2),
        ("major2", MAJOR2),
        ("unknown", UNKNOWN_PAYLOAD),
    ] {
        let container = alice_zip::container::read_legacy_alice_zip(bytes).is_ok();
        let header = AliceFileHeader::from_bytes(bytes).is_ok();
        assert_eq!(container, header, "{name}");
    }
}
