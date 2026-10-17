//! The arithmetic identifier of residual files (versions 5 to 7)
//!
//! A residual is taken against generated values, so its file records the
//! arithmetic that generated them, and the reader compares the stored
//! identifier with `alice_zip::law::check_semantics`.
//!
//! Fixtures: `tests/data/residual/python_semantics.bin` (the Python writer,
//! numpy arithmetic: `alice_zip.semantics.PYTHON_SEMANTICS_ID`, checked by
//! `tests/test_residual_semantics.py`) and `rust_semantics.bin` (this crate's
//! writer, reproduced byte for byte below; `ALICE_WRITE_FIXTURES=1` rewrites
//! it). `python_none.bin` is a file of the layouts both readers share,
//! written without the identifier.

use alice_core::residual::{
    compress_original, ResidualCompressionMethod as M, ResidualData, ResidualError,
};
use alice_zip::law::SEMANTICS_ID;

const PYTHON: &[u8] = include_bytes!("../../tests/data/residual/python_semantics.bin");
const RUST: &[u8] = include_bytes!("../../tests/data/residual/rust_semantics.bin");
const LEGACY: &[u8] = include_bytes!("../../tests/data/residual/python_none.bin");

fn header(b: &[u8]) -> String {
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    String::from_utf8(b[4..4 + n].to_vec()).unwrap()
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn written_here() -> Vec<u8> {
    let original: Vec<u8> = [5.0f32, 5.5, 6.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    compress_original(&original, "float32", &[4.0, 5.0, 6.5, 4.25], M::None)
        .unwrap()
        .to_bytes()
}

#[test]
fn the_writer_records_this_crates_identifier_and_reproduces_the_fixture() {
    let bytes = written_here();
    if std::env::var_os("ALICE_WRITE_FIXTURES").is_some() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual");
        std::fs::write(dir.join("rust_semantics.bin"), &bytes).unwrap();
    }
    assert_eq!(
        bytes, RUST,
        "rust_semantics.bin is not this writer's output"
    );
    let h = header(&bytes);
    assert!(h.contains("\"version\":5"), "{h}");
    assert!(
        h.contains(&format!("\"semantics\":\"{}\"", hex(&SEMANTICS_ID))),
        "{h}"
    );
    let rd = ResidualData::from_bytes(&bytes).unwrap();
    assert_eq!(rd.metadata.semantics, Some(SEMANTICS_ID));
    assert!(ResidualData::from_bytes_strict(&bytes).is_ok());
    // written back as read
    assert_eq!(rd.to_bytes(), bytes);
}

#[test]
fn a_residual_the_python_writer_generated_under_numpy_is_refused_naming_both() {
    let h = header(PYTHON);
    let at = h
        .find("\"semantics\":\"")
        .expect("the Python file records its identifier")
        + 13;
    let stored_hex = &h[at..at + 64];
    assert_ne!(stored_hex, hex(&SEMANTICS_ID));
    match ResidualData::from_bytes(PYTHON) {
        Err(ResidualError::SemanticsMismatch { stored, running }) => {
            assert_eq!(hex(&stored), stored_hex);
            assert_eq!(running, SEMANTICS_ID);
            let text = ResidualError::SemanticsMismatch { stored, running }.to_string();
            assert!(
                text.contains(stored_hex) && text.contains(&hex(&SEMANTICS_ID)),
                "{text}"
            );
        }
        other => panic!("expected SemanticsMismatch, got {other:?}"),
    }
}

#[test]
fn a_file_without_the_identifier_reads_as_legacy_and_a_strict_read_refuses_it() {
    assert!(!header(LEGACY).contains("semantics"));
    let rd = ResidualData::from_bytes(LEGACY).unwrap();
    assert_eq!(rd.metadata.semantics, None);
    assert!(matches!(
        ResidualData::from_bytes_strict(LEGACY),
        Err(ResidualError::LegacySemantics)
    ));
}

#[test]
fn the_identifier_on_a_file_of_versions_1_to_4_is_refused() {
    // a version 2 header that carries "semantics" (no writer emits it)
    let h = header(RUST).replace("\"version\":5", "\"version\":2");
    let mut b = (h.len() as u32).to_le_bytes().to_vec();
    b.extend_from_slice(h.as_bytes());
    let n = u32::from_le_bytes([RUST[0], RUST[1], RUST[2], RUST[3]]) as usize;
    b.extend_from_slice(&RUST[4 + n..]);
    assert!(matches!(
        ResidualData::from_bytes(&b),
        Err(ResidualError::InvalidHeader(m)) if m.contains("semantics")
    ));
}

#[test]
fn a_malformed_identifier_is_refused() {
    for bad in ["zz", "ABAB", ""] {
        let h = header(RUST);
        let at = h.find("\"semantics\":\"").unwrap() + 13;
        let h = format!(
            "{}{}{}",
            &h[..at],
            bad.repeat(if bad.is_empty() { 1 } else { 64 / bad.len() }),
            &h[at + 64..]
        );
        let mut b = (h.len() as u32).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        let n = u32::from_le_bytes([RUST[0], RUST[1], RUST[2], RUST[3]]) as usize;
        b.extend_from_slice(&RUST[4 + n..]);
        assert!(
            matches!(
                ResidualData::from_bytes(&b),
                Err(ResidualError::InvalidHeader(_))
            ),
            "{bad}"
        );
    }
}
