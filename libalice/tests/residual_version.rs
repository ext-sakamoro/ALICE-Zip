//! ResidualData refuses a header version that was never written
//!
//! Fixtures: tests/data/residual/make_fixtures.py (the JSON carries the keys
//! of both the Rust and the Python writer, so both readers parse the same
//! bytes; tests/test_residual_version.py asserts the same verdicts).

use alice_core::residual::{ResidualCompressionMethod, ResidualData, ResidualError};

const V2: &[u8] = include_bytes!("../../tests/data/residual/residual_v2.bin");
const V3: &[u8] = include_bytes!("../../tests/data/residual/residual_v3.bin");
const V5: &[u8] = include_bytes!("../../tests/data/residual/residual_v5.bin");

#[test]
fn version_2_is_read() {
    let r = ResidualData::from_bytes(V2).expect("version 2");
    assert_eq!(r.method, ResidualCompressionMethod::Zlib);
    assert_eq!(r.original_len, 2);
}

#[test]
fn a_later_version_is_refused_instead_of_being_read_as_version_2() {
    assert!(matches!(
        ResidualData::from_bytes(V5),
        Err(ResidualError::UnsupportedVersion(5))
    ));
}

#[test]
fn version_3_without_exceptions_is_refused() {
    // version 3 is a file with exceptions (tests/test_residual_exceptions.py)
    assert!(matches!(
        ResidualData::from_bytes(V3),
        Err(ResidualError::MissingField(f)) if f == "exceptions"
    ));
}

#[test]
fn a_version_written_as_a_float_or_a_string_is_refused() {
    // writers emit "version":2; the Python reader takes only a JSON integer
    // too (tests/test_residual_version.py)
    for (name, bytes) in [
        (
            "float",
            &include_bytes!("../../tests/data/residual/residual_v2_float.bin")[..],
        ),
        (
            "string",
            &include_bytes!("../../tests/data/residual/residual_v2_string.bin")[..],
        ),
    ] {
        assert!(
            matches!(
                ResidualData::from_bytes(bytes),
                Err(ResidualError::InvalidHeader(_))
            ),
            "{name}"
        );
    }
}
