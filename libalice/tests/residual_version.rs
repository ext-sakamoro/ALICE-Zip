//! ResidualData refuses a header version that was never written
//!
//! Fixtures: tests/data/residual/make_fixtures.py (the JSON carries the keys
//! of both the Rust and the Python writer, so both readers parse the same
//! bytes; tests/test_residual_version.py asserts the same verdicts).

use alice_core::residual::{ResidualCompressionMethod, ResidualData, ResidualError};

const V2: &[u8] = include_bytes!("../../tests/data/residual/residual_v2.bin");
const V3: &[u8] = include_bytes!("../../tests/data/residual/residual_v3.bin");

#[test]
fn version_2_is_read() {
    let r = ResidualData::from_bytes(V2).expect("version 2");
    assert_eq!(r.method, ResidualCompressionMethod::Zlib);
    assert_eq!(r.original_len, 2);
}

#[test]
fn a_later_version_is_refused_instead_of_being_read_as_version_2() {
    assert!(matches!(
        ResidualData::from_bytes(V3),
        Err(ResidualError::UnsupportedVersion(3))
    ));
}
