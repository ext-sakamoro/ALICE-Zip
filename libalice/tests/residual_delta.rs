//! Delta residuals keep their first value; files without it are refused
//!
//! Shared fixtures (tests/data/residual/make_fixtures.py) with
//! tests/test_residual_delta.py:
//!   residual_delta_python_legacy.bin  "delta", no base, xz          -> refused
//!   residual_delta_rust_legacy.bin    "delta", base_value, LZMA alone -> read
//!   residual_delta2.bin               "delta2", xz                   -> read

use alice_core::residual::{
    choose_compression, compress_residual_delta, decompress, ResidualCompressionMethod,
    ResidualData, ResidualError,
};

const PY_LEGACY: &[u8] =
    include_bytes!("../../tests/data/residual/residual_delta_python_legacy.bin");
const RS_LEGACY: &[u8] = include_bytes!("../../tests/data/residual/residual_delta_rust_legacy.bin");
const DELTA2: &[u8] = include_bytes!("../../tests/data/residual/residual_delta2.bin");
const VALUES: [f32; 4] = [5.0, 5.5, 6.0, 4.0];

#[test]
fn the_writer_records_delta2_in_xz_and_restores_the_values() {
    let rd = compress_residual_delta(&VALUES);
    assert_eq!(rd.method, ResidualCompressionMethod::Delta2);
    assert_eq!(
        &rd.compressed[..6],
        &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
        "xz, as Python writes"
    );
    let back = decompress(&ResidualData::from_bytes(&rd.to_bytes()).unwrap()).unwrap();
    assert_eq!(back, VALUES);
}

#[test]
fn a_delta_file_without_its_base_is_refused() {
    assert!(matches!(
        ResidualData::from_bytes(PY_LEGACY),
        Err(ResidualError::DeltaWithoutBase)
    ));
}

#[test]
fn an_earlier_rust_delta_file_reads_lzma_alone() {
    assert_eq!(
        decompress(&ResidualData::from_bytes(RS_LEGACY).unwrap()).unwrap(),
        VALUES
    );
}

#[test]
fn a_delta2_file_written_by_python_reads_xz() {
    assert_eq!(
        decompress(&ResidualData::from_bytes(DELTA2).unwrap()).unwrap(),
        VALUES
    );
}

#[test]
fn choose_compression_never_labels_bytes_with_another_method() {
    // whatever the chooser picks, the recorded method decodes its own bytes
    for data in [
        &VALUES[..],
        &[0.0f32; 64][..],
        &[1.0, -1.0, 1.0, -1.0, 3.5][..],
    ] {
        let rd = choose_compression(data, false);
        let back = decompress(&ResidualData::from_bytes(&rd.to_bytes()).unwrap()).unwrap();
        assert_eq!(back, data, "{:?}", rd.method);
    }
}
