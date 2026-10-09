//! Delta residuals come back bit for bit; files without their base are refused
//!
//! Shared fixtures (tests/data/residual/make_fixtures.py) with
//! tests/test_residual_delta.py:
//!   residual_delta_python_legacy.bin  "delta", no base, xz            -> refused
//!   residual_delta_rust_legacy.bin    "delta", base_value, LZMA alone -> read
//!   residual_bitdelta.bin             "bitdelta", xz                   -> read
//! The delta streams themselves are compared with bitdelta_streams.txt by the
//! unit tests in src/residual.rs.

use alice_core::residual::{
    compress_residual_delta, decompress, ResidualCompressionMethod, ResidualData, ResidualError,
};

const PY_LEGACY: &[u8] =
    include_bytes!("../../tests/data/residual/residual_delta_python_legacy.bin");
const RS_LEGACY: &[u8] = include_bytes!("../../tests/data/residual/residual_delta_rust_legacy.bin");
const BITDELTA: &[u8] = include_bytes!("../../tests/data/residual/residual_bitdelta.bin");
const VALUES: [f32; 4] = [5.0, 5.5, 6.0, 4.0];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Writes `data` with the delta writer, reads the bytes back, returns the bits.
fn round_trip(data: &[f32]) -> Vec<u32> {
    let rd = compress_residual_delta(data);
    assert_eq!(rd.method, ResidualCompressionMethod::BitDelta);
    bits(&decompress(&ResidualData::from_bytes(&rd.to_bytes()).unwrap()).unwrap())
}

#[test]
fn the_writer_records_bitdelta_in_xz() {
    let rd = compress_residual_delta(&VALUES);
    assert_eq!(rd.method, ResidualCompressionMethod::BitDelta);
    assert_eq!(
        &rd.compressed[..6],
        &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
        "xz, as Python writes"
    );
}

#[test]
fn values_far_apart_in_magnitude_come_back_bit_for_bit() {
    // a float difference loses these (3.0 came back as 0.0, 0.3 as 0.296875)
    for v in [
        &[1e-8f32, 1.0, 1e8, 3.0][..],
        &[0.1, 123_456.7, 0.3][..],
        &[f32::from_bits(0x47), 1.0, f32::from_bits(0x47)][..],
        &VALUES[..],
    ] {
        assert_eq!(round_trip(v), bits(v), "{v:?}");
    }
}

#[test]
fn special_values_come_back_bit_for_bit() {
    let v: Vec<f32> = [
        0x7FC0_0001u32, // NaN with a payload
        0x3F80_0000,
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x8000_0000, // -0.0
        0x0000_0000,
        0x0000_0001, // smallest subnormal
        0x7F7F_FFFF, // largest finite
        0xFFFF_FFFF, // NaN, all bits set
        0x3F80_0000,
    ]
    .into_iter()
    .map(f32::from_bits)
    .collect();
    assert_eq!(round_trip(&v), bits(&v));
}

#[test]
fn random_bit_patterns_come_back_bit_for_bit() {
    // every u32 is an f32 bit pattern; a fixed linear congruential sequence
    let mut x = 0x2545_F491_u32;
    let v: Vec<f32> = (0..10_000)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(x)
        })
        .collect();
    assert_eq!(round_trip(&v), bits(&v));
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
fn a_bitdelta_file_written_by_python_reads_xz() {
    assert_eq!(
        decompress(&ResidualData::from_bytes(BITDELTA).unwrap()).unwrap(),
        VALUES
    );
}

#[test]
fn the_float_difference_method_that_never_shipped_is_not_read() {
    let mut v = BITDELTA.to_vec();
    let at = v.windows(10).position(|w| w == b"\"bitdelta\"").unwrap();
    v.splice(at..at + 10, b"\"delta2\"".iter().copied());
    // the JSON length prefix no longer matches; fix it
    let len = u32::from_le_bytes([v[0], v[1], v[2], v[3]]) - 2;
    v[..4].copy_from_slice(&len.to_le_bytes());
    assert!(matches!(
        ResidualData::from_bytes(&v),
        Err(ResidualError::UnknownMethod(_))
    ));
}

#[test]
fn choose_compression_never_labels_bytes_with_another_method() {
    use alice_core::residual::choose_compression;
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
