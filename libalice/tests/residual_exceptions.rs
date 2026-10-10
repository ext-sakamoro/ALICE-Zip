//! Lossless writing from the original: positions that generated + residual
//! cannot rebuild bit for bit are kept as exceptions (the original's bytes).
//! The same rule and format as the Python writer (tests/test_residual_exceptions.py);
//! both writers are given tests/data/residual/exception_cases.txt.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use alice_core::residual::{
    compress_original, reconstruct_bytes, ResidualCompressionMethod as M, ResidualData,
    ResidualError,
};

const DTYPES: [&str; 11] = [
    "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64",
];

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual")
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// (original bytes, generated) of each dtype in exception_cases.txt.
fn cases() -> BTreeMap<String, (Vec<u8>, Vec<f64>)> {
    let text = std::fs::read_to_string(dir().join("exception_cases.txt")).unwrap();
    let mut out: BTreeMap<String, (Vec<u8>, Vec<f64>)> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let c: Vec<&str> = line.split_whitespace().collect();
        let e = out.entry(c[0].to_owned()).or_default();
        e.0.extend(hex(c[1]));
        e.1.push(f64::from_bits(u64::from_str_radix(c[2], 16).unwrap()));
    }
    out
}

#[test]
fn the_shared_cases_come_back_bit_for_bit() {
    let cases = cases();
    assert_eq!(cases.len(), 11);
    for (dtype, (original, generated)) in &cases {
        for method in [M::None, M::Lzma, M::Zlib, M::BitDelta] {
            let rd = compress_original(original, dtype, generated, method).unwrap();
            let read = ResidualData::from_bytes(&rd.to_bytes()).unwrap();
            assert!(!read.metadata.exception_positions.is_empty(), "{dtype}");
            assert_eq!(
                reconstruct_bytes(generated, &read).unwrap(),
                *original,
                "{dtype} {method:?}"
            );
        }
    }
}

#[test]
fn both_writers_files_rebuild_the_original() {
    let mut compared = 0;
    for (dtype, (original, generated)) in cases() {
        for writer in ["python", "rust"] {
            let bytes = std::fs::read(dir().join(format!("{writer}_exc_{dtype}.bin"))).unwrap();
            let rd = ResidualData::from_bytes(&bytes).unwrap();
            assert_eq!(
                reconstruct_bytes(&generated, &rd).unwrap(),
                original,
                "{writer} {dtype}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 22);
}

#[test]
fn finite_data_the_rule_rebuilds_keeps_version_2() {
    let original: Vec<u8> = [5.0f32, 5.5, 6.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rd = compress_original(&original, "float32", &[4.0, 5.0, 6.5, 4.25], M::Lzma).unwrap();
    let b = rd.to_bytes();
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let h = String::from_utf8(b[4..4 + n].to_vec()).unwrap();
    assert!(
        h.contains("\"version\":2") && !h.contains("exceptions"),
        "{h}"
    );
}

#[test]
fn quantized_cannot_keep_exceptions() {
    let original: Vec<u8> = [1.0f32, f32::NAN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert!(matches!(
        compress_original(&original, "float32", &[1.0, 1.0], M::Quantized),
        Err(ResidualError::NotLossless)
    ));
}

#[test]
fn a_well_formed_file_with_exceptions_rebuilds_its_original() {
    let rd = ResidualData::from_bytes(&std::fs::read(dir().join("residual_exc_ok.bin")).unwrap())
        .unwrap();
    let out = reconstruct_bytes(&[0.0, 1.0, -1.0, 4.0], &rd).unwrap();
    let bits: Vec<u32> = out
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(bits, [0x40A0_0000, 0x7FC0_0001, 0x40A0_0000, 0xFF80_0000]);
}

/// Random bit patterns of every dtype, generated near the value, random
/// float64 bits (NaN, infinities, subnormals included) or the value itself.
#[test]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn random_bit_patterns_come_back_bit_for_bit() {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        x
    };
    for dtype in DTYPES {
        let width = match dtype {
            "int8" | "uint8" => 1,
            "float16" | "int16" | "uint16" => 2,
            "float32" | "int32" | "uint32" => 4,
            _ => 8,
        };
        for _ in 0..4 {
            let n = 2000;
            let original: Vec<u8> = (0..n * width).map(|_| (next() >> 56) as u8).collect();
            let generated: Vec<f64> = (0..n)
                .map(|i| {
                    let r = next();
                    match r % 3 {
                        0 => f64::from_bits(next()),
                        1 => (i as f64) * 0.5 - 300.0,
                        _ => ((r >> 11) as f64) / 1e6,
                    }
                })
                .collect();
            for method in [M::Lzma, M::BitDelta] {
                let rd = compress_original(&original, dtype, &generated, method).unwrap();
                let read = ResidualData::from_bytes(&rd.to_bytes()).unwrap();
                assert_eq!(
                    reconstruct_bytes(&generated, &read).unwrap(),
                    original,
                    "{dtype} {method:?}"
                );
            }
        }
    }
}

#[test]
fn a_value_that_is_not_finite_is_an_exception_even_when_it_would_rebuild() {
    // inf - 2.0 is inf and 2.0 + inf is inf again, but the rule keeps every
    // value that is not finite as it is (tests/test_residual_exceptions.py)
    let original: Vec<u8> = [f32::INFINITY, 1.0, f32::NEG_INFINITY]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let generated = [2.0, 1.0, f64::NEG_INFINITY];
    let rd = compress_original(&original, "float32", &generated, M::None).unwrap();
    assert_eq!(rd.metadata.exception_positions, [0, 2]);
}

#[test]
fn values_close_to_generated_need_no_exceptions_with_a_float64_residual() {
    // o - g is exact when they are close, so a float64 residual rebuilds every
    // value (tests/test_residual_exceptions.py); a float32 residual could not
    let mut x: u64 = 7;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let values: Vec<f64> = (0..2000).map(|_| (next() - 0.5) * 2000.0).collect();
    let generated: Vec<f64> = values.iter().map(|v| v + (next() - 0.5) * 0.4).collect();
    let original: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rd = compress_original(&original, "float64", &generated, M::Lzma).unwrap();
    let read = ResidualData::from_bytes(&rd.to_bytes()).unwrap();
    assert!(read.metadata.exception_positions.is_empty());
    assert!(read.metadata.residual_f64);
    assert_eq!(reconstruct_bytes(&generated, &read).unwrap(), original);
}

#[test]
fn a_float64_file_rebuilds_its_original() {
    let rd = ResidualData::from_bytes(&std::fs::read(dir().join("residual_v4_ok.bin")).unwrap())
        .unwrap();
    let out = reconstruct_bytes(&[0.0, 1.0, -1.0, 4.0], &rd).unwrap();
    let bits: Vec<u64> = out
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        bits,
        [
            0x4014_0000_0000_0000,
            0x7FF0_0000_0000_0001,
            0x4014_0000_0000_0000,
            0x4020_0000_0000_0000
        ]
    );
}

#[test]
fn the_writers_files_have_version_4_and_the_residual_dtype() {
    for dtype in DTYPES {
        let b = std::fs::read(dir().join(format!("rust_exc_{dtype}.bin"))).unwrap();
        let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        let h = String::from_utf8(b[4..4 + n].to_vec()).unwrap();
        let rdt = if matches!(dtype, "float64" | "int32" | "uint32" | "int64" | "uint64") {
            "float64"
        } else {
            "float32"
        };
        assert!(h.contains("\"version\":4"), "{h}");
        assert!(h.contains(&format!("\"residual_dtype\":\"{rdt}\"")), "{h}");
    }
}
