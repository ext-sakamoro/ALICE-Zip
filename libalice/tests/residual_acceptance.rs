//! Which residual files the Rust reader accepts, and what it returns, from
//! the table both readers are tested against (tests/data/residual/acceptance.txt).
//!
//! The files are the real output of each writer (the Python writer, this
//! crate's writer, this crate's writer before the header keys were aligned)
//! plus hand-made files for header grammar.

use std::path::Path;

use alice_core::residual::{decompress, decompress_f64, ResidualData, ResidualError};

const VALUES: [u32; 4] = [0x40A0_0000, 0x40B0_0000, 0x40C0_0000, 0x4080_0000];
const SPECIAL11: [u32; 11] = [
    0x7FC0_0001,
    0x3F80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x8000_0000,
    0x0000_0000,
    0x0000_0001,
    0x7F7F_FFFF,
    0xFFFF_FFFF,
    0x3200_0000,
    0x4CBE_BC20,
];

/// `SPECIAL11` with two signaling NaNs (what the current writers were given).
const SPECIAL: [u32; 13] = [
    0x7FC0_0001,
    0x3F80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x8000_0000,
    0x0000_0000,
    0x0000_0001,
    0x7F7F_FFFF,
    0xFFFF_FFFF,
    0x3200_0000,
    0x4CBE_BC20,
    0x7F80_0001,
    0xFF80_0001,
];

/// The residual values of the `dtype_*` files, returned as `f32` whatever the
/// dtype of the original.
const WIDE: [f32; 4] = [-36.54, 300.5, 5.5, -129.75];

fn quantized_expected(name: &str) -> Vec<u32> {
    let text = std::fs::read_to_string(dir().join("quantized_expected.txt")).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with(&format!("{name} |")))
        .unwrap();
    line.split(" | ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .map(|h| u32::from_str_radix(h, 16).unwrap())
        .collect()
}

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual")
}

#[test]
fn the_rust_reader_accepts_exactly_the_files_the_table_lists() {
    let table = std::fs::read_to_string(dir().join("acceptance.txt")).unwrap();
    let mut compared = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let c: Vec<&str> = line.split_whitespace().collect();
        let (name, writer, rust, values) = (c[0], c[1], c[2], c[4]);
        let bytes = std::fs::read(dir().join(name)).unwrap();
        // a float64 residual (version 4) is read with decompress_f64
        let read = ResidualData::from_bytes(&bytes)
            .and_then(|r| decompress_f64(&r).map(|_| r))
            .and_then(|r| {
                if r.metadata.residual_f64 {
                    Ok(Vec::new())
                } else {
                    decompress(&r)
                }
            });
        assert_eq!(
            read.is_ok(),
            rust == "accept",
            "{name} ({writer}): {:?}",
            read.as_ref().err()
        );
        let want: Option<Vec<u32>> = match values {
            "values" => Some(VALUES.to_vec()),
            "special" => Some(SPECIAL.to_vec()),
            "special11" => Some(SPECIAL11.to_vec()),
            "wide" => Some(WIDE.iter().map(|v| v.to_bits()).collect()),
            "quant8" | "quant16" | "quant32" | "rand8" | "rand16" | "rand32" => {
                Some(quantized_expected(values))
            }
            _ => None,
        };
        if let (Ok(out), Some(want)) = (read, want) {
            let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{name} ({writer})");
        }
        compared += 1;
    }
    assert_eq!(compared, 159, "every row compared");
}

#[test]
fn the_original_shape_and_dtype_are_kept() {
    let read =
        |name: &str| ResidualData::from_bytes(&std::fs::read(dir().join(name)).unwrap()).unwrap();
    let r = read("python_lzma_2d.bin");
    assert_eq!(
        (r.shape.clone(), r.dtype.as_str(), r.original_len),
        (vec![2, 2], "float32", 4)
    );
    let r = read("python_lzma_float64.bin");
    assert_eq!((r.shape.clone(), r.dtype.as_str()), (vec![4], "float64"));
    // written back with the same header
    let b = std::fs::read(dir().join("python_lzma_2d.bin")).unwrap();
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let out = read("python_lzma_2d.bin").to_bytes();
    let m = u32::from_le_bytes([out[0], out[1], out[2], out[3]]) as usize;
    assert_eq!(&out[4..4 + m], &b[4..4 + n]);
}

#[test]
fn this_crate_writes_the_canonical_header() {
    // the Python writer's keys, in its order
    let header = |name: &str| {
        let b = std::fs::read(dir().join(name)).unwrap();
        let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        String::from_utf8(b[4..4 + n].to_vec()).unwrap()
    };
    assert_eq!(
        header("rust_none.bin"),
        r#"{"method":"none","shape":[4],"dtype":"float32","quant_bits":null,"version":2}"#
    );
    assert_eq!(
        header("rust_delta.bin"),
        r#"{"method":"bitdelta","shape":[4],"dtype":"float32","quant_bits":null,"version":2}"#
    );
    assert_eq!(
        header("rust_quantized.bin"),
        r#"{"method":"quantized","shape":[4],"dtype":"float32","quant_bits":8,"version":2}"#
    );
}

#[test]
fn shape_and_original_len_that_disagree_are_refused() {
    let bytes = std::fs::read(dir().join("residual_shape_len_mismatch.bin")).unwrap();
    assert!(matches!(
        ResidualData::from_bytes(&bytes),
        Err(ResidualError::InvalidHeader(m)) if m.contains("disagree")
    ));
}
