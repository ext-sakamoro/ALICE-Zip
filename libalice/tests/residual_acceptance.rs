//! Which residual files the Rust reader accepts, and what it returns, from
//! the table both readers are tested against (tests/data/residual/acceptance.txt).
//!
//! The files are the real output of each writer (the Python writer, this
//! crate's writer, this crate's writer before the header keys were aligned)
//! plus hand-made files for header grammar.

use std::path::Path;

use alice_core::residual::{decompress, ResidualData, ResidualError};

const VALUES: [u32; 4] = [0x40A0_0000, 0x40B0_0000, 0x40C0_0000, 0x4080_0000];
const SPECIAL: [u32; 11] = [
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
        let read = ResidualData::from_bytes(&bytes).and_then(|r| decompress(&r));
        assert_eq!(
            read.is_ok(),
            rust == "accept",
            "{name} ({writer}): {:?}",
            read.as_ref().err()
        );
        if let (Ok(out), Some(want)) = (
            read,
            match values {
                "values" => Some(&VALUES[..]),
                "special" => Some(&SPECIAL[..]),
                _ => None,
            },
        ) {
            let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{name} ({writer})");
        }
        compared += 1;
    }
    assert_eq!(compared, 33, "every row compared");
}

#[test]
fn a_layout_this_reader_cannot_return_is_refused_by_name() {
    for name in [
        "python_lzma_float64.bin",
        "python_lzma_2d.bin",
        "python_quantized.bin",
    ] {
        let bytes = std::fs::read(dir().join(name)).unwrap();
        assert!(
            matches!(
                ResidualData::from_bytes(&bytes).and_then(|r| decompress(&r)),
                Err(ResidualError::UnsupportedLayout(_))
            ),
            "{name}"
        );
    }
}

#[test]
fn this_crate_writes_the_canonical_header() {
    // the Python writer's keys, in its order; the quantized container adds its
    // own parameters after them
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
    assert!(header("rust_quantized.bin").starts_with(
        r#"{"method":"quantized","shape":[4],"dtype":"float32","quant_bits":8,"version":2,"#
    ));
}
