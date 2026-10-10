//! `reconstruct_bytes` against the vectors both implementations are held to
//! (tests/data/residual/reconstruct_vectors.txt, computed by make_fixtures.py
//! with Python floats, ints and struct only).

use std::path::Path;

use alice_core::residual::{choose_compression, reconstruct_bytes, ResidualError};

#[test]
fn reconstruct_gives_every_vector() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/data/residual/reconstruct_vectors.txt");
    let table = std::fs::read_to_string(path).unwrap();
    let mut compared = 0;
    let mut dtypes = std::collections::BTreeSet::new();
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let c: Vec<&str> = line.split_whitespace().collect();
        let g = f64::from_bits(u64::from_str_radix(c[1], 16).unwrap());
        let r = f32::from_bits(u32::from_str_radix(c[2], 16).unwrap());
        let mut rd = choose_compression(&[r], false);
        rd.dtype = c[0].to_owned();
        let got = reconstruct_bytes(&[g], &rd);
        if c[3] == "error" {
            assert!(
                matches!(&got, Err(ResidualError::Corrupted(m)) if m.contains("NaN")),
                "{line}: {got:?}"
            );
        } else {
            let hex: String = got.unwrap().iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, c[3], "{line}");
        }
        dtypes.insert(c[0].to_owned());
        compared += 1;
    }
    assert_eq!((compared, dtypes.len()), (304, 11));
}

#[test]
fn generated_and_residual_of_different_lengths_are_refused() {
    let rd = choose_compression(&[1.0, 2.0], false);
    assert!(matches!(
        reconstruct_bytes(&[1.0], &rd),
        Err(ResidualError::Corrupted(_))
    ));
}
