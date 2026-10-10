//! Which header keys each residual version takes (tests/data/residual/
//! header_gating.txt, shared with tests/test_residual_header_gating.py):
//! every cell key x version must get the same verdict from both readers.

use std::path::Path;

use alice_core::residual::ResidualData;

#[test]
fn the_rust_reader_gives_every_verdict() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual");
    let table = std::fs::read_to_string(dir.join("header_gating.txt")).unwrap();
    let mut compared = 0;
    for line in table.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let c: Vec<&str> = line.split_whitespace().collect();
        let read = ResidualData::from_bytes(&std::fs::read(dir.join(c[0])).unwrap());
        assert_eq!(read.is_ok(), c[3] == "accept", "{line}: {:?}", read.err());
        compared += 1;
    }
    assert_eq!(compared, 20);
}
