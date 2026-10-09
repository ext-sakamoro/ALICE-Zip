//! Which residual fixtures the Rust reader accepts, from the table both
//! readers are tested against (tests/data/residual/acceptance.txt).

use std::path::Path;

use alice_core::residual::{decompress, ResidualData};

#[test]
fn the_rust_reader_accepts_exactly_the_files_the_table_lists() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual");
    let table = std::fs::read_to_string(dir.join("acceptance.txt")).unwrap();
    let mut compared = 0;
    for line in table.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let bytes = std::fs::read(dir.join(cols[0])).unwrap();
        let read = ResidualData::from_bytes(&bytes).and_then(|r| decompress(&r));
        assert_eq!(read.is_ok(), cols[1] == "accept", "{}: {:?}", cols[0], read.err());
        compared += 1;
    }
    assert_eq!(compared, 10, "every row compared");
}
