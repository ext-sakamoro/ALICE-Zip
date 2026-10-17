//! The released 0.10 reader and the containers the current version writes
//!
//! Every container the current version writes starts with the critical
//! `FMID` section, a tag 0.10 does not define, so 0.10 refuses it with
//! `UnknownCritical`; it still reads the containers written without it.

use released::container::{Container, ContainerError, ContainerView, Tag};

const CURRENT: [(&str, &[u8]); 6] = [
    (
        "empty",
        include_bytes!("../../../tests/data/container/empty.bin"),
    ),
    (
        "one",
        include_bytes!("../../../tests/data/container/one.bin"),
    ),
    (
        "dup_tag",
        include_bytes!("../../../tests/data/container/dup_tag.bin"),
    ),
    (
        "zero_len",
        include_bytes!("../../../tests/data/container/zero_len.bin"),
    ),
    (
        "unknown_noncritical",
        include_bytes!("../../../tests/data/container/unknown_noncritical.bin"),
    ),
    (
        "bundle",
        include_bytes!("../../../tests/data/container/bundle.bin"),
    ),
];
const EARLIER: [(&str, &[u8]); 2] = [
    (
        "v1_one",
        include_bytes!("../../../tests/data/container/v1_one.bin"),
    ),
    (
        "v1_lids",
        include_bytes!("../../../tests/data/container/v1_lids.bin"),
    ),
];

#[test]
fn the_released_reader_refuses_every_current_container() {
    let refused = ContainerError::UnknownCritical { tag: Tag(*b"FMID") };
    let mut compared = 0;
    for (name, bytes) in CURRENT {
        assert_eq!(ContainerView::open(bytes).err(), Some(refused), "{name}");
        assert_eq!(Container::from_bytes(bytes), Err(refused), "{name}");
        compared += 1;
    }
    assert_eq!(compared, CURRENT.len());
}

#[test]
fn the_released_reader_still_reads_the_containers_written_without_the_identifier() {
    for (name, bytes) in EARLIER {
        let c = Container::from_bytes(bytes).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(c.to_bytes(), bytes, "{name}");
    }
}
