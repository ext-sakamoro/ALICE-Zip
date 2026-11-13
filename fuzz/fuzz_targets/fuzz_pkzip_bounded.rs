//! PKZIP archives from arbitrary bytes: the central directory is checked when the archive is opened and every read
//! is held to the caller's limits, so the reader never panics, never returns an entry longer than the entry limit
//! or than the size the entry declares, and never lets the reads together produce more than the total limit
#![no_main]
use alice_zip::pkzip::{Archive, Limits};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    if input.len() < 4 || input.len() > 65536 {
        return;
    }
    // the first four bytes choose the entry limit and the total limit (up to 64 KiB each), the rest is the archive
    let max_entry = u64::from(u16::from_le_bytes([input[0], input[1]]));
    let max_total = u64::from(u16::from_le_bytes([input[2], input[3]]));
    let Ok(archive) = Archive::new(&input[4..], Limits::new(max_entry, max_total)) else {
        return;
    };
    let mut produced = 0u64;
    for i in 0..archive.entries().len() {
        if let Ok(out) = archive.read_index(i) {
            let len = out.len() as u64;
            assert!(len <= max_entry, "entry {i}: {len} bytes past the entry limit {max_entry}");
            assert_eq!(len, archive.entries()[i].size(), "entry {i}: not the declared size");
            produced += len;
        }
    }
    assert!(produced <= max_total, "{produced} bytes past the total limit {max_total}");
    assert_eq!(produced, archive.spent(), "spent counts exactly what was produced");
});
