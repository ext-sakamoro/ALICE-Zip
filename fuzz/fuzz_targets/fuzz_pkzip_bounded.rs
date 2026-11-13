//! PKZIP archives from arbitrary bytes: the central directory is checked when the archive is opened, so the reader
//! never panics, returns every entry at exactly its declared size, never decodes a deflated entry past the entry
//! limit or the deflated entries together past the total limit, and returns stored entries no longer than the input
#![no_main]
use alice_zip::pkzip::{Archive, Limits, Method};
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
    let mut deflated = 0u64;
    for i in 0..archive.entries().len() {
        let entry = &archive.entries()[i];
        if let Ok(out) = archive.read_index(i) {
            let len = out.len() as u64;
            assert_eq!(len, entry.size(), "entry {i}: not the declared size");
            if entry.method() == Method::Deflated {
                assert!(
                    len <= max_entry,
                    "entry {i}: {len} bytes past the entry limit {max_entry}"
                );
                deflated += len;
            } else {
                assert!(
                    len <= (input.len() - 4) as u64,
                    "entry {i}: a stored entry longer than the archive"
                );
            }
        }
    }
    assert!(
        deflated <= max_total,
        "{deflated} deflated bytes past the total limit {max_total}"
    );
});
