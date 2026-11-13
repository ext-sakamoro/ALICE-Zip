//! `pkzip::Archive`: a PKZIP reader that reads an entry only within the caller's limits
//!
//! oracle: the archives in `tests/data/pkzip/` are written by Python's `zipfile` (`scripts/gen_pkzip_fixtures.py`),
//! an independent writer, and the expected entry bytes are recomputed here from the formula the generator wrote
//! (`content`), not taken from the reader under test. The refusals patch those archives the way a hostile archive
//! would: a size that lies, two entries over the same bytes (the non-recursive zip bomb), a flag or method the reader
//! does not handle.
#![cfg(feature = "std")]

use std::io;
use std::path::Path;

use alice_zip::pkzip::{Archive, Limits, Method, PkzipError};

const BIG: u64 = 1 << 30;

fn data(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pkzip")
            .join(name),
    )
    .unwrap()
}

/// byte i of the entry with index k (the generator's formula)
fn content(index: usize, length: usize) -> Vec<u8> {
    (0..length)
        .map(|i| ((31 * i + 7 * index) % 251) as u8)
        .collect()
}

fn pkzip_error(e: &io::Error) -> PkzipError {
    *e.get_ref()
        .and_then(|i| i.downcast_ref::<PkzipError>())
        .unwrap_or_else(|| panic!("not a PkzipError: {e}"))
}

fn le16(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

/// offsets of the central directory records and of the end record (the archives here have no ZIP64 end records)
fn central(b: &[u8]) -> (Vec<usize>, usize) {
    let eocd = (0..b.len() - 21)
        .rev()
        .find(|&i| b[i..i + 4] == *b"PK\x05\x06")
        .unwrap();
    let mut at = u32::from_le_bytes(b[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    for _ in 0..le16(b, eocd + 10) {
        assert_eq!(&b[at..at + 4], b"PK\x01\x02");
        out.push(at);
        at += 46 + le16(b, at + 28) + le16(b, at + 30) + le16(b, at + 32);
    }
    (out, eocd)
}

fn central_of(b: &[u8], name: &str) -> usize {
    let (records, _) = central(b);
    *records
        .iter()
        .find(|&&r| &b[r + 46..r + 46 + le16(b, r + 28)] == name.as_bytes())
        .unwrap_or_else(|| panic!("{name} not in the central directory"))
}

fn local_of(b: &[u8], name: &str) -> usize {
    let r = central_of(b, name);
    u32::from_le_bytes(b[r + 42..r + 46].try_into().unwrap()) as usize
}

/// `name`'s uncompressed size set to `size` in its central record and its local header: the archive lies about how
/// far the entry expands
fn declaring(mut b: Vec<u8>, name: &str, size: u32) -> Vec<u8> {
    let (r, l) = (central_of(&b, name), local_of(&b, name));
    b[r + 24..r + 28].copy_from_slice(&size.to_le_bytes());
    b[l + 22..l + 26].copy_from_slice(&size.to_le_bytes());
    b
}

/// a copy of `name`'s central record, renamed `alias` (same length) and pointing at local header `offset`, appended
/// to the central directory: a second entry over bytes the archive already holds
fn aliased(b: &[u8], name: &str, alias: &str, offset: u32) -> Vec<u8> {
    assert_eq!(name.len(), alias.len());
    let (_, eocd) = central(b);
    let r = central_of(b, name);
    let len = 46 + le16(b, r + 28) + le16(b, r + 30) + le16(b, r + 32);
    let mut rec = b[r..r + len].to_vec();
    rec[46..46 + alias.len()].copy_from_slice(alias.as_bytes());
    rec[42..46].copy_from_slice(&offset.to_le_bytes());
    let mut out = b[..eocd].to_vec();
    out.extend_from_slice(&rec);
    let mut end = b[eocd..].to_vec();
    for at in [8, 10] {
        let n = le16(&end, at) as u16 + 1;
        end[at..at + 2].copy_from_slice(&n.to_le_bytes());
    }
    let size = u32::from_le_bytes(end[12..16].try_into().unwrap()) + len as u32;
    end[12..16].copy_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&end);
    out
}

#[test]
fn every_entry_of_the_zipfile_archives_reads_back_as_written() {
    let tsv = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pkzip/expected.tsv"),
    )
    .unwrap();
    let mut compared = 0;
    let mut archives: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    for line in tsv.lines().filter(|l| !l.starts_with('#')) {
        let cols: Vec<String> = line.split('\t').map(str::to_string).collect();
        match archives.last_mut() {
            Some((a, rows)) if *a == cols[0] => rows.push(cols),
            _ => archives.push((cols[0].clone(), vec![cols])),
        }
    }
    for (name, rows) in &archives {
        let archive = Archive::new(data(name), Limits::new(BIG, BIG)).unwrap();
        assert_eq!(archive.entries().len(), rows.len(), "{name}");
        for (k, row) in rows.iter().enumerate() {
            let entry = &archive.entries()[k];
            assert_eq!(entry.name(), row[1].as_bytes(), "{name}");
            let method = if row[2] == "0" {
                Method::Stored
            } else {
                Method::Deflated
            };
            assert_eq!(entry.method(), method, "{name} {}", row[1]);
            let length: usize = row[3].parse().unwrap();
            assert_eq!(entry.size(), length as u64);
            assert_eq!(format!("{:08x}", entry.crc32()), row[4]);
            let want = if row[1].ends_with('/') {
                Vec::new()
            } else {
                content(k, length)
            };
            assert_eq!(archive.read(&row[1]).unwrap(), want, "{name} {}", row[1]);
            assert_eq!(archive.read_index(k).unwrap(), want, "{name} #{k}");
            compared += 1;
        }
    }
    assert_eq!(archives.len(), 5);
    assert_eq!(compared, 17, "every row of expected.tsv was compared");
}

#[test]
fn an_entry_that_declares_more_than_the_entry_limit_is_refused_before_it_is_read() {
    let archive = Archive::new(data("basic.zip"), Limits::new(49_999, BIG)).unwrap();
    let e = archive.read("b.bin").unwrap_err();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::EntryTooLarge {
            declared: 50_000,
            limit: 49_999
        }
    );
    assert_eq!(
        archive.read("a.bin").unwrap(),
        content(0, 1000),
        "the limit is per entry"
    );
    // the limit itself is allowed
    let archive = Archive::new(data("basic.zip"), Limits::new(50_000, BIG)).unwrap();
    assert_eq!(archive.read("b.bin").unwrap().len(), 50_000);
}

#[test]
fn an_archive_whose_deflated_entries_declare_more_than_the_total_is_refused_when_it_is_opened() {
    // basic.zip's deflated entries declare 50 000 + 1 + 0 bytes; its stored entries (1 000 + 0 + 0) are bounded by
    // the archive itself and do not count
    let e = Archive::new(data("basic.zip"), Limits::new(BIG, 50_000))
        .err()
        .unwrap();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::TotalTooLarge {
            declared: 50_001,
            limit: 50_000
        }
    );
    let archive = Archive::new(data("basic.zip"), Limits::new(BIG, 50_001)).unwrap();
    // every read succeeds, in any order and any number of times: the limit does not depend on what was read
    for _ in 0..2 {
        for name in ["dir/c.txt", "b.bin", "a.bin", "empty.z"] {
            archive.read(name).unwrap();
        }
    }
}

#[test]
fn a_stored_entry_is_not_held_to_the_limits_and_is_read_in_place() {
    // stored data is bounded by the input the caller supplies, so a limit of 0 does not refuse it
    let archive = Archive::new(data("basic.zip"), Limits::new(0, 0)).err();
    assert!(
        archive.is_some(),
        "the deflated b.bin still counts against the total of 0"
    );
    let archive = Archive::new(data("aligned.zip"), Limits::new(0, 0)).unwrap();
    assert_eq!(archive.read("archive/data.pkl").unwrap(), content(0, 300));
    assert_eq!(
        archive.stored("archive/data/0").unwrap(),
        content(1, 40).as_slice()
    );
    let basic = Archive::new(data("basic.zip"), Limits::new(BIG, BIG)).unwrap();
    assert_eq!(
        basic.stored("b.bin").unwrap_err().kind(),
        io::ErrorKind::InvalidInput,
        "deflated: use read"
    );
    assert_eq!(
        basic.stored("nope").unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
}

#[test]
fn a_deflated_entry_that_expands_past_its_declared_size_is_refused() {
    // the declared size passes the entry limit, so the read is what must stop: the decoder is asked for at most the
    // declared size and one byte more
    let archive = Archive::new(
        declaring(data("basic.zip"), "b.bin", 1000),
        Limits::new(BIG, BIG),
    )
    .unwrap();
    let e = archive.read("b.bin").unwrap_err();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::ExpandsPastDeclared { declared: 1000 }
    );
    assert!(e.to_string().contains("expands past"), "{e}");
}

#[test]
fn a_deflated_entry_that_ends_short_of_its_declared_size_is_refused() {
    let archive = Archive::new(
        declaring(data("basic.zip"), "b.bin", 50_001),
        Limits::new(BIG, BIG),
    )
    .unwrap();
    let e = archive.read("b.bin").unwrap_err();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::ShortOfDeclared {
            declared: 50_001,
            actual: 50_000
        }
    );
}

#[test]
fn a_stored_entry_whose_sizes_differ_is_refused_when_the_archive_is_opened() {
    let e = Archive::new(
        declaring(data("basic.zip"), "a.bin", 999),
        Limits::new(BIG, BIG),
    )
    .err()
    .unwrap();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::StoredSizeMismatch {
            compressed: 1000,
            declared: 999
        }
    );
}

#[test]
fn an_entry_whose_bytes_do_not_match_its_crc_is_refused() {
    let mut b = data("basic.zip");
    let l = local_of(&b, "a.bin");
    let at = l + 30 + le16(&b, l + 26) + le16(&b, l + 28) + 500;
    b[at] ^= 1;
    let archive = Archive::new(b, Limits::new(BIG, BIG)).unwrap();
    let e = archive.read("a.bin").unwrap_err();
    assert!(
        matches!(pkzip_error(&e), PkzipError::CrcMismatch { .. }),
        "{e}"
    );
    let e = archive.stored("a.bin").unwrap_err();
    assert!(
        matches!(pkzip_error(&e), PkzipError::CrcMismatch { .. }),
        "in place too: {e}"
    );
}

#[test]
fn two_entries_over_the_same_bytes_are_refused_when_the_archive_is_opened() {
    // the non-recursive zip bomb: many central records point into one compressed stream, so the archive expands
    // by the number of records, not by what it holds
    let b = data("basic.zip");
    let same = aliased(&b, "b.bin", "B.bin", local_of(&b, "b.bin") as u32);
    let e = Archive::new(same, Limits::new(BIG, BIG)).err().unwrap();
    assert!(matches!(pkzip_error(&e), PkzipError::Overlap { .. }), "{e}");
    // a record whose header starts inside another entry's data
    let inside = aliased(&b, "b.bin", "B.bin", local_of(&b, "b.bin") as u32 + 40);
    let e = Archive::new(inside, Limits::new(BIG, BIG)).err().unwrap();
    assert!(
        matches!(
            pkzip_error(&e),
            PkzipError::Overlap { .. } | PkzipError::Malformed(_)
        ),
        "{e}"
    );
}

#[test]
fn two_entries_with_the_same_name_are_refused() {
    let mut b = data("basic.zip");
    let (r, l) = (central_of(&b, "b.bin"), local_of(&b, "b.bin"));
    b[r + 46] = b'a';
    b[l + 30] = b'a';
    let e = Archive::new(b, Limits::new(BIG, BIG)).err().unwrap();
    assert_eq!(pkzip_error(&e), PkzipError::DuplicateName { index: 1 });
}

#[test]
fn an_encrypted_entry_and_an_unknown_method_are_refused() {
    let mut b = data("basic.zip");
    let r = central_of(&b, "a.bin");
    b[r + 8] |= 1;
    let e = Archive::new(b, Limits::new(BIG, BIG)).err().unwrap();
    assert_eq!(pkzip_error(&e), PkzipError::Encrypted { index: 0 });

    let mut b = data("basic.zip");
    let r = central_of(&b, "b.bin");
    b[r + 10] = 12; // bzip2
    let e = Archive::new(b, Limits::new(BIG, BIG)).err().unwrap();
    assert_eq!(
        pkzip_error(&e),
        PkzipError::UnsupportedMethod {
            index: 1,
            method: 12
        }
    );
}

#[test]
fn a_local_header_that_names_another_entry_is_refused() {
    let mut b = data("basic.zip");
    let l = local_of(&b, "b.bin");
    b[l + 30] = b'x';
    let archive = Archive::new(b, Limits::new(BIG, BIG));
    let e = match archive {
        Err(e) => e,
        Ok(a) => a.read("b.bin").unwrap_err(),
    };
    assert!(matches!(pkzip_error(&e), PkzipError::Malformed(_)), "{e}");
}

#[test]
fn every_truncation_of_an_archive_is_refused_without_a_panic() {
    let b = data("basic.zip");
    for n in 0..b.len() {
        let r = std::panic::catch_unwind(|| {
            Archive::new(b[..n].to_vec(), Limits::new(BIG, BIG)).is_err()
        });
        assert_eq!(r.ok(), Some(true), "prefix of {n} bytes");
    }
}

#[test]
fn a_missing_name_is_not_found() {
    let archive = Archive::new(data("basic.zip"), Limits::new(BIG, BIG)).unwrap();
    assert_eq!(
        archive.read("nope").unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(
        archive.read_index(6).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );
}

#[test]
fn a_stored_entry_whose_range_runs_past_the_archive_is_refused_when_the_archive_is_opened() {
    // both sizes of a stored entry raised past the archive's end: the record would have the read slice bytes the
    // archive does not hold
    let mut b = data("basic.zip");
    let (r, l) = (central_of(&b, "a.bin"), local_of(&b, "a.bin"));
    for (rec, at) in [(r, 20), (r, 24), (l, 18), (l, 22)] {
        b[rec + at..rec + at + 4].copy_from_slice(&0x0010_0000u32.to_le_bytes());
    }
    let e = Archive::new(b, Limits::new(BIG, BIG)).err().unwrap();
    assert!(matches!(pkzip_error(&e), PkzipError::Malformed(_)), "{e}");
}

#[test]
fn a_zip64_locator_that_points_at_the_end_of_the_address_space_is_refused_without_a_panic() {
    // found by fuzz_pkzip_bounded: the record's offset plus its length overflowed
    for offset in [u64::MAX, u64::MAX - 2, u64::MAX - 55] {
        let mut b = Vec::new();
        b.extend_from_slice(b"PK\x06\x07");
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(b"PK\x05\x06");
        b.extend_from_slice(&[0u8; 18]);
        let r = std::panic::catch_unwind(|| {
            Archive::new(b.clone(), Limits::new(BIG, BIG))
                .err()
                .map(|e| pkzip_error(&e))
        });
        assert!(
            matches!(r, Ok(Some(PkzipError::Malformed(_)))),
            "offset {offset:#x}: {r:?}"
        );
    }
}
