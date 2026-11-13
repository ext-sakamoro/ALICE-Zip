//! A PKZIP reader that reads an entry only within the caller's limits
//!
//! [`Archive::new`] reads the central directory and refuses an archive whose
//! records cannot all be honoured: two entries over the same bytes (the
//! non-recursive zip bomb, where many records point into one compressed stream
//! so the archive expands by the number of records), two entries with the same
//! name, an encrypted entry, a method other than stored or deflate, a stored
//! entry whose two sizes differ, a record that points outside the archive.
//!
//! **Stored data is bounded by the input length you supply; deflated output by
//! your caps.** A stored entry is its own bytes inside the archive, and no two
//! entries may cover the same bytes, so the stored entries together are at
//! most the archive's length: they are checked (two equal sizes, inside the
//! archive, CRC-32) but not held to the caps, and [`Archive::in_place`] returns
//! them in place. A deflated entry can expand without bound, so two caps the
//! caller must give ([`Limits`]) hold it: the most bytes one deflated entry
//! may declare, and the most bytes all deflated entries may declare together.
//! The total is checked once, when the archive is opened, so whether a read is
//! allowed does not depend on what was read before or how often.
//!
//! [`Archive::read`] decodes one entry: the size it declares is checked against
//! the entry cap before anything is decoded, and the decoder is asked for at
//! most that size and one byte more, so an entry that expands past what it
//! declares is refused at that byte, and one that ends short of it is refused
//! too. The effective limit of a read is therefore `min(declared size, entry
//! cap)`, as for
//! [`compression::zlib_decompress`](crate::compression::zlib_decompress). The
//! CRC-32 of the result is checked last.
//!
//! The sizes, offsets and CRC come from the central directory (with its ZIP64
//! extra field and end records); a local header is read only for where the
//! entry's data starts, and its name must match. Data descriptors are
//! therefore not read. Archives that span several disks are refused.
//!
//! A refusal is an [`io::Error`] of kind `InvalidData` carrying a
//! [`PkzipError`]: read it back with
//! `err.get_ref().and_then(|e| e.downcast_ref::<PkzipError>())`.

use core::fmt;
use std::io::{self, Read};

use flate2::read::DeflateDecoder;

/// The two caps on deflated output: both are required (stored entries are bounded by the archive itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    max_entry: u64,
    max_total: u64,
}

impl Limits {
    /// `max_entry`: the most bytes one deflated entry may declare (and so expand to); `max_total`: the most bytes
    /// all deflated entries of an archive may declare together, checked when the archive is opened.
    #[must_use]
    pub const fn new(max_entry: u64, max_total: u64) -> Self {
        Self {
            max_entry,
            max_total,
        }
    }

    /// The most bytes one deflated entry may declare.
    #[must_use]
    pub const fn max_entry(&self) -> u64 {
        self.max_entry
    }

    /// The most bytes all deflated entries of an archive may declare together.
    #[must_use]
    pub const fn max_total(&self) -> u64 {
        self.max_total
    }
}

/// How an entry's bytes are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// As they are (method 0).
    Stored,
    /// Raw deflate (method 8).
    Deflated,
}

/// Why an archive or an entry was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PkzipError {
    /// The deflated entry declares more bytes than the entry cap: refused before decoding.
    EntryTooLarge {
        /// The size the archive declares.
        declared: u64,
        /// The caller's entry limit.
        limit: u64,
    },
    /// The deflated entries together declare more bytes than the total cap: refused when the archive is opened.
    TotalTooLarge {
        /// The sum of the sizes the deflated entries declare (saturated at `u64::MAX`).
        declared: u64,
        /// The caller's total cap.
        limit: u64,
    },
    /// The entry expands past the size it declares: decoding stopped one byte past it.
    ExpandsPastDeclared {
        /// The size the archive declares.
        declared: u64,
    },
    /// The entry ends before the size it declares.
    ShortOfDeclared {
        /// The size the archive declares.
        declared: u64,
        /// The bytes the entry produced.
        actual: u64,
    },
    /// The entry's bytes do not have the CRC-32 the archive records.
    CrcMismatch {
        /// The CRC-32 the archive records.
        expected: u32,
        /// The CRC-32 of the bytes read.
        actual: u32,
    },
    /// A stored entry whose compressed and uncompressed sizes differ.
    StoredSizeMismatch {
        /// The compressed (stored) size.
        compressed: u64,
        /// The uncompressed size.
        declared: u64,
    },
    /// Two entries (central directory indices) cover the same bytes of the archive.
    Overlap {
        /// The entry that starts first.
        first: usize,
        /// The entry that starts inside it.
        second: usize,
    },
    /// The entry at this central directory index has the name of an earlier one.
    DuplicateName {
        /// The later entry's index.
        index: usize,
    },
    /// The entry at this index is encrypted.
    Encrypted {
        /// The entry's index.
        index: usize,
    },
    /// The entry at this index uses a method other than stored (0) or deflate (8).
    UnsupportedMethod {
        /// The entry's index.
        index: usize,
        /// The method number.
        method: u16,
    },
    /// The archive's structure is not a readable PKZIP archive.
    Malformed(&'static str),
}

impl fmt::Display for PkzipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntryTooLarge { declared, limit } => {
                write!(
                    f,
                    "ZIP entry declares {declared} bytes, above the entry limit of {limit} bytes"
                )
            }
            Self::TotalTooLarge { declared, limit } => write!(
                f,
                "the deflated ZIP entries declare {declared} bytes together, above the total limit of {limit} bytes"
            ),
            Self::ExpandsPastDeclared { declared } => {
                write!(f, "ZIP entry expands past the {declared} bytes it declares")
            }
            Self::ShortOfDeclared { declared, actual } => {
                write!(
                    f,
                    "ZIP entry ends after {actual} of the {declared} bytes it declares"
                )
            }
            Self::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "ZIP entry has CRC-32 {actual:08x}, the archive records {expected:08x}"
                )
            }
            Self::StoredSizeMismatch {
                compressed,
                declared,
            } => write!(
                f,
                "stored ZIP entry has {compressed} stored bytes but declares {declared}"
            ),
            Self::Overlap { first, second } => {
                write!(
                    f,
                    "ZIP entries {first} and {second} cover the same bytes of the archive"
                )
            }
            Self::DuplicateName { index } => write!(f, "ZIP entry {index} repeats an earlier name"),
            Self::Encrypted { index } => write!(f, "ZIP entry {index} is encrypted"),
            Self::UnsupportedMethod { index, method } => {
                write!(
                    f,
                    "ZIP entry {index} uses method {method} (only stored and deflate are read)"
                )
            }
            Self::Malformed(what) => write!(f, "not a readable ZIP archive: {what}"),
        }
    }
}

impl std::error::Error for PkzipError {}

fn refuse(e: PkzipError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

fn malformed(what: &'static str) -> io::Error {
    refuse(PkzipError::Malformed(what))
}

/// One entry of the central directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    name: Vec<u8>,
    method: Method,
    compressed_size: u64,
    size: u64,
    crc32: u32,
    header_offset: u64,
    data_offset: u64,
}

impl Entry {
    /// The name as the archive stores it.
    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    /// The name, when it is UTF-8.
    #[must_use]
    pub fn name_str(&self) -> Option<&str> {
        core::str::from_utf8(&self.name).ok()
    }

    /// How the bytes are stored.
    #[must_use]
    pub fn method(&self) -> Method {
        self.method
    }

    /// The bytes the entry occupies in the archive.
    #[must_use]
    pub fn size_in_archive(&self) -> u64 {
        self.compressed_size
    }

    /// The bytes the entry declares it expands to.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The CRC-32 the archive records for the entry's bytes.
    #[must_use]
    pub fn crc32(&self) -> u32 {
        self.crc32
    }
}

/// A PKZIP archive held in memory, read within [`Limits`].
#[derive(Debug)]
pub struct Archive<D: AsRef<[u8]>> {
    data: D,
    entries: Vec<Entry>,
    limits: Limits,
}

const EOCD: &[u8; 4] = b"PK\x05\x06";
const EOCD_LEN: usize = 22;
const ZIP64_LOCATOR: &[u8; 4] = b"PK\x06\x07";
const ZIP64_EOCD: &[u8; 4] = b"PK\x06\x06";
const CENTRAL: &[u8; 4] = b"PK\x01\x02";
const CENTRAL_LEN: usize = 46;
const LOCAL: &[u8; 4] = b"PK\x03\x04";
const LOCAL_LEN: usize = 30;

fn u16_at(b: &[u8], at: usize) -> io::Result<u16> {
    b.get(at..at.saturating_add(2))
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| malformed("a record runs past the end of the archive"))
}

fn u32_at(b: &[u8], at: usize) -> io::Result<u32> {
    b.get(at..at.saturating_add(4))
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| malformed("a record runs past the end of the archive"))
}

fn u64_at(b: &[u8], at: usize) -> io::Result<u64> {
    b.get(at..at.saturating_add(8))
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| malformed("a record runs past the end of the archive"))
}

fn index(v: u64) -> io::Result<usize> {
    usize::try_from(v).map_err(|_| malformed("an offset does not fit in memory"))
}

/// Where the central directory is and how many records it holds, from the end record (and its ZIP64 records when
/// a field of the end record is saturated)
fn directory(b: &[u8]) -> io::Result<(usize, usize, u64)> {
    if b.len() < EOCD_LEN {
        return Err(malformed("shorter than an end of central directory record"));
    }
    // the end record is followed only by its comment (at most 65535 bytes): the last signature whose comment
    // length reaches exactly the end of the archive
    let lowest = b.len().saturating_sub(EOCD_LEN + usize::from(u16::MAX));
    let eocd = (lowest..=b.len() - EOCD_LEN)
        .rev()
        .find(|&i| {
            &b[i..i + 4] == EOCD
                && u16_at(b, i + 20).is_ok_and(|c| i + EOCD_LEN + usize::from(c) == b.len())
        })
        .ok_or_else(|| malformed("no end of central directory record"))?;
    let disk = u16_at(b, eocd + 4)?;
    let cd_disk = u16_at(b, eocd + 6)?;
    let on_disk = u16_at(b, eocd + 8)?;
    let count16 = u16_at(b, eocd + 10)?;
    let size32 = u32_at(b, eocd + 12)?;
    let offset32 = u32_at(b, eocd + 16)?;
    let saturated = count16 == u16::MAX || size32 == u32::MAX || offset32 == u32::MAX;
    // a ZIP64 end locator sits right before the end record whenever the writer wrote ZIP64 end records, which a
    // writer may do without saturating a field of the end record (Python's zipfile does)
    let locator = eocd
        .checked_sub(20)
        .filter(|&l| &b[l..l + 4] == ZIP64_LOCATOR);
    if saturated && locator.is_none() {
        return Err(malformed("no ZIP64 end locator"));
    }
    let (disk, cd_disk, on_disk, count, size, offset, end) = if let Some(locator) = locator {
        if u32_at(b, locator + 16)? != 1 {
            return Err(malformed("the archive spans several disks"));
        }
        let at = index(u64_at(b, locator + 8)?)?;
        if b.get(at..at.saturating_add(4)) != Some(ZIP64_EOCD.as_slice())
            || at.saturating_add(56) > locator
        {
            return Err(malformed("no ZIP64 end of central directory record"));
        }
        (
            u32_at(b, at + 16)?,
            u32_at(b, at + 20)?,
            u64_at(b, at + 24)?,
            u64_at(b, at + 32)?,
            u64_at(b, at + 40)?,
            u64_at(b, at + 48)?,
            at,
        )
    } else {
        (
            u32::from(disk),
            u32::from(cd_disk),
            u64::from(on_disk),
            u64::from(count16),
            u64::from(size32),
            u64::from(offset32),
            eocd,
        )
    };
    if disk != 0 || cd_disk != 0 || on_disk != count {
        return Err(malformed("the archive spans several disks"));
    }
    let (start, size) = (index(offset)?, index(size)?);
    if start.checked_add(size) != Some(end) {
        return Err(malformed(
            "the central directory does not end where the end record starts",
        ));
    }
    // every record is at least CENTRAL_LEN bytes, so the count is bounded by the directory's size before any
    // allocation from it
    if count > (size / CENTRAL_LEN) as u64 {
        return Err(malformed(
            "more records than the central directory can hold",
        ));
    }
    Ok((start, end, count))
}

/// The ZIP64 extended information extra field (id 0x0001) fills, in this order, the fields the record saturates
fn zip64_fields(extra: &[u8], size: &mut u64, csize: &mut u64, offset: &mut u64) -> io::Result<()> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let id = u16_at(extra, at)?;
        let len = usize::from(u16_at(extra, at + 2)?);
        let body = extra
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| malformed("an extra field runs past its record"))?;
        if id == 1 {
            let mut p = 0;
            for field in [size, csize, offset] {
                if *field == u64::from(u32::MAX) {
                    *field = u64_at(body, p).map_err(|_| malformed("a ZIP64 field is missing"))?;
                    p += 8;
                }
            }
            return Ok(());
        }
        at += 4 + len;
    }
    if *size == u64::from(u32::MAX)
        || *csize == u64::from(u32::MAX)
        || *offset == u64::from(u32::MAX)
    {
        return Err(malformed("a saturated size or offset has no ZIP64 field"));
    }
    Ok(())
}

impl<D: AsRef<[u8]>> Archive<D> {
    /// Read the central directory of `data`; refuses an archive whose records cannot all be honoured (see the
    /// module documentation).
    ///
    /// # Errors
    ///
    /// An [`io::Error`] of kind `InvalidData` carrying a [`PkzipError`].
    pub fn new(data: D, limits: Limits) -> io::Result<Self> {
        let b = data.as_ref();
        let (start, end, count) = directory(b)?;
        let mut entries: Vec<Entry> = Vec::with_capacity(index(count)?);
        // the byte range each entry covers: its local header, name, extra field and data
        let mut spans: Vec<(u64, u64, usize)> = Vec::with_capacity(entries.capacity());
        let mut at = start;
        for i in 0..index(count)? {
            if b.get(at..at.saturating_add(4)) != Some(CENTRAL.as_slice()) {
                return Err(malformed("a central directory record has no signature"));
            }
            let flags = u16_at(b, at + 8)?;
            let method = u16_at(b, at + 10)?;
            let crc32 = u32_at(b, at + 16)?;
            let mut csize = u64::from(u32_at(b, at + 20)?);
            let mut size = u64::from(u32_at(b, at + 24)?);
            let name_len = usize::from(u16_at(b, at + 28)?);
            let extra_len = usize::from(u16_at(b, at + 30)?);
            let comment_len = usize::from(u16_at(b, at + 32)?);
            let mut offset = u64::from(u32_at(b, at + 42)?);
            let name_at = at + CENTRAL_LEN;
            let next = name_at + name_len + extra_len + comment_len;
            if next > end {
                return Err(malformed(
                    "a central directory record runs past the directory",
                ));
            }
            let name = b[name_at..name_at + name_len].to_vec();
            zip64_fields(
                &b[name_at + name_len..name_at + name_len + extra_len],
                &mut size,
                &mut csize,
                &mut offset,
            )?;
            if flags & 1 != 0 {
                return Err(refuse(PkzipError::Encrypted { index: i }));
            }
            let method = match method {
                0 => Method::Stored,
                8 => Method::Deflated,
                m => {
                    return Err(refuse(PkzipError::UnsupportedMethod {
                        index: i,
                        method: m,
                    }))
                }
            };
            if method == Method::Stored && csize != size {
                return Err(refuse(PkzipError::StoredSizeMismatch {
                    compressed: csize,
                    declared: size,
                }));
            }
            if entries.iter().any(|e| e.name == name) {
                return Err(refuse(PkzipError::DuplicateName { index: i }));
            }
            let local = index(offset)?;
            if local.checked_add(LOCAL_LEN).is_none_or(|e| e > start) {
                return Err(malformed("a local header lies outside the entries' area"));
            }
            let data_offset = (local + LOCAL_LEN) as u64
                + u64::from(u16_at(b, local + 26)?)
                + u64::from(u16_at(b, local + 28)?);
            let data_end = data_offset
                .checked_add(csize)
                .filter(|&e| e <= start as u64)
                .ok_or_else(|| malformed("an entry's data runs past the entries' area"))?;
            spans.push((offset, data_end, i));
            entries.push(Entry {
                name,
                method,
                compressed_size: csize,
                size,
                crc32,
                header_offset: offset,
                data_offset,
            });
            at = next;
        }
        if at != end {
            return Err(malformed(
                "the central directory has bytes after its last record",
            ));
        }
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(refuse(PkzipError::Overlap {
                    first: w[0].2,
                    second: w[1].2,
                }));
            }
        }
        for e in &entries {
            let local = index(e.header_offset)?;
            if b.get(local..local + 4) != Some(LOCAL.as_slice()) {
                return Err(malformed("a local header has no signature"));
            }
            let name_len = usize::from(u16_at(b, local + 26)?);
            if b.get(local + LOCAL_LEN..local + LOCAL_LEN + name_len) != Some(e.name.as_slice()) {
                return Err(malformed("a local header names another entry"));
            }
        }
        let declared = entries
            .iter()
            .filter(|e| e.method == Method::Deflated)
            .fold(0u64, |t, e| t.saturating_add(e.size));
        if declared > limits.max_total {
            return Err(refuse(PkzipError::TotalTooLarge {
                declared,
                limit: limits.max_total,
            }));
        }
        Ok(Self {
            data,
            entries,
            limits,
        })
    }

    /// The entries in central directory order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The limits every read is held to.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// The bytes of the entry named `name`, within the limits.
    ///
    /// # Errors
    ///
    /// `NotFound` when no entry has that name; otherwise an [`io::Error`] of kind `InvalidData` carrying a
    /// [`PkzipError`].
    pub fn read(&self, name: &str) -> io::Result<Vec<u8>> {
        let i = self
            .entries
            .iter()
            .position(|e| e.name == name.as_bytes())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "no ZIP entry with that name")
            })?;
        self.read_index(i)
    }

    /// The bytes of the entry at central directory index `i`, within the limits.
    ///
    /// # Errors
    ///
    /// `NotFound` when there is no such entry; otherwise an [`io::Error`] of kind `InvalidData` carrying a
    /// [`PkzipError`].
    pub fn read_index(&self, i: usize) -> io::Result<Vec<u8>> {
        let e = self
            .entries
            .get(i)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no ZIP entry at that index"))?;
        match e.method {
            Method::Stored => self.stored_bytes(e).map(<[u8]>::to_vec),
            Method::Deflated => {
                if e.size > self.limits.max_entry {
                    return Err(refuse(PkzipError::EntryTooLarge {
                        declared: e.size,
                        limit: self.limits.max_entry,
                    }));
                }
                let out = read_declared(DeflateDecoder::new(self.raw(e)?), index(e.size)?)?;
                check_crc(e, &out)?;
                Ok(out)
            }
        }
    }

    /// The bytes of the stored entry named `name`, in place (nothing is copied or decoded; the CRC-32 is checked).
    ///
    /// # Errors
    ///
    /// `NotFound` when no entry has that name; `InvalidInput` when the entry is deflated (read it with
    /// [`Archive::read`]); [`PkzipError::CrcMismatch`] inside an `InvalidData` error.
    pub fn in_place(&self, name: &str) -> io::Result<&[u8]> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name.as_bytes())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "no ZIP entry with that name")
            })?;
        if e.method != Method::Stored {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the ZIP entry is deflated: read it with Archive::read",
            ));
        }
        self.stored_bytes(e)
    }

    fn stored_bytes(&self, e: &Entry) -> io::Result<&[u8]> {
        let raw = self.raw(e)?;
        check_crc(e, raw)?;
        Ok(raw)
    }

    /// The entry's bytes in the archive (the range was checked to lie inside it when the archive was opened)
    fn raw(&self, e: &Entry) -> io::Result<&[u8]> {
        let start = index(e.data_offset)?;
        Ok(&self.data.as_ref()[start..start + index(e.compressed_size)?])
    }
}

fn check_crc(e: &Entry, bytes: &[u8]) -> io::Result<()> {
    let mut crc = flate2::Crc::new();
    crc.update(bytes);
    if crc.sum() != e.crc32 {
        return Err(refuse(PkzipError::CrcMismatch {
            expected: e.crc32,
            actual: crc.sum(),
        }));
    }
    Ok(())
}

/// Read `reader` to its end when it produces exactly `declared` bytes: it is asked for at most `declared + 1`
fn read_declared(reader: impl Read, declared: usize) -> io::Result<Vec<u8>> {
    match crate::compression::read_bounded(reader, declared) {
        Ok(out) if out.len() == declared => Ok(out),
        Ok(out) => Err(refuse(PkzipError::ShortOfDeclared {
            declared: declared as u64,
            actual: out.len() as u64,
        })),
        Err(err) if is_too_large(&err) => Err(refuse(PkzipError::ExpandsPastDeclared {
            declared: declared as u64,
        })),
        Err(err) => Err(err),
    }
}

fn is_too_large(err: &io::Error) -> bool {
    err.get_ref()
        .and_then(|e| e.downcast_ref::<crate::compression::DecompressError>())
        .is_some_and(|e| matches!(e, crate::compression::DecompressError::TooLarge { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that keeps producing bytes: the bound has to come from the reader of the entry, not from it
    struct Endless;

    impl Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            buf.fill(7);
            Ok(buf.len())
        }
    }

    #[test]
    fn a_reader_that_never_ends_is_stopped_one_byte_past_the_declared_size() {
        let e = read_declared(Endless, 100).unwrap_err();
        assert_eq!(
            e.get_ref().and_then(|i| i.downcast_ref::<PkzipError>()),
            Some(&PkzipError::ExpandsPastDeclared { declared: 100 })
        );
    }

    #[test]
    fn a_reader_of_exactly_the_declared_size_is_read() {
        assert_eq!(read_declared(&[1u8, 2, 3][..], 3).unwrap(), vec![1, 2, 3]);
        let e = read_declared(&[1u8, 2][..], 3).unwrap_err();
        assert_eq!(
            e.get_ref().and_then(|i| i.downcast_ref::<PkzipError>()),
            Some(&PkzipError::ShortOfDeclared {
                declared: 3,
                actual: 2
            })
        );
    }
}
