//! Writing and reading a container as a stream (files too large to hold in memory)
//!
//! The layout is the one of [`super`]: header, section table, payloads, trailer.
//! The table holds the length and SHA-256 of every payload and comes before the
//! payloads, and the trailer is the SHA-256 of every byte before it, the table
//! included. A streamed writer therefore cannot finish in one pass:
//! [`StreamWriter`] writes the header and a placeholder table, streams the
//! payloads while hashing each, writes the real table, then reads its own output
//! back to compute the trailer (checking every payload against the table on the
//! way, so a payload changed on disk in between is not sealed with a trailer).
//! It needs `Read + Write + Seek` (a file). The bytes are the same as
//! [`Container::to_bytes`] for the same sections.
//!
//! [`StreamReader`] makes the same checks, in the same order, as
//! [`ContainerView::open_with_known`]: it hashes the whole file once for the
//! trailer before anything else is used. A section is then read with
//! [`StreamReader::section`]; its SHA-256 is checked when the section has been
//! read to its end (the file could have changed since it was opened), so its
//! bytes are **unverified** until [`UnverifiedSection::finish`] returns `Ok` (or
//! a read returns the end of the section without an error).
//!
//! [`Container::to_bytes`]: super::Container::to_bytes
//! [`ContainerView::open_with_known`]: super::ContainerView::open_with_known

use std::io::{self, Read, Seek, SeekFrom, Write};

use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use super::{
    check_entries, identifier, parse_header, parse_table, ContainerError, Entry, FormatStatus, Tag,
    ENTRY_LEN, FORMAT_ID, HEADER_LEN, MAGIC, MAJOR, MINOR, TRAILER_LEN,
};

/// Why a streamed container could not be written or read
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamError {
    /// The container's own rules (the same errors as [`super::ContainerView`])
    Container(ContainerError),
    /// Reading, writing or seeking the underlying file failed
    Io(io::Error),
    /// More or fewer sections were written than declared (counting the format
    /// identifier as section 0)
    SectionCount {
        /// Sections declared
        declared: usize,
        /// Sections written so far (or when the writer finished)
        written: usize,
    },
    /// A section writer was dropped before its `finish`
    Unfinished,
}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Container(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "container stream: {e}"),
            Self::SectionCount { declared, written } => write!(
                f,
                "container stream: {written} sections written, {declared} declared"
            ),
            Self::Unfinished => write!(f, "container stream: a section was not finished"),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<io::Error> for StreamError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ContainerError> for StreamError {
    fn from(e: ContainerError) -> Self {
        Self::Container(e)
    }
}

type Result<T> = core::result::Result<T, StreamError>;

/// Writes a container section by section into `inner`.
///
/// The sections are declared up front (the format identifier is added as
/// section 0, as [`super::Container::new`] does), then written in that order.
/// [`StreamWriter::finish`] writes the table and the trailer; a writer abandoned
/// before it ([`StreamWriter::into_unfinished`] or a drop) leaves a file with a
/// zeroed table and no trailer, which no reader opens.
pub struct StreamWriter<W: Read + Write + Seek> {
    inner: W,
    start: u64,
    declared: Vec<(Tag, bool)>,
    /// (length, SHA-256) of every section written so far
    written: Vec<(u64, [u8; 32])>,
    poisoned: bool,
}

impl<W: Read + Write + Seek> StreamWriter<W> {
    /// Writes the header and a placeholder table at the current position of
    /// `inner`, then the format identifier section.
    ///
    /// # Errors
    ///
    /// [`StreamError::Io`].
    pub fn new(mut inner: W, semantics_id: [u8; 32], sections: &[(Tag, bool)]) -> Result<Self> {
        let start = inner.stream_position()?;
        let mut declared = Vec::with_capacity(sections.len() + 1);
        declared.push((Tag::FMID, true));
        declared.extend_from_slice(sections);
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&MAJOR.to_le_bytes());
        header.extend_from_slice(&MINOR.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&semantics_id);
        header.extend_from_slice(&(declared.len() as u64).to_le_bytes());
        inner.write_all(&header)?;
        inner.write_all(&alloc::vec![0u8; ENTRY_LEN * declared.len()])?;
        let mut w = Self {
            inner,
            start,
            declared,
            written: Vec::new(),
            poisoned: false,
        };
        w.write_section(&FORMAT_ID)?;
        Ok(w)
    }

    /// Starts the next declared section; write its payload, then call
    /// [`SectionWriter::finish`].
    ///
    /// # Errors
    ///
    /// [`StreamError::SectionCount`] when every declared section is written,
    /// [`StreamError::Unfinished`] after a section writer was dropped unfinished.
    pub fn begin_section(&mut self) -> Result<SectionWriter<'_, W>> {
        if self.poisoned {
            return Err(StreamError::Unfinished);
        }
        if self.written.len() == self.declared.len() {
            return Err(StreamError::SectionCount {
                declared: self.declared.len(),
                written: self.written.len(),
            });
        }
        Ok(SectionWriter {
            parent: self,
            hasher: Sha256::new(),
            len: 0,
            finished: false,
        })
    }

    /// Writes the next declared section from a slice and returns its SHA-256.
    ///
    /// # Errors
    ///
    /// As [`Self::begin_section`], and [`StreamError::Io`].
    pub fn write_section(&mut self, payload: &[u8]) -> Result<[u8; 32]> {
        let mut s = self.begin_section()?;
        s.write_all(payload)?;
        s.finish()
    }

    /// Writes the table and the trailer and returns `inner`.
    ///
    /// The trailer is computed by reading the written bytes back; every payload
    /// is checked against the SHA-256 recorded while it was written.
    ///
    /// # Errors
    ///
    /// [`StreamError::SectionCount`] when fewer sections were written than
    /// declared, [`StreamError::Unfinished`], [`StreamError::Container`] with
    /// [`ContainerError::SectionHash`] when a payload changed on disk, and
    /// [`StreamError::Io`]. On an error no trailer is written.
    pub fn finish(mut self) -> Result<W> {
        if self.poisoned {
            return Err(StreamError::Unfinished);
        }
        if self.written.len() != self.declared.len() {
            return Err(StreamError::SectionCount {
                declared: self.declared.len(),
                written: self.written.len(),
            });
        }
        let mut table = Vec::with_capacity(ENTRY_LEN * self.declared.len());
        let mut offset = (HEADER_LEN + ENTRY_LEN * self.declared.len()) as u64;
        for ((tag, critical), (len, sha)) in self.declared.iter().zip(&self.written) {
            table.extend_from_slice(&tag.0);
            table.extend_from_slice(&u16::from(*critical).to_le_bytes());
            table.extend_from_slice(&0u16.to_le_bytes());
            table.extend_from_slice(&offset.to_le_bytes());
            table.extend_from_slice(&len.to_le_bytes());
            table.extend_from_slice(sha);
            offset += len;
        }
        self.inner
            .seek(SeekFrom::Start(self.start + HEADER_LEN as u64))?;
        self.inner.write_all(&table)?;

        // second pass: header and table, then each payload checked against its recorded hash
        self.inner.seek(SeekFrom::Start(self.start))?;
        let mut trailer = Sha256::new();
        let mut head = alloc::vec![0u8; HEADER_LEN + table.len()];
        self.inner.read_exact(&mut head)?;
        trailer.update(&head);
        let mut buf = alloc::vec![0u8; 64 * 1024];
        for (index, (len, sha)) in self.written.iter().enumerate() {
            let mut section = Sha256::new();
            let mut left = *len;
            while left > 0 {
                let n = usize::try_from(left).map_or(buf.len(), |l| l.min(buf.len()));
                self.inner.read_exact(&mut buf[..n])?;
                section.update(&buf[..n]);
                trailer.update(&buf[..n]);
                left -= n as u64;
            }
            if <[u8; 32]>::from(section.finalize()) != *sha {
                return Err(ContainerError::SectionHash { index }.into());
            }
        }
        let end = self.inner.stream_position()?;
        if end != offset + self.start {
            return Err(ContainerError::Layout.into());
        }
        self.inner.write_all(&trailer.finalize())?;
        self.inner.flush()?;
        Ok(self.inner)
    }

    /// Returns `inner` without the table and trailer (the file does not open)
    pub fn into_unfinished(self) -> W {
        self.inner
    }
}

/// Streams one payload into a [`StreamWriter`]; call [`Self::finish`]
pub struct SectionWriter<'a, W: Read + Write + Seek> {
    parent: &'a mut StreamWriter<W>,
    hasher: Sha256,
    len: u64,
    finished: bool,
}

impl<W: Read + Write + Seek> SectionWriter<'_, W> {
    /// Ends the section and returns the SHA-256 of its payload.
    ///
    /// # Errors
    ///
    /// [`StreamError::Io`] when the underlying writer cannot be flushed.
    pub fn finish(mut self) -> Result<[u8; 32]> {
        self.finished = true;
        let sha: [u8; 32] = core::mem::take(&mut self.hasher).finalize().into();
        self.parent.written.push((self.len, sha));
        Ok(sha)
    }
}

impl<W: Read + Write + Seek> Write for SectionWriter<'_, W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let n = self.parent.inner.write(data)?;
        self.hasher.update(&data[..n]);
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.parent.inner.flush()
    }
}

impl<W: Read + Write + Seek> Drop for SectionWriter<'_, W> {
    fn drop(&mut self) {
        if !self.finished {
            self.parent.poisoned = true;
        }
    }
}

/// Reads a container from a file without loading its payloads.
pub struct StreamReader<R: Read + Seek> {
    inner: R,
    start: u64,
    minor: u16,
    semantics_id: [u8; 32],
    format: FormatStatus,
    head: Vec<u8>,
    entries: Vec<Entry>,
}

fn container_len(inner: &mut (impl Read + Seek), start: u64) -> Result<usize> {
    let end = inner.seek(SeekFrom::End(0))?;
    usize::try_from(end - start).map_err(|_| ContainerError::Layout.into())
}

impl<R: Read + Seek> StreamReader<R> {
    /// Opens the container that starts at the current position of `inner` and
    /// runs to its end, with the checks of [`super::ContainerView::open_with_known`].
    ///
    /// # Errors
    ///
    /// [`StreamError::Container`] with the error the view would return, and
    /// [`StreamError::Io`].
    pub fn open_with_known(mut inner: R, known: &[Tag]) -> Result<Self> {
        let start = inner.stream_position()?;
        let total = container_len(&mut inner, start)?;
        if total < HEADER_LEN + TRAILER_LEN {
            return Err(ContainerError::TooShort.into());
        }
        inner.seek(SeekFrom::Start(start))?;
        let mut header = alloc::vec![0u8; HEADER_LEN];
        inner.read_exact(&mut header)?;
        let head = parse_header(&header, total)?;
        let mut table = alloc::vec![0u8; head.count * ENTRY_LEN];
        inner.read_exact(&mut table)?;
        let entries = parse_table(&table, head.count, total)?;

        // the trailer: every byte before it
        inner.seek(SeekFrom::Start(start))?;
        let body_end = total - TRAILER_LEN;
        let mut hasher = Sha256::new();
        copy_hashing(&mut inner, body_end as u64, |chunk| hasher.update(chunk))?;
        let mut trailer = [0u8; TRAILER_LEN];
        inner.read_exact(&mut trailer)?;
        if <[u8; 32]>::from(hasher.finalize()) != trailer {
            return Err(ContainerError::Trailer.into());
        }

        // section 0: its hash, and its payload when it is an identifier of the identifier's length
        let first = match entries.first() {
            Some(e) => {
                inner.seek(SeekFrom::Start(start + e.offset as u64))?;
                let mut payload = Vec::new();
                let mut section = Sha256::new();
                let keep = e.tag == Tag::FMID && e.len == FORMAT_ID.len();
                copy_hashing(&mut inner, e.len as u64, |chunk| {
                    section.update(chunk);
                    if keep {
                        payload.extend_from_slice(chunk);
                    }
                })?;
                Some((<[u8; 32]>::from(section.finalize()) == e.sha, payload))
            }
            None => None,
        };
        let format = check_entries(
            &entries,
            known,
            first.as_ref().map(|(ok, p)| (*ok, p.as_slice())),
        )?;
        let mut all = header;
        all.extend_from_slice(&table);
        Ok(Self {
            inner,
            start,
            minor: head.minor,
            semantics_id: head.semantics_id,
            format,
            head: all,
            entries,
        })
    }

    /// [`Self::open_with_known`], refusing a container without the format
    /// identifier ([`FormatStatus::Legacy`]).
    ///
    /// # Errors
    ///
    /// As [`Self::open_with_known`], and [`ContainerError::LegacyFormat`].
    pub fn open_strict_with_known(inner: R, known: &[Tag]) -> Result<Self> {
        let r = Self::open_with_known(inner, known)?;
        if r.format == FormatStatus::Legacy {
            return Err(ContainerError::LegacyFormat.into());
        }
        Ok(r)
    }

    /// Number of sections
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the container has no sections
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How the container identifies its format
    #[must_use]
    pub const fn format_status(&self) -> FormatStatus {
        self.format
    }

    /// Minor version
    #[must_use]
    pub const fn minor(&self) -> u16 {
        self.minor
    }

    /// Semantics identifier of the writer
    #[must_use]
    pub const fn semantics_id(&self) -> [u8; 32] {
        self.semantics_id
    }

    /// The container's identifier (as [`super::ContainerView::id`])
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        identifier(&self.head[..HEADER_LEN], &self.head[HEADER_LEN..])
    }

    /// Tag of section `index`
    #[must_use]
    pub fn tag(&self, index: usize) -> Tag {
        self.entries[index].tag
    }

    /// Whether section `index` is critical
    #[must_use]
    pub fn critical(&self, index: usize) -> bool {
        self.entries[index].critical
    }

    /// Length of section `index`
    #[must_use]
    pub fn section_len(&self, index: usize) -> u64 {
        self.entries[index].len as u64
    }

    /// SHA-256 of section `index` as recorded in the table
    #[must_use]
    pub fn sha256(&self, index: usize) -> [u8; 32] {
        self.entries[index].sha
    }

    /// Reads section `index`; its bytes are unverified until the section has
    /// been read to its end without an error.
    ///
    /// # Errors
    ///
    /// [`StreamError::Io`] when seeking fails.
    pub fn section(&mut self, index: usize) -> Result<UnverifiedSection<'_, R>> {
        let e = self.entries[index];
        self.inner
            .seek(SeekFrom::Start(self.start + e.offset as u64))?;
        Ok(UnverifiedSection {
            inner: &mut self.inner,
            index,
            left: e.len as u64,
            expected: e.sha,
            hasher: Sha256::new(),
            checked: false,
        })
    }
}

/// Reads `len` bytes from `inner`, handing each piece to `f`
fn copy_hashing(inner: &mut impl Read, len: u64, mut f: impl FnMut(&[u8])) -> Result<()> {
    let mut buf = alloc::vec![0u8; 64 * 1024];
    let mut left = len;
    while left > 0 {
        let n = usize::try_from(left).map_or(buf.len(), |l| l.min(buf.len()));
        inner.read_exact(&mut buf[..n])?;
        f(&buf[..n]);
        left -= n as u64;
    }
    Ok(())
}

/// One section of a [`StreamReader`], read as it is.
///
/// ⚠️ The bytes are **unverified** until the end of the section: the SHA-256
/// recorded in the table is compared when the last byte has been read, and a
/// mismatch is returned as an error from that read (and from [`Self::finish`]).
pub struct UnverifiedSection<'a, R: Read + Seek> {
    inner: &'a mut R,
    index: usize,
    left: u64,
    expected: [u8; 32],
    hasher: Sha256,
    checked: bool,
}

impl<R: Read + Seek> UnverifiedSection<'_, R> {
    fn check(&mut self) -> core::result::Result<(), ContainerError> {
        if !self.checked {
            self.checked = true;
            let got: [u8; 32] = core::mem::take(&mut self.hasher).finalize().into();
            if got != self.expected {
                return Err(ContainerError::SectionHash { index: self.index });
            }
        }
        Ok(())
    }

    /// Reads the rest of the section and checks its SHA-256.
    ///
    /// # Errors
    ///
    /// [`ContainerError::SectionHash`] when the section does not match the table,
    /// and [`StreamError::Io`].
    pub fn finish(mut self) -> Result<()> {
        let mut sink = alloc::vec![0u8; 64 * 1024];
        while self.left > 0 {
            let n = self.read(&mut sink)?;
            if n == 0 {
                break;
            }
        }
        self.check()?;
        Ok(())
    }
}

impl<R: Read + Seek> Read for UnverifiedSection<'_, R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.left == 0 {
            self.check()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            return Ok(0);
        }
        let want = usize::try_from(self.left).map_or(out.len(), |l| l.min(out.len()));
        let n = self.inner.read(&mut out[..want])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the file ended inside a section",
            ));
        }
        self.hasher.update(&out[..n]);
        self.left -= n as u64;
        if self.left == 0 {
            self.check()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }
        Ok(n)
    }
}
