//! A container that holds several payloads, each identified by its SHA-256
//!
//! A container is an envelope: every section carries the bytes of an existing
//! format unchanged (a world snapshot, a shape file, a law record, a residual
//! stream …) and that format keeps its own version. The container fixes only
//! the header, the section table, the integrity of the whole file and how
//! sections refer to each other.
//!
//! # Layout
//!
//! All integers are little endian.
//!
//! | range | content |
//! |---|---|
//! | `[0..8)` | [`MAGIC`] `89 41 4C 43 0D 0A 1A 0A` |
//! | `[8..10)` | major version `u16` = [`MAJOR`] |
//! | `[10..12)` | minor version `u16` |
//! | `[12..16)` | reserved `u32`, must be 0 |
//! | `[16..48)` | semantics id: the arithmetic the writer evaluated with |
//! | `[48..56)` | section count `u64` |
//! | `56 + 56·i` | section entry `i`: tag (4 bytes as stored), flags `u16` (bit 0 = critical, other bits must be 0), reserved `u16` = 0, offset `u64`, length `u64`, SHA-256 of the payload (32 bytes) |
//! | after the table | the payloads in table order, with no gap between them |
//! | last 32 bytes | trailer: SHA-256 of every preceding byte |
//!
//! The first byte of the magic is not ASCII, so no ASCII magic of another
//! format can be a prefix of it, and a transfer that rewrites line endings
//! or drops the eighth bit changes the magic.
//!
//! # Identifier and trailer
//!
//! The two hashes answer different questions:
//!
//! * [`Container::id`] is the identity of the content: SHA-256 over the
//!   domain [`ID_DOMAIN`], the header and the section table, each prefixed
//!   with its length as a big-endian `u64` (the same encoding as
//!   [`crate::law::SignalLaw::law_id`]). The table holds the SHA-256 of every
//!   payload, so the identifier covers every byte of content. The trailer is
//!   not part of it.
//! * The trailer checks that the file arrived intact. Readers always verify
//!   it; there is no way to read a container without doing so.
//!
//! # Versions and unknown sections
//!
//! A reader refuses a major version other than [`MAJOR`]. It reads any minor
//! version. A section whose tag it does not know is refused if the section
//! is marked critical, and kept unchanged otherwise, so writing back what was
//! read reproduces the same bytes.
//!
//! # References between sections
//!
//! A [`Tag::SREF`] section lists, per referring item, the SHA-256 of the
//! section it refers to, so the same payload is stored once however many
//! items use it, and replacing a payload changes the identifier. A
//! [`Tag::LIDS`] section lists, per section that holds a law, the law's
//! identifier and the semantics id it was computed under. A reference to a
//! section that is not in the container is refused.
//!
//! # Files written before this format
//!
//! [`read_any`] also reads the earlier `ALICE_ZIP` file header (65 or 66
//! bytes) and returns it as a container with two sections: the header bytes
//! as [`Tag::PROV`] and the payload as [`Tag::RAW`]. Only the versions that
//! were written (1.0 and 1.1) and the field values that exist are accepted.

use alloc::vec::Vec;
use core::fmt;

use sha2::{Digest, Sha256};

/// First 8 bytes of every container.
pub const MAGIC: [u8; 8] = [0x89, b'A', b'L', b'C', 0x0D, 0x0A, 0x1A, 0x0A];
/// Major version this crate writes and reads.
pub const MAJOR: u16 = 1;
/// Minor version this crate writes.
pub const MINOR: u16 = 0;
/// Domain separation for [`Container::id`].
pub const ID_DOMAIN: &[u8] = b"alice/container/v1";
/// Length of the header.
pub const HEADER_LEN: usize = 56;
/// Length of one section entry in the table.
pub const ENTRY_LEN: usize = 56;
/// Length of the trailer.
pub const TRAILER_LEN: usize = 32;

/// Section kind: four bytes stored as they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub [u8; 4]);

impl Tag {
    /// Raw data.
    pub const RAW: Self = Self(*b"RAW_");
    /// Residual against a law.
    pub const RESD: Self = Self(*b"RESD");
    /// A law.
    pub const LAW: Self = Self(*b"LAW_");
    /// Verdict history of a law.
    pub const VRDT: Self = Self(*b"VRDT");
    /// Shape.
    pub const SHAP: Self = Self(*b"SHAP");
    /// Physical state.
    pub const PHYS: Self = Self(*b"PHYS");
    /// Asset.
    pub const ASET: Self = Self(*b"ASET");
    /// Input sequence or replay.
    pub const SIMS: Self = Self(*b"SIMS");
    /// Provenance.
    pub const PROV: Self = Self(*b"PROV");
    /// Reference values a reader can check the other sections against.
    pub const ORCL: Self = Self(*b"ORCL");
    /// References from items to sections, by SHA-256.
    pub const SREF: Self = Self(*b"SREF");
    /// Law identifiers of the sections that hold a law.
    pub const LIDS: Self = Self(*b"LIDS");

    /// The tags this version of the format defines.
    pub const KNOWN: [Self; 12] = [
        Self::RAW,
        Self::RESD,
        Self::LAW,
        Self::VRDT,
        Self::SHAP,
        Self::PHYS,
        Self::ASET,
        Self::SIMS,
        Self::PROV,
        Self::ORCL,
        Self::SREF,
        Self::LIDS,
    ];

    /// Whether this version of the format defines the tag.
    #[must_use]
    pub fn is_known(self) -> bool {
        Self::KNOWN.contains(&self)
    }
}

/// One section: a tag, whether a reader must understand it, and the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Kind of the payload.
    pub tag: Tag,
    /// A reader that does not know `tag` refuses the container when set.
    pub critical: bool,
    /// The bytes of the format the section holds.
    pub payload: Vec<u8>,
}

/// A container held in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Container {
    /// Minor version (written as is; [`MINOR`] for a new container).
    pub minor: u16,
    /// Arithmetic the writer evaluated with.
    pub semantics_id: [u8; 32],
    /// Sections in table order.
    pub sections: Vec<Section>,
}

/// One row of a [`Tag::SREF`] section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// SHA-256 of the payload referred to.
    pub target: [u8; 32],
    /// Tag the referred section must have.
    pub kind: Tag,
    /// How the referring side turns the payload into its object.
    pub builder: Vec<u8>,
    /// Index of the referred section in the table.
    pub section: usize,
}

/// One row of a [`Tag::LIDS`] section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LawIdEntry {
    /// Index of the section that holds the law.
    pub section: usize,
    /// The law's identifier.
    pub law_id: [u8; 32],
    /// The semantics id the identifier was computed under.
    pub semantics_id: [u8; 32],
}

/// Why a container could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContainerError {
    /// Fewer bytes than the header, the table and the trailer need.
    TooShort,
    /// The first bytes are not a format this module reads.
    BadMagic,
    /// A major version other than [`MAJOR`].
    UnsupportedMajor {
        /// The major version found.
        major: u16,
    },
    /// A reserved field or flag bit is not 0.
    Reserved,
    /// Offsets and lengths do not tile the file: a gap, an overlap, a range
    /// past the end, extra bytes, or a count larger than the file can hold.
    Layout,
    /// The trailer does not match the bytes before it.
    Trailer,
    /// A payload does not match the SHA-256 in its table entry.
    SectionHash {
        /// Index of the section.
        index: usize,
    },
    /// A section marked critical has a tag this version does not define.
    UnknownCritical {
        /// The tag.
        tag: Tag,
    },
    /// A [`Tag::SREF`] or [`Tag::LIDS`] payload is not well formed.
    Malformed {
        /// Tag of the section.
        tag: Tag,
    },
    /// A reference names a payload that is not in the container, or a
    /// section with another tag than the reference states.
    DanglingSection {
        /// SHA-256 the reference names.
        target: [u8; 32],
    },
    /// A reference names a builder the caller does not accept.
    UnknownBuilder {
        /// Row of the reference.
        row: usize,
    },
    /// A law identifier was computed under other semantics than the header
    /// states.
    SemanticsMismatch {
        /// Semantics id in the header.
        container: [u8; 32],
        /// Semantics id in the [`Tag::LIDS`] row.
        section: [u8; 32],
    },
    /// A law identifier differs from the one recomputed from the section.
    LawIdMismatch {
        /// Index of the section that holds the law.
        section: usize,
    },
    /// An `ALICE_ZIP` header with a version that was never written.
    LegacyVersion {
        /// Major version found.
        major: u8,
        /// Minor version found.
        minor: u8,
    },
    /// An `ALICE_ZIP` header field holds a value that was never written.
    LegacyField {
        /// Name of the field.
        field: &'static str,
        /// The value found.
        value: u8,
    },
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "container: too short"),
            Self::BadMagic => write!(f, "container: unrecognised magic"),
            Self::UnsupportedMajor { major } => {
                write!(f, "container: major version {major} is not supported")
            }
            Self::Reserved => write!(f, "container: a reserved field or flag bit is set"),
            Self::Layout => write!(f, "container: offsets and lengths do not tile the file"),
            Self::Trailer => write!(f, "container: trailer does not match the contents"),
            Self::SectionHash { index } => {
                write!(f, "container: section {index} does not match its SHA-256")
            }
            Self::UnknownCritical { tag } => {
                write!(f, "container: unknown critical section {tag:?}")
            }
            Self::Malformed { tag } => write!(f, "container: malformed {tag:?} section"),
            Self::DanglingSection { .. } => {
                write!(
                    f,
                    "container: a reference names a section that is not present"
                )
            }
            Self::UnknownBuilder { row } => {
                write!(f, "container: reference {row} names an unaccepted builder")
            }
            Self::SemanticsMismatch { .. } => write!(
                f,
                "container: a law identifier was computed under other semantics than the header"
            ),
            Self::LawIdMismatch { section } => write!(
                f,
                "container: the law identifier of section {section} differs from the recomputed one"
            ),
            Self::LegacyVersion { major, minor } => {
                write!(f, "ALICE_ZIP: version {major}.{minor} was never written")
            }
            Self::LegacyField { field, value } => {
                write!(f, "ALICE_ZIP: {field} = {value:#04x} was never written")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContainerError {}

/// Bytes of one row of a [`Tag::SREF`] section before its builder name.
const SREF_ROW_MIN: usize = 32 + 4 + 2;
/// Bytes of one row of a [`Tag::LIDS`] section.
const LIDS_ROW: usize = 8 + 32 + 32;
/// Header length of an `ALICE_ZIP` file of version 1.0.
const LEGACY_V1_LEN: usize = 65;
/// Header length of an `ALICE_ZIP` file of version 1.1.
const LEGACY_V2_LEN: usize = 66;
/// Magic of an `ALICE_ZIP` file.
const LEGACY_MAGIC: &[u8; 9] = b"ALICE_ZIP";
/// `payload_type` values the `ALICE_ZIP` writer defines.
const LEGACY_PAYLOAD_TYPES: [u8; 6] = [0x00, 0x10, 0x11, 0x12, 0x20, 0x30];
/// Compression engines the `ALICE_ZIP` writer indexes (`0..=3`).
const LEGACY_ENGINES: u8 = 4;

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn array32(b: &[u8], at: usize) -> [u8; 32] {
    let mut a = [0u8; 32];
    a.copy_from_slice(&b[at..at + 32]);
    a
}

/// `len(x)` as the big-endian `u64` that prefixes `x` in [`identifier`].
fn be_len(x: &[u8]) -> [u8; 8] {
    (x.len() as u64).to_be_bytes()
}

/// Identity over the header and the table (see the module documentation).
fn identifier(header: &[u8], table: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(be_len(ID_DOMAIN));
    h.update(ID_DOMAIN);
    h.update(be_len(header));
    h.update(header);
    h.update(be_len(table));
    h.update(table);
    h.finalize().into()
}

/// A section entry as read from the table.
#[derive(Debug, Clone, Copy)]
struct Entry {
    tag: Tag,
    critical: bool,
    offset: usize,
    len: usize,
    sha: [u8; 32],
}

/// A container read from bytes without copying them.
///
/// [`ContainerView::open`] checks the header, the table and the trailer.
/// A payload's SHA-256 is checked when the payload is asked for.
#[derive(Debug, Clone)]
pub struct ContainerView<'a> {
    bytes: &'a [u8],
    minor: u16,
    semantics_id: [u8; 32],
    entries: Vec<Entry>,
}

/// What the first bytes of a file are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
    /// A container of this module.
    Container,
    /// An `ALICE_ZIP` file written before the container existed.
    LegacyAliceZip,
    /// Neither.
    Unknown,
}

impl Container {
    /// An empty container written with semantics `semantics_id`.
    #[must_use]
    pub fn new(semantics_id: [u8; 32]) -> Self {
        Self {
            minor: MINOR,
            semantics_id,
            sections: Vec::new(),
        }
    }

    /// Appends a section and returns the SHA-256 of its payload.
    pub fn push(&mut self, tag: Tag, critical: bool, payload: Vec<u8>) -> [u8; 32] {
        let sha = sha256(&payload);
        self.sections.push(Section {
            tag,
            critical,
            payload,
        });
        sha
    }

    /// The header and the section table.
    fn header_and_table(&self) -> (Vec<u8>, Vec<u8>) {
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&MAJOR.to_le_bytes());
        header.extend_from_slice(&self.minor.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&self.semantics_id);
        header.extend_from_slice(&(self.sections.len() as u64).to_le_bytes());
        let mut table = Vec::with_capacity(ENTRY_LEN * self.sections.len());
        let mut offset = (HEADER_LEN + ENTRY_LEN * self.sections.len()) as u64;
        for s in &self.sections {
            table.extend_from_slice(&s.tag.0);
            table.extend_from_slice(&u16::from(s.critical).to_le_bytes());
            table.extend_from_slice(&0u16.to_le_bytes());
            table.extend_from_slice(&offset.to_le_bytes());
            table.extend_from_slice(&(s.payload.len() as u64).to_le_bytes());
            table.extend_from_slice(&sha256(&s.payload));
            offset += s.payload.len() as u64;
        }
        (header, table)
    }

    /// The container's bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let (header, table) = self.header_and_table();
        let body: usize = self.sections.iter().map(|s| s.payload.len()).sum();
        let mut out = Vec::with_capacity(header.len() + table.len() + body + TRAILER_LEN);
        out.extend_from_slice(&header);
        out.extend_from_slice(&table);
        for s in &self.sections {
            out.extend_from_slice(&s.payload);
        }
        let trailer = sha256(&out);
        out.extend_from_slice(&trailer);
        out
    }

    /// Identity of the content (see the module documentation).
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        let (header, table) = self.header_and_table();
        identifier(&header, &table)
    }

    /// Reads a container and checks everything: the trailer, every payload,
    /// unknown critical sections, the references of [`Tag::SREF`] and the
    /// semantics of [`Tag::LIDS`].
    ///
    /// # Errors
    ///
    /// The first problem found, as a [`ContainerError`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContainerError> {
        let view = ContainerView::open(bytes)?;
        let mut sections = Vec::with_capacity(view.len());
        for i in 0..view.len() {
            sections.push(Section {
                tag: view.tag(i),
                critical: view.critical(i),
                payload: view.section(i)?.to_vec(),
            });
        }
        view.references()?;
        view.law_ids()?;
        Ok(Self {
            minor: view.minor,
            semantics_id: view.semantics_id,
            sections,
        })
    }
}

impl<'a> ContainerView<'a> {
    /// Checks the header, the table and the trailer.
    ///
    /// The checks run in this order, and the first that fails is returned:
    /// length, magic, major version, reserved fields and flag bits, layout,
    /// trailer, unknown critical sections.
    ///
    /// # Errors
    ///
    /// [`ContainerError::TooShort`], [`ContainerError::BadMagic`],
    /// [`ContainerError::UnsupportedMajor`], [`ContainerError::Reserved`],
    /// [`ContainerError::Layout`], [`ContainerError::Trailer`] or
    /// [`ContainerError::UnknownCritical`].
    pub fn open(bytes: &'a [u8]) -> Result<Self, ContainerError> {
        if bytes.len() < HEADER_LEN + TRAILER_LEN {
            return Err(ContainerError::TooShort);
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(ContainerError::BadMagic);
        }
        let major = u16_at(bytes, 8);
        if major != MAJOR {
            return Err(ContainerError::UnsupportedMajor { major });
        }
        if u32_at(bytes, 12) != 0 {
            return Err(ContainerError::Reserved);
        }
        let minor = u16_at(bytes, 10);
        let semantics_id = array32(bytes, 16);
        let body_end = bytes.len() - TRAILER_LEN;
        // checked before anything is allocated for the table
        let count = usize::try_from(u64_at(bytes, 48))
            .ok()
            .filter(|&c| {
                c.checked_mul(ENTRY_LEN)
                    .is_some_and(|t| t <= body_end - HEADER_LEN)
            })
            .ok_or(ContainerError::Layout)?;
        let mut cursor = HEADER_LEN + count * ENTRY_LEN;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let at = HEADER_LEN + i * ENTRY_LEN;
            let flags = u16_at(bytes, at + 4);
            if flags & !1 != 0 || u16_at(bytes, at + 6) != 0 {
                return Err(ContainerError::Reserved);
            }
            let offset = u64_at(bytes, at + 8);
            let len = u64_at(bytes, at + 16);
            if offset != cursor as u64 {
                return Err(ContainerError::Layout);
            }
            let len = usize::try_from(len)
                .ok()
                .filter(|&l| cursor.checked_add(l).is_some_and(|end| end <= body_end))
                .ok_or(ContainerError::Layout)?;
            let mut tag = [0u8; 4];
            tag.copy_from_slice(&bytes[at..at + 4]);
            entries.push(Entry {
                tag: Tag(tag),
                critical: flags & 1 == 1,
                offset: cursor,
                len,
                sha: array32(bytes, at + 24),
            });
            cursor += len;
        }
        if cursor != body_end {
            return Err(ContainerError::Layout);
        }
        if sha256(&bytes[..body_end]) != bytes[body_end..] {
            return Err(ContainerError::Trailer);
        }
        if let Some(e) = entries.iter().find(|e| e.critical && !e.tag.is_known()) {
            return Err(ContainerError::UnknownCritical { tag: e.tag });
        }
        Ok(Self {
            bytes,
            minor,
            semantics_id,
            entries,
        })
    }

    /// Number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the container has no section.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Minor version.
    #[must_use]
    pub fn minor(&self) -> u16 {
        self.minor
    }

    /// Semantics id in the header.
    #[must_use]
    pub fn semantics_id(&self) -> [u8; 32] {
        self.semantics_id
    }

    /// Identity of the content (see the module documentation).
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        let table_end = HEADER_LEN + ENTRY_LEN * self.entries.len();
        identifier(
            &self.bytes[..HEADER_LEN],
            &self.bytes[HEADER_LEN..table_end],
        )
    }

    /// Tag of section `index`.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn tag(&self, index: usize) -> Tag {
        self.entries[index].tag
    }

    /// Whether section `index` is critical.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn critical(&self, index: usize) -> bool {
        self.entries[index].critical
    }

    /// SHA-256 of section `index` as the table states it.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn sha256(&self, index: usize) -> [u8; 32] {
        self.entries[index].sha
    }

    /// Payload of section `index`, after checking its SHA-256.
    ///
    /// # Errors
    ///
    /// [`ContainerError::SectionHash`] if the payload does not match.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    pub fn section(&self, index: usize) -> Result<&'a [u8], ContainerError> {
        let e = self.entries[index];
        let payload = &self.bytes[e.offset..e.offset + e.len];
        if sha256(payload) != e.sha {
            return Err(ContainerError::SectionHash { index });
        }
        Ok(payload)
    }

    /// Index of the first section whose table entry states `sha`.
    #[must_use]
    pub fn find_sha(&self, sha: &[u8; 32]) -> Option<usize> {
        self.entries.iter().position(|e| &e.sha == sha)
    }

    /// Indices of the sections tagged `tag`.
    fn indices_of(&self, tag: Tag) -> impl Iterator<Item = usize> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter(move |(_, e)| e.tag == tag)
            .map(|(i, _)| i)
    }

    /// Rows of every [`Tag::SREF`] section, each resolved to a section.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`], [`ContainerError::SectionHash`] or
    /// [`ContainerError::DanglingSection`].
    pub fn references(&self) -> Result<Vec<Reference>, ContainerError> {
        let malformed = ContainerError::Malformed { tag: Tag::SREF };
        let mut rows = Vec::new();
        for index in self.indices_of(Tag::SREF) {
            let p = self.section(index)?;
            if p.len() < 8 {
                return Err(malformed);
            }
            let count = usize::try_from(u64_at(p, 0))
                .ok()
                .filter(|&c| {
                    c.checked_mul(SREF_ROW_MIN)
                        .is_some_and(|n| n <= p.len() - 8)
                })
                .ok_or(malformed)?;
            let mut at = 8;
            for _ in 0..count {
                if p.len() - at < SREF_ROW_MIN {
                    return Err(malformed);
                }
                let target = array32(p, at);
                let mut kind = [0u8; 4];
                kind.copy_from_slice(&p[at + 32..at + 36]);
                let builder_len = usize::from(u16_at(p, at + 36));
                at += SREF_ROW_MIN;
                if p.len() - at < builder_len {
                    return Err(malformed);
                }
                rows.push((target, Tag(kind), p[at..at + builder_len].to_vec()));
                at += builder_len;
            }
            if at != p.len() {
                return Err(malformed);
            }
        }
        // the whole table is parsed before any row is resolved, so a
        // malformed table is reported as such and not as a missing section
        let mut out = Vec::with_capacity(rows.len());
        for (target, kind, builder) in rows {
            let section = self
                .entries
                .iter()
                .position(|e| e.sha == target && e.tag == kind)
                .ok_or(ContainerError::DanglingSection { target })?;
            self.section(section)?;
            out.push(Reference {
                target,
                kind,
                builder,
                section,
            });
        }
        Ok(out)
    }

    /// [`Self::references`], refusing a builder not in `known`.
    ///
    /// # Errors
    ///
    /// As [`Self::references`], and [`ContainerError::UnknownBuilder`].
    pub fn references_with_builders(
        &self,
        known: &[&[u8]],
    ) -> Result<Vec<Reference>, ContainerError> {
        let refs = self.references()?;
        if let Some(row) = refs
            .iter()
            .position(|r| !known.contains(&r.builder.as_slice()))
        {
            return Err(ContainerError::UnknownBuilder { row });
        }
        Ok(refs)
    }

    /// Rows of every [`Tag::LIDS`] section. A row computed under other
    /// semantics than the header states is refused.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`], [`ContainerError::SectionHash`] or
    /// [`ContainerError::SemanticsMismatch`].
    pub fn law_ids(&self) -> Result<Vec<LawIdEntry>, ContainerError> {
        let rows = self.law_ids_unchecked_semantics()?;
        if let Some(r) = rows.iter().find(|r| r.semantics_id != self.semantics_id) {
            return Err(ContainerError::SemanticsMismatch {
                container: self.semantics_id,
                section: r.semantics_id,
            });
        }
        Ok(rows)
    }

    /// Rows of every [`Tag::LIDS`] section without comparing their
    /// semantics id with the header; the caller sees both in each row.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`] or [`ContainerError::SectionHash`].
    pub fn law_ids_unchecked_semantics(&self) -> Result<Vec<LawIdEntry>, ContainerError> {
        let malformed = ContainerError::Malformed { tag: Tag::LIDS };
        let mut out = Vec::new();
        for index in self.indices_of(Tag::LIDS) {
            let p = self.section(index)?;
            if p.len() < 8 {
                return Err(malformed);
            }
            let count = usize::try_from(u64_at(p, 0))
                .ok()
                .filter(|&c| c.checked_mul(LIDS_ROW) == Some(p.len() - 8))
                .ok_or(malformed)?;
            for row in 0..count {
                let at = 8 + row * LIDS_ROW;
                let section = usize::try_from(u64_at(p, at))
                    .ok()
                    .filter(|&s| s < self.entries.len())
                    .ok_or(malformed)?;
                out.push(LawIdEntry {
                    section,
                    law_id: array32(p, at + 8),
                    semantics_id: array32(p, at + 40),
                });
            }
        }
        Ok(out)
    }

    /// Compares every [`Tag::LIDS`] row with the identifier `recompute`
    /// derives from the section's payload. `recompute` returns `None` for a
    /// payload it cannot interpret, which is refused as a mismatch.
    ///
    /// # Errors
    ///
    /// As [`Self::law_ids`], and [`ContainerError::LawIdMismatch`].
    pub fn verify_law_ids<F>(&self, mut recompute: F) -> Result<(), ContainerError>
    where
        F: FnMut(usize, &[u8], &[u8; 32]) -> Option<[u8; 32]>,
    {
        for row in self.law_ids()? {
            let payload = self.section(row.section)?;
            if recompute(row.section, payload, &row.semantics_id) != Some(row.law_id) {
                return Err(ContainerError::LawIdMismatch {
                    section: row.section,
                });
            }
        }
        Ok(())
    }
}

/// What the first bytes of `bytes` are.
#[must_use]
pub fn detect(bytes: &[u8]) -> Format {
    if bytes.starts_with(&MAGIC) {
        Format::Container
    } else if bytes.starts_with(LEGACY_MAGIC) {
        Format::LegacyAliceZip
    } else {
        Format::Unknown
    }
}

/// Reads an `ALICE_ZIP` file as a container: the header bytes as
/// [`Tag::PROV`] and the payload as [`Tag::RAW`], with an all-zero semantics
/// id.
///
/// Only what the writer produced is accepted: version 1.0 (65-byte header)
/// or 1.1 (66-byte header), a `file_type` of 1 to 5, an engine index of 0 to
/// 3, a defined `payload_type`, and a payload exactly as long as the header
/// states. The header's `original_hash` is the hash of the decompressed
/// data, so it is checked where the payload is decompressed, not here.
///
/// # Errors
///
/// [`ContainerError::TooShort`], [`ContainerError::BadMagic`],
/// [`ContainerError::LegacyVersion`], [`ContainerError::LegacyField`] or
/// [`ContainerError::Layout`] (the payload length differs from the length
/// the header states).
pub fn read_legacy_alice_zip(bytes: &[u8]) -> Result<Container, ContainerError> {
    if bytes.len() < LEGACY_V1_LEN {
        return Err(ContainerError::TooShort);
    }
    if !bytes.starts_with(LEGACY_MAGIC) {
        return Err(ContainerError::BadMagic);
    }
    let (major, minor) = (bytes[9], bytes[10]);
    let header_len = match (major, minor) {
        (1, 0) => LEGACY_V1_LEN,
        (1, 1) => LEGACY_V2_LEN,
        _ => return Err(ContainerError::LegacyVersion { major, minor }),
    };
    if bytes.len() < header_len {
        return Err(ContainerError::TooShort);
    }
    if !(1..=5).contains(&bytes[11]) {
        return Err(ContainerError::LegacyField {
            field: "file_type",
            value: bytes[11],
        });
    }
    if bytes[12] >= LEGACY_ENGINES {
        return Err(ContainerError::LegacyField {
            field: "engine",
            value: bytes[12],
        });
    }
    let size_at = if header_len == LEGACY_V2_LEN {
        if !LEGACY_PAYLOAD_TYPES.contains(&bytes[13]) {
            return Err(ContainerError::LegacyField {
                field: "payload_type",
                value: bytes[13],
            });
        }
        22
    } else {
        21
    };
    let payload = &bytes[header_len..];
    if u64_at(bytes, size_at) != payload.len() as u64 {
        return Err(ContainerError::Layout);
    }
    let mut c = Container::new([0; 32]);
    c.push(Tag::PROV, true, bytes[..header_len].to_vec());
    c.push(Tag::RAW, true, payload.to_vec());
    Ok(c)
}

/// Reads a container or an `ALICE_ZIP` file, telling them apart by their
/// first bytes.
///
/// # Errors
///
/// [`ContainerError::BadMagic`] for anything else, otherwise as
/// [`Container::from_bytes`] or [`read_legacy_alice_zip`].
pub fn read_any(bytes: &[u8]) -> Result<Container, ContainerError> {
    match detect(bytes) {
        Format::Container => Container::from_bytes(bytes),
        Format::LegacyAliceZip => read_legacy_alice_zip(bytes),
        Format::Unknown => Err(ContainerError::BadMagic),
    }
}
