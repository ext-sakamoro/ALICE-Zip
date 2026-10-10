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
    /// Data offered as the original of an `ALICE_ZIP` file is not as long as
    /// the header states.
    OriginalSize {
        /// `original_size` in the header.
        stated: u64,
        /// Length of the data offered.
        actual: u64,
    },
    /// Data offered as the original of an `ALICE_ZIP` file does not match
    /// the header's `original_hash`.
    OriginalHash,
    /// An `ALICE_ZIP` payload of a kind `decompress_legacy_alice_zip` does
    /// not decode: everything except the LZMA fallback (`payload_type`
    /// `0x30`).
    UnsupportedPayload {
        /// `payload_type` in the header; `None` for version 1.0.
        payload_type: Option<u8>,
    },
    /// An `ALICE_ZIP` file states an `original_size` larger than the caller
    /// accepts.
    LegacyTooLarge {
        /// `original_size` in the header.
        stated: u64,
        /// The limit the caller gave.
        limit: u64,
    },
    /// An `ALICE_ZIP` LZMA fallback payload is not as the writer produces
    /// it: the metadata length, the metadata JSON, the xz stream, or a
    /// result longer than `original_size` or not of the size the metadata's
    /// shape and dtype give.
    LegacyPayload,
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
            Self::OriginalSize { stated, actual } => write!(
                f,
                "ALICE_ZIP: the original is {actual} bytes, the header states {stated}"
            ),
            Self::OriginalHash => {
                write!(f, "ALICE_ZIP: the original does not match original_hash")
            }
            Self::UnsupportedPayload { payload_type } => match payload_type {
                Some(t) => write!(f, "ALICE_ZIP: payload_type {t:#04x} is not decoded here"),
                None => write!(f, "ALICE_ZIP: a version 1.0 payload is not decoded here"),
            },
            Self::LegacyTooLarge { stated, limit } => write!(
                f,
                "ALICE_ZIP: original_size {stated} exceeds the limit {limit}"
            ),
            Self::LegacyPayload => {
                write!(f, "ALICE_ZIP: the LZMA fallback payload is malformed")
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
/// `payload_type` of the lossless LZMA fallback payload.
const LEGACY_LZMA_FALLBACK: u8 = 0x30;
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

/// The header of an `ALICE_ZIP` file (version 1.0 or 1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacyHeader {
    /// Major version (1).
    pub major: u8,
    /// Minor version (0 or 1).
    pub minor: u8,
    /// `file_type` (1 to 5).
    pub file_type: u8,
    /// Index of the compression engine (0 to 3).
    pub engine: u8,
    /// `payload_type`; version 1.0 has no such field.
    pub payload_type: Option<u8>,
    /// Length of the original data.
    pub original_size: u64,
    /// Length of the payload after the header.
    pub compressed_size: u64,
    /// SHA-256 of the original data; `None` when the writer stored all
    /// zeros, which is how it marks that no hash was recorded.
    pub original_hash: Option<[u8; 32]>,
}

impl LegacyHeader {
    /// Whether decompressing the payload reproduces the original exactly, so
    /// that `original_hash` can be checked against the result.
    ///
    /// Only the LZMA fallback payload (`payload_type` `0x30`) is lossless.
    /// A procedural payload (and every version 1.0 file, which has no
    /// `payload_type` and is procedural) stores generator parameters and is
    /// regenerated approximately, and the media and texture payloads store
    /// parameters too; for those the hash describes data the reader does not
    /// reproduce, so it cannot match by design.
    #[must_use]
    pub fn original_hash_checkable(&self) -> bool {
        self.payload_type == Some(LEGACY_LZMA_FALLBACK)
    }

    /// Checks data offered as the original: its length against
    /// `original_size`, and its SHA-256 against `original_hash` when the
    /// header records one.
    ///
    /// The container does not decompress the payload, so it cannot produce
    /// the original itself; a reader that decompresses passes its result
    /// here, for payloads where [`Self::original_hash_checkable`] holds.
    ///
    /// # Errors
    ///
    /// [`ContainerError::OriginalSize`] or [`ContainerError::OriginalHash`].
    pub fn verify_original(&self, original: &[u8]) -> Result<(), ContainerError> {
        let actual = original.len() as u64;
        if actual != self.original_size {
            return Err(ContainerError::OriginalSize {
                stated: self.original_size,
                actual,
            });
        }
        match self.original_hash {
            Some(h) if sha256(original) != h => Err(ContainerError::OriginalHash),
            _ => Ok(()),
        }
    }
}

/// Parses and checks the header of an `ALICE_ZIP` file, returning it and its
/// length (65 for version 1.0, 66 for version 1.1).
///
/// Only what the writer produced is accepted: version 1.0 or 1.1, a
/// `file_type` of 1 to 5, an engine index of 0 to 3 and a defined
/// `payload_type`. Bytes after the header are not read.
///
/// # Errors
///
/// [`ContainerError::TooShort`], [`ContainerError::BadMagic`],
/// [`ContainerError::LegacyVersion`] or [`ContainerError::LegacyField`].
pub fn parse_legacy_alice_zip_header(
    bytes: &[u8],
) -> Result<(LegacyHeader, usize), ContainerError> {
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
    let (payload_type, sizes_at) = if header_len == LEGACY_V2_LEN {
        if !LEGACY_PAYLOAD_TYPES.contains(&bytes[13]) {
            return Err(ContainerError::LegacyField {
                field: "payload_type",
                value: bytes[13],
            });
        }
        (Some(bytes[13]), 14)
    } else {
        (None, 13)
    };
    let hash = array32(bytes, sizes_at + 16);
    Ok((
        LegacyHeader {
            major,
            minor,
            file_type: bytes[11],
            engine: bytes[12],
            payload_type,
            original_size: u64_at(bytes, sizes_at),
            compressed_size: u64_at(bytes, sizes_at + 8),
            original_hash: (hash != [0; 32]).then_some(hash),
        },
        header_len,
    ))
}

/// Reads an `ALICE_ZIP` file as a container: the header bytes as
/// [`Tag::PROV`] and the payload as [`Tag::RAW`], with an all-zero semantics
/// id.
///
/// The header is checked as in [`parse_legacy_alice_zip_header`], and the
/// payload must be exactly as long as the header states. The payload is not
/// decompressed here, so `original_hash` is not checked here;
/// `decompress_legacy_alice_zip` (`lzma` feature) decompresses the LZMA
/// fallback payload and checks its result with
/// [`LegacyHeader::verify_original`].
///
/// # Errors
///
/// As [`parse_legacy_alice_zip_header`], and [`ContainerError::Layout`]
/// (the payload length differs from the length the header states).
pub fn read_legacy_alice_zip(bytes: &[u8]) -> Result<Container, ContainerError> {
    let (header, header_len) = parse_legacy_alice_zip_header(bytes)?;
    let payload = &bytes[header_len..];
    if header.compressed_size != payload.len() as u64 {
        return Err(ContainerError::Layout);
    }
    let mut c = Container::new([0; 32]);
    c.push(Tag::PROV, true, bytes[..header_len].to_vec());
    c.push(Tag::RAW, true, payload.to_vec());
    Ok(c)
}

/// An array decoded from an `ALICE_ZIP` LZMA fallback payload.
#[cfg(feature = "lzma")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyArray {
    /// Dimensions, as the writer recorded them.
    pub shape: Vec<u64>,
    /// `NumPy` dtype name, as the writer recorded it (for example `float64`).
    pub dtype: alloc::string::String,
    /// The array's bytes in the writer's order (`ndarray.tobytes()`), checked
    /// against `original_size` and `original_hash`.
    pub data: Vec<u8>,
}

/// Decodes an `ALICE_ZIP` file whose payload is the LZMA fallback
/// (`payload_type` `0x30`), the one payload that reproduces the original
/// exactly, and checks the result with [`LegacyHeader::verify_original`].
///
/// The payload is `meta_len` (u32 LE) ‖ metadata JSON ‖ an xz stream, as the
/// Python writer (`ALICEZip.compress`) produces it. The metadata must be in
/// the writer's form, `{"shape":[..],"dtype":".."}`, with a native-order
/// dtype name whose size is known (a byte-order prefix such as `>f8` is
/// refused), and the decoded bytes must be exactly the shape's element count
/// times that size.
///
/// # Memory
///
/// `original_size` comes from the file, and a few bytes of LZMA2 can expand
/// to megabytes, so the caller states how large a result it accepts:
/// `limit` bounds `original_size`. Before anything is decoded, the xz
/// container is parsed (one stream with a CRC-64 check, one block whose only
/// filter is LZMA2, its index and footer, nothing after it, as the writer
/// produces it) and the LZMA2 chunk headers are read; the unpacked sizes they
/// state must add up to `original_size`. The decoder then produces each
/// chunk only up to its stated size, so memory stays proportional to
/// `original_size` (at most `limit`), whatever the payload claims.
///
/// Procedural, media and texture payloads store generator parameters that
/// only the Python package regenerates; they are refused here.
///
/// # Errors
///
/// As [`parse_legacy_alice_zip_header`]; [`ContainerError::Layout`] (the
/// payload length differs from the header);
/// [`ContainerError::UnsupportedPayload`];
/// [`ContainerError::LegacyTooLarge`] (`original_size` exceeds `limit`);
/// [`ContainerError::LegacyPayload`]; [`ContainerError::OriginalSize`] or
/// [`ContainerError::OriginalHash`].
#[cfg(feature = "lzma")]
pub fn decompress_legacy_alice_zip(
    bytes: &[u8],
    limit: u64,
) -> Result<LegacyArray, ContainerError> {
    let (header, header_len) = parse_legacy_alice_zip_header(bytes)?;
    let payload = &bytes[header_len..];
    if header.compressed_size != payload.len() as u64 {
        return Err(ContainerError::Layout);
    }
    if header.payload_type != Some(LEGACY_LZMA_FALLBACK) {
        return Err(ContainerError::UnsupportedPayload {
            payload_type: header.payload_type,
        });
    }
    if header.original_size > limit {
        return Err(ContainerError::LegacyTooLarge {
            stated: header.original_size,
            limit,
        });
    }
    let bad = ContainerError::LegacyPayload;
    let meta_len = u32::from_le_bytes(payload.get(..4).ok_or(bad)?.try_into().map_err(|_| bad)?);
    let meta_end = usize::try_from(meta_len)
        .ok()
        .and_then(|m| m.checked_add(4))
        .filter(|&e| e <= payload.len())
        .ok_or(bad)?;
    let (shape, dtype) = parse_legacy_meta(&payload[4..meta_end]).ok_or(bad)?;
    let item = legacy_dtype_size(dtype).ok_or(bad)?;

    let xz = xz::parse(&payload[meta_end..]).ok_or(bad)?;
    if xz.unpacked != header.original_size {
        return Err(bad);
    }
    // The chunk headers bound what the decoder produces (each chunk stops at
    // its stated size), so the output stays near `original_size`.
    let mut data = Vec::new();
    if !xz.chunks.is_empty() {
        lzma_rs::lzma2_decompress(&mut &*xz.chunks, &mut data).map_err(|_| bad)?;
    }
    if xz::crc64(&data) != xz.check {
        return Err(bad);
    }
    header.verify_original(&data)?;

    let elements = shape
        .iter()
        .try_fold(1_u64, |n, &d| n.checked_mul(d))
        .ok_or(bad)?;
    if elements.checked_mul(item) != Some(data.len() as u64) {
        return Err(bad);
    }
    Ok(LegacyArray {
        shape,
        dtype: dtype.into(),
        data,
    })
}

/// The xz container (`.xz`, version 1.2.0 of the format) as Python's
/// `lzma.compress` writes it: one stream, CRC-64 check, one block whose only
/// filter is LZMA2, no stream padding. Every field is checked; anything else
/// is `None`.
#[cfg(feature = "lzma")]
mod xz {
    use crc::{Crc, CRC_32_ISO_HDLC, CRC_64_XZ};

    const MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];
    const FOOTER_MAGIC: [u8; 2] = *b"YZ";
    /// Stream flags: check type 0x04 (CRC-64).
    const FLAGS_CRC64: [u8; 2] = [0x00, 0x04];
    const FILTER_LZMA2: u64 = 0x21;
    /// Largest LZMA2 dictionary size byte the format defines (4 GiB − 1).
    const MAX_DICT_BYTE: u8 = 40;

    pub(super) fn crc32(b: &[u8]) -> u32 {
        Crc::<u32>::new(&CRC_32_ISO_HDLC).checksum(b)
    }

    pub(super) fn crc64(b: &[u8]) -> u64 {
        Crc::<u64>::new(&CRC_64_XZ).checksum(b)
    }

    /// The parts of a parsed stream the decoder needs.
    pub(super) struct Stream<'a> {
        /// The LZMA2 chunks of the block, up to and including the end marker.
        pub chunks: &'a [u8],
        /// Sum of the unpacked sizes the chunk headers state.
        pub unpacked: u64,
        /// CRC-64 of the uncompressed data, from the block.
        pub check: u64,
    }

    fn u32_le(b: &[u8], at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
    }

    /// Reads a multibyte integer (7 bits per byte, at most 9 bytes, no
    /// redundant trailing zero byte) and returns it with its length.
    fn varint(b: &[u8]) -> Option<(u64, usize)> {
        let mut v = 0_u64;
        for (i, &byte) in b.iter().enumerate().take(9) {
            v |= u64::from(byte & 0x7F) << (7 * i);
            if byte & 0x80 == 0 {
                if i > 0 && byte == 0 {
                    return None;
                }
                return Some((v, i + 1));
            }
        }
        None
    }

    /// Walks the LZMA2 chunk headers without decoding and returns the length
    /// of the chunk data (end marker included) and the unpacked total.
    fn scan_lzma2(b: &[u8]) -> Option<(usize, u64)> {
        let mut at = 0_usize;
        let mut total = 0_u64;
        loop {
            let control = *b.get(at)?;
            // The first chunk resets the dictionary (uncompressed 0x01, or
            // LZMA with a dictionary reset, 0xE0 and up), as the format
            // requires; only an empty block may end at once.
            if at == 0 && !matches!(control, 0x00 | 0x01 | 0xE0..=0xFF) {
                return None;
            }
            match control {
                0x00 => return Some((at + 1, total)),
                0x01 | 0x02 => {
                    let size =
                        usize::from(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?))
                            + 1;
                    at = at.checked_add(3 + size)?;
                    total = total.checked_add(size as u64)?;
                }
                0x80..=0xFF => {
                    let low =
                        u64::from(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?));
                    let unpacked = ((u64::from(control & 0x1F) << 16) | low) + 1;
                    let packed =
                        usize::from(u16::from_be_bytes(b.get(at + 3..at + 5)?.try_into().ok()?))
                            + 1;
                    let props = usize::from(control >= 0xC0);
                    // Each LZMA chunk starts a range coder, whose first byte
                    // is always 0 (liblzma refuses anything else).
                    if *b.get(at + 5 + props)? != 0 {
                        return None;
                    }
                    at = at.checked_add(5 + props + packed)?;
                    total = total.checked_add(unpacked)?;
                }
                _ => return None,
            }
            if at > b.len() {
                return None;
            }
        }
    }

    /// Index at `at`: indicator, record count (1 with `record`, 0 without),
    /// the record (unpadded size, unpacked size), padding, CRC-32. Returns
    /// the index length.
    fn index(b: &[u8], at: usize, record: Option<(u64, u64)>) -> Option<usize> {
        if *b.get(at)? != 0x00 {
            return None;
        }
        let mut i = at + 1;
        let (records, n) = varint(b.get(i..)?)?;
        i += n;
        match record {
            None if records == 0 => {}
            Some((unpadded, unpacked)) if records == 1 => {
                let (u, n) = varint(b.get(i..)?)?;
                i += n;
                let (v, n) = varint(b.get(i..)?)?;
                i += n;
                if u != unpadded || v != unpacked {
                    return None;
                }
            }
            _ => return None,
        }
        let end = at + (i - at).next_multiple_of(4);
        if b.get(i..end)?.iter().any(|&x| x != 0) || u32_le(b, end)? != crc32(&b[at..end]) {
            return None;
        }
        Some(end + 4 - at)
    }

    /// Stream footer at `at`: CRC-32, backward size, flags, magic, and
    /// nothing after it.
    fn footer(b: &[u8], at: usize, index_size: usize) -> Option<()> {
        (b.len() == at + 12
            && u32_le(b, at)? == crc32(&b[at + 4..at + 10])
            && u64::from(u32_le(b, at + 4)?) == (index_size / 4 - 1) as u64
            && b[at + 8..at + 10] == FLAGS_CRC64
            && b[at + 10..at + 12] == FOOTER_MAGIC)
            .then_some(())
    }

    pub(super) fn parse(b: &[u8]) -> Option<Stream<'_>> {
        // Stream header: magic, flags, CRC-32 of the flags.
        if b.get(..6)? != MAGIC || b.get(6..8)? != FLAGS_CRC64 || u32_le(b, 8)? != crc32(&b[6..8]) {
            return None;
        }
        // Empty input: no block, an index with no record.
        if *b.get(12)? == 0x00 {
            let index_size = index(b, 12, None)?;
            footer(b, 12 + index_size, index_size)?;
            return Some(Stream {
                chunks: &[],
                unpacked: 0,
                check: crc64(&[]),
            });
        }
        // Block header.
        let block = 12;
        let header_len = (usize::from(b[block]) + 1) * 4;
        let h = b.get(block..block + header_len)?;
        if u32_le(h, header_len - 4)? != crc32(&h[..header_len - 4]) {
            return None;
        }
        let flags = h[1];
        // One filter, optional sizes, reserved bits zero.
        if flags & 0x3C != 0 || flags & 0x03 != 0 {
            return None;
        }
        let mut at = 2;
        let mut stated_packed = None;
        let mut stated_unpacked = None;
        if flags & 0x40 != 0 {
            let (v, n) = varint(&h[at..header_len - 4])?;
            stated_packed = Some(v);
            at += n;
        }
        if flags & 0x80 != 0 {
            let (v, n) = varint(&h[at..header_len - 4])?;
            stated_unpacked = Some(v);
            at += n;
        }
        let (id, n) = varint(&h[at..header_len - 4])?;
        at += n;
        let (props_len, n) = varint(&h[at..header_len - 4])?;
        at += n;
        if id != FILTER_LZMA2 || props_len != 1 || *h.get(at)? > MAX_DICT_BYTE {
            return None;
        }
        at += 1;
        if at > header_len - 4 || h[at..header_len - 4].iter().any(|&x| x != 0) {
            return None;
        }

        // Compressed data, block padding, check.
        let data = block + header_len;
        let (chunks_len, unpacked) = scan_lzma2(b.get(data..)?)?;
        if stated_packed.is_some_and(|v| v != chunks_len as u64)
            || stated_unpacked.is_some_and(|v| v != unpacked)
        {
            return None;
        }
        let unpadded = header_len + chunks_len + 8;
        let padded_end = data + chunks_len.next_multiple_of(4);
        if b.get(data + chunks_len..padded_end)?
            .iter()
            .any(|&x| x != 0)
        {
            return None;
        }
        let check = u64::from_le_bytes(b.get(padded_end..padded_end + 8)?.try_into().ok()?);

        let at = padded_end + 8;
        let index_size = index(b, at, Some((unpadded as u64, unpacked)))?;
        footer(b, at + index_size, index_size)?;
        Some(Stream {
            chunks: &b[data..data + chunks_len],
            unpacked,
            check,
        })
    }
}

/// Parses the writer's metadata, `{"shape":[d,..],"dtype":"name"}` with no
/// spaces (`json.dumps(.., separators=(',', ':'))`). Anything else is `None`.
#[cfg(feature = "lzma")]
fn parse_legacy_meta(meta: &[u8]) -> Option<(Vec<u64>, &str)> {
    let mut rest = meta.strip_prefix(b"{\"shape\":[")?;
    let mut shape = Vec::new();
    if let Some(r) = rest.strip_prefix(b"]") {
        rest = r;
    } else {
        loop {
            let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
            if digits == 0 || (digits > 1 && rest[0] == b'0') {
                return None;
            }
            let text = core::str::from_utf8(&rest[..digits]).ok()?;
            shape.push(text.parse::<u64>().ok()?);
            rest = &rest[digits..];
            match rest.split_first()? {
                (b',', r) => rest = r,
                (b']', r) => {
                    rest = r;
                    break;
                }
                _ => return None,
            }
        }
    }
    let rest = rest.strip_prefix(b",\"dtype\":\"")?;
    let name_len = rest
        .iter()
        .take_while(|b| b.is_ascii_alphanumeric())
        .count();
    if rest.get(name_len..)? != b"\"}" {
        return None;
    }
    Some((shape, core::str::from_utf8(&rest[..name_len]).ok()?))
}

/// Size in bytes of one element of a `NumPy` dtype the writer can record.
#[cfg(feature = "lzma")]
fn legacy_dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" | "float16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" | "complex64" => 8,
        "complex128" => 16,
        _ => return None,
    })
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
