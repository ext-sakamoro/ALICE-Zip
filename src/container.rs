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
        todo!("Display for ContainerError")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContainerError {}

/// A container read from bytes without copying them.
///
/// [`ContainerView::open`] checks the header, the table and the trailer.
/// A payload's SHA-256 is checked when the payload is first asked for.
#[derive(Debug, Clone)]
pub struct ContainerView<'a> {
    bytes: &'a [u8],
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
        todo!("Container::new")
    }

    /// Appends a section and returns the SHA-256 of its payload.
    pub fn push(&mut self, tag: Tag, critical: bool, payload: Vec<u8>) -> [u8; 32] {
        todo!("Container::push")
    }

    /// The container's bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        todo!("Container::to_bytes")
    }

    /// Identity of the content (see the module documentation).
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        todo!("Container::id")
    }

    /// Reads a container and checks everything: the trailer, every payload,
    /// unknown critical sections, the references of [`Tag::SREF`] and the
    /// semantics of [`Tag::LIDS`].
    ///
    /// # Errors
    ///
    /// The first problem found, as a [`ContainerError`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContainerError> {
        todo!("Container::from_bytes")
    }
}

impl<'a> ContainerView<'a> {
    /// Checks the header, the table and the trailer.
    ///
    /// # Errors
    ///
    /// [`ContainerError::TooShort`], [`ContainerError::BadMagic`],
    /// [`ContainerError::UnsupportedMajor`], [`ContainerError::Reserved`],
    /// [`ContainerError::Layout`], [`ContainerError::Trailer`] or
    /// [`ContainerError::UnknownCritical`].
    pub fn open(bytes: &'a [u8]) -> Result<Self, ContainerError> {
        todo!("ContainerView::open")
    }

    /// Number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        todo!("ContainerView::len")
    }

    /// Whether the container has no section.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        todo!("ContainerView::is_empty")
    }

    /// Minor version.
    #[must_use]
    pub fn minor(&self) -> u16 {
        todo!("ContainerView::minor")
    }

    /// Semantics id in the header.
    #[must_use]
    pub fn semantics_id(&self) -> [u8; 32] {
        todo!("ContainerView::semantics_id")
    }

    /// Identity of the content (see the module documentation).
    #[must_use]
    pub fn id(&self) -> [u8; 32] {
        todo!("ContainerView::id")
    }

    /// Tag of section `index`.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn tag(&self, index: usize) -> Tag {
        todo!("ContainerView::tag")
    }

    /// Whether section `index` is critical.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn critical(&self, index: usize) -> bool {
        todo!("ContainerView::critical")
    }

    /// SHA-256 of section `index` as the table states it.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[must_use]
    pub fn sha256(&self, index: usize) -> [u8; 32] {
        todo!("ContainerView::sha256")
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
        todo!("ContainerView::section")
    }

    /// Index of the first section whose table entry states `sha`.
    #[must_use]
    pub fn find_sha(&self, sha: &[u8; 32]) -> Option<usize> {
        todo!("ContainerView::find_sha")
    }

    /// Rows of every [`Tag::SREF`] section, each resolved to a section.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`], [`ContainerError::SectionHash`] or
    /// [`ContainerError::DanglingSection`].
    pub fn references(&self) -> Result<Vec<Reference>, ContainerError> {
        todo!("ContainerView::references")
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
        todo!("ContainerView::references_with_builders")
    }

    /// Rows of every [`Tag::LIDS`] section. A row computed under other
    /// semantics than the header states is refused.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`], [`ContainerError::SectionHash`] or
    /// [`ContainerError::SemanticsMismatch`].
    pub fn law_ids(&self) -> Result<Vec<LawIdEntry>, ContainerError> {
        todo!("ContainerView::law_ids")
    }

    /// Rows of every [`Tag::LIDS`] section without comparing their
    /// semantics id with the header; the caller sees both in each row.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Malformed`] or [`ContainerError::SectionHash`].
    pub fn law_ids_unchecked_semantics(&self) -> Result<Vec<LawIdEntry>, ContainerError> {
        todo!("ContainerView::law_ids_unchecked_semantics")
    }

    /// Compares every [`Tag::LIDS`] row with the identifier `recompute`
    /// derives from the section's payload. `recompute` returns `None` for a
    /// payload it cannot interpret, which is refused as a mismatch.
    ///
    /// # Errors
    ///
    /// As [`Self::law_ids`], and [`ContainerError::LawIdMismatch`].
    pub fn verify_law_ids<F>(&self, recompute: F) -> Result<(), ContainerError>
    where
        F: FnMut(usize, &[u8], &[u8; 32]) -> Option<[u8; 32]>,
    {
        todo!("ContainerView::verify_law_ids")
    }
}

/// What the first bytes of `bytes` are.
#[must_use]
pub fn detect(bytes: &[u8]) -> Format {
    todo!("detect")
}

/// Reads an `ALICE_ZIP` file as a container: the header bytes as
/// [`Tag::PROV`] and the payload as [`Tag::RAW`], with an all-zero semantics
/// id.
///
/// # Errors
///
/// [`ContainerError::TooShort`], [`ContainerError::BadMagic`],
/// [`ContainerError::LegacyVersion`], [`ContainerError::LegacyField`] or
/// [`ContainerError::Layout`] (the payload length differs from the length
/// the header states).
pub fn read_legacy_alice_zip(bytes: &[u8]) -> Result<Container, ContainerError> {
    todo!("read_legacy_alice_zip")
}

/// Reads a container or an `ALICE_ZIP` file, telling them apart by their
/// first bytes.
///
/// # Errors
///
/// [`ContainerError::BadMagic`] for anything else, otherwise as
/// [`Container::from_bytes`] or [`read_legacy_alice_zip`].
pub fn read_any(bytes: &[u8]) -> Result<Container, ContainerError> {
    todo!("read_any")
}
