//! The reference to a law, v2: which law, by kind, semantics, identifier and its named values
//!
//! A [`LawRef`] names one law of any kind (a motion, an orbit, a signal law) and carries the rows
//! the law reads. Its identifier is the v2 law identifier ([`crate::law_id::LawIdHasher`]) of the
//! kind and semantics with, for each row in name order, the row's name (as a string value) and
//! then its value, so the bytes alone say which law they name: [`LawRef::from_bytes`] recomputes the
//! identifier and refuses bytes whose stated identifier differs, naming both. Evaluating the law
//! is left to the owner of its kind.
//!
//! The identifier names the law, not this encoding of it: it does not commit to "a law
//! reference", so any crate that hashes the same kind, semantics and (name, value) sequence with
//! [`LawIdHasher`] names the same law, which is what lets a reference be checked against the law
//! its owner computes. The kind is the namespace that keeps two families of laws apart.
//!
//! One law has exactly one byte string, so a signature or a digest over the bytes names the law:
//! the rows are kept in strictly ascending bytewise order of their names. [`LawRef::new`] sorts
//! them; [`LawRef::from_bytes`] refuses rows out of that order or with a repeated name.
//!
//! The bytes, every integer big-endian:
//!
//! ```text
//! magic "LOLR" | u16 version 2
//! u8 len(kind) | kind                       1..=255 bytes of UTF-8
//! semantics id (32) | law id (32)
//! u8 row count                              0..=MAX_ROWS
//! each row: u8 len(name) | name             1..=255 bytes of UTF-8, strictly ascending
//!           u8 tag | value                  the tags of the law identifier:
//!   0x01 bytes  u64 len | data      0x02 u32 (4)    0x03 u64 (8)
//!   0x04 f64    bits (8)            0x05 str    u64 len | UTF-8
//!   0x06 id (32)                    0x07 f32    bits (4)
//! nothing after the last row
//! ```
//!
//! A float row holds the value's bits; a NaN other than the canonical one is refused by
//! [`LawRef::new`] and by [`LawRef::from_bytes`] ([`alice_det_math::law_bits`]). Every length is checked against the bytes left before anything is
//! read, and nothing is sized from a stated count (the row count is at most [`MAX_ROWS`]).
//! `scripts/law_ref_v2_reference.py` is an independent implementation (standard-library Python)
//! and `tests/law_ref_v2.rs` checks that both agree on `tests/data/law_ref/v2_probes.txt`.
//!
//! Author: Moroya Sakamoto

use alloc::string::String;
use alloc::vec::Vec;

use alice_det_math::law_bits::{f32_bits, f64_bits, NonCanonicalNan};

use crate::law_id::{tag, LawIdHasher};

/// The first four bytes of a law reference
pub const LAW_REF_MAGIC: [u8; 4] = *b"LOLR";

/// The version this module reads and writes
pub const LAW_REF_VERSION: u16 = 2;

/// The most rows a law reference holds
pub const MAX_ROWS: usize = 64;

/// The longest kind and the longest row name, in bytes
pub const MAX_NAME_LEN: usize = 255;

/// One value a law reads
///
/// Two values are equal when they have the same type and the same bits (a float is compared by
/// its bits, so `-0.0` is not `+0.0` and the canonical NaN equals itself). New value types can be added
/// with new tags, so a `match` on it needs a wildcard arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Value {
    /// A byte string
    Bytes(Vec<u8>),
    /// A 32-bit unsigned integer
    U32(u32),
    /// A 64-bit unsigned integer
    U64(u64),
    /// A 64-bit float
    F64(f64),
    /// A 32-bit float (not widened to `f64`)
    F32(f32),
    /// A string
    Str(String),
    /// A 32-byte identifier (another law's, a semantics identifier, a digest)
    Id([u8; 32]),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Id(a), Self::Id(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

/// Why a law reference is refused
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LawRefError {
    /// The bytes do not start with [`LAW_REF_MAGIC`]
    BadMagic,
    /// The version is not [`LAW_REF_VERSION`]
    UnsupportedVersion(u16),
    /// The bytes end before the reference does
    Truncated,
    /// The kind is empty
    EmptyKind,
    /// The kind is longer than [`MAX_NAME_LEN`] bytes
    KindTooLong(usize),
    /// The kind, a row name or a string value is not UTF-8
    NotUtf8,
    /// More than [`MAX_ROWS`] rows
    TooManyRows(usize),
    /// A row name is empty
    EmptyName,
    /// A row name is longer than [`MAX_NAME_LEN`] bytes
    NameTooLong(usize),
    /// Two rows have this name
    DuplicateName(String),
    /// A row's name is not after the previous row's name (rows are in ascending name order)
    UnsortedRows,
    /// A value tag this version does not define (or reserves)
    UnknownTag(u8),
    /// A float row holds a NaN other than the canonical one
    NonCanonicalNan(NonCanonicalNan),
    /// Bytes remain after the last row
    TrailingBytes,
    /// The stated law identifier is not the one the kind, semantics and rows give
    LawIdMismatch {
        /// The identifier in the bytes
        stated: [u8; 32],
        /// The identifier the kind, semantics and rows give
        computed: [u8; 32],
    },
}

impl core::fmt::Display for LawRefError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not a law reference (magic)"),
            Self::UnsupportedVersion(v) => write!(f, "law reference version {v} is not 2"),
            Self::Truncated => f.write_str("the bytes end inside the law reference"),
            Self::EmptyKind => f.write_str("the kind is empty"),
            Self::KindTooLong(n) => write!(f, "the kind is {n} bytes, over {MAX_NAME_LEN}"),
            Self::NotUtf8 => f.write_str("a kind, row name or string is not UTF-8"),
            Self::TooManyRows(n) => write!(f, "{n} rows, over {MAX_ROWS}"),
            Self::EmptyName => f.write_str("a row name is empty"),
            Self::NameTooLong(n) => write!(f, "a row name is {n} bytes, over {MAX_NAME_LEN}"),
            Self::DuplicateName(n) => write!(f, "two rows are named {n:?}"),
            Self::UnsortedRows => f.write_str("the rows are not in ascending name order"),
            Self::UnknownTag(t) => write!(f, "value tag {t:#04x} is not defined"),
            Self::NonCanonicalNan(e) => write!(f, "a float row holds a {e}"),
            Self::TrailingBytes => f.write_str("bytes remain after the last row"),
            Self::LawIdMismatch { stated, computed } => {
                f.write_str("the stated law id ")?;
                hex(f, stated)?;
                f.write_str(" is not the computed ")?;
                hex(f, computed)
            }
        }
    }
}

fn hex(f: &mut core::fmt::Formatter<'_>, b: &[u8; 32]) -> core::fmt::Result {
    b.iter().try_for_each(|x| write!(f, "{x:02x}"))
}

#[cfg(feature = "std")]
impl std::error::Error for LawRefError {}

/// A reference to one law: its kind, semantics, identifier and rows
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LawRef {
    kind: Vec<u8>,
    semantics_id: [u8; 32],
    law_id: [u8; 32],
    rows: Vec<(String, Value)>,
}

fn check_kind(kind: &[u8]) -> Result<(), LawRefError> {
    if kind.is_empty() {
        return Err(LawRefError::EmptyKind);
    }
    if kind.len() > MAX_NAME_LEN {
        return Err(LawRefError::KindTooLong(kind.len()));
    }
    core::str::from_utf8(kind).map_err(|_| LawRefError::NotUtf8)?;
    Ok(())
}

/// `name` as the row after `prev` (the previous row's name, if any)
fn check_name(name: &str, prev: Option<&str>) -> Result<(), LawRefError> {
    if name.is_empty() {
        return Err(LawRefError::EmptyName);
    }
    if name.len() > MAX_NAME_LEN {
        return Err(LawRefError::NameTooLong(name.len()));
    }
    match prev.map(|p| p.as_bytes().cmp(name.as_bytes())) {
        Some(core::cmp::Ordering::Equal) => Err(LawRefError::DuplicateName(name.into())),
        Some(core::cmp::Ordering::Greater) => Err(LawRefError::UnsortedRows),
        _ => Ok(()),
    }
}

fn compute_law_id(
    kind: &[u8],
    semantics_id: &[u8; 32],
    rows: &[(String, Value)],
) -> Result<[u8; 32], LawRefError> {
    let mut h = LawIdHasher::new(kind, semantics_id);
    for (name, value) in rows {
        h = h.str(name);
        h = match value {
            Value::Bytes(b) => h.bytes(b),
            Value::U32(v) => h.u32(*v),
            Value::U64(v) => h.u64(*v),
            Value::F64(v) => h.f64(*v).map_err(LawRefError::NonCanonicalNan)?,
            Value::F32(v) => h.f32(*v).map_err(LawRefError::NonCanonicalNan)?,
            Value::Str(s) => h.str(s),
            Value::Id(id) => h.id(id),
        };
    }
    Ok(h.finish())
}

impl LawRef {
    /// The reference to the law of `kind` under `semantics_id` with `rows` (in any order: they
    /// are sorted by name); its identifier is computed here
    ///
    /// # Errors
    /// What [`Self::from_bytes`] would refuse in the same reference: an empty, too long or
    /// non-UTF-8 kind, more than [`MAX_ROWS`] rows, an empty, too long or repeated row name, a
    /// float row holding a NaN other than the canonical one.
    pub fn new(
        kind: &[u8],
        semantics_id: [u8; 32],
        mut rows: Vec<(String, Value)>,
    ) -> Result<Self, LawRefError> {
        check_kind(kind)?;
        if rows.len() > MAX_ROWS {
            return Err(LawRefError::TooManyRows(rows.len()));
        }
        rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (i, (name, _)) in rows.iter().enumerate() {
            check_name(name, i.checked_sub(1).map(|j| rows[j].0.as_str()))?;
        }
        let law_id = compute_law_id(kind, &semantics_id, &rows)?;
        Ok(Self {
            kind: kind.to_vec(),
            semantics_id,
            law_id,
            rows,
        })
    }

    /// The kind of law (its owner evaluates it)
    #[must_use]
    pub fn kind(&self) -> &[u8] {
        &self.kind
    }

    /// The semantics the law is evaluated under
    #[must_use]
    pub const fn semantics_id(&self) -> &[u8; 32] {
        &self.semantics_id
    }

    /// The law identifier
    #[must_use]
    pub const fn law_id(&self) -> &[u8; 32] {
        &self.law_id
    }

    /// The rows, in ascending name order
    #[must_use]
    pub fn rows(&self) -> &[(String, Value)] {
        &self.rows
    }

    /// The value of the row named `name`
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.rows
            .binary_search_by(|(n, _)| n.as_bytes().cmp(name.as_bytes()))
            .ok()
            .map(|i| &self.rows[i].1)
    }

    /// The bytes of this reference
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&LAW_REF_MAGIC);
        out.extend_from_slice(&LAW_REF_VERSION.to_be_bytes());
        // the lengths fit in a byte: `new` and `from_bytes` bound them by MAX_NAME_LEN and MAX_ROWS
        out.push(self.kind.len() as u8);
        out.extend_from_slice(&self.kind);
        out.extend_from_slice(&self.semantics_id);
        out.extend_from_slice(&self.law_id);
        out.push(self.rows.len() as u8);
        for (name, value) in &self.rows {
            out.push(name.len() as u8);
            out.extend_from_slice(name.as_bytes());
            match value {
                Value::Bytes(b) => {
                    out.push(tag::BYTES);
                    out.extend_from_slice(&(b.len() as u64).to_be_bytes());
                    out.extend_from_slice(b);
                }
                Value::U32(v) => {
                    out.push(tag::U32);
                    out.extend_from_slice(&v.to_be_bytes());
                }
                Value::U64(v) => {
                    out.push(tag::U64);
                    out.extend_from_slice(&v.to_be_bytes());
                }
                // the bits were checked by `new` / `from_bytes` (a canonical NaN or a number)
                Value::F64(v) => {
                    out.push(tag::F64);
                    out.extend_from_slice(&v.to_bits().to_be_bytes());
                }
                Value::F32(v) => {
                    out.push(tag::F32);
                    out.extend_from_slice(&v.to_bits().to_be_bytes());
                }
                Value::Str(s) => {
                    out.push(tag::STR);
                    out.extend_from_slice(&(s.len() as u64).to_be_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
                Value::Id(id) => {
                    out.push(tag::ID);
                    out.extend_from_slice(id);
                }
            }
        }
        out
    }

    /// The reference in `bytes`, refused unless the stated law identifier is the one its kind,
    /// semantics and rows give
    ///
    /// # Errors
    /// Any [`LawRefError`]; [`LawRefError::LawIdMismatch`] names both identifiers.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LawRefError> {
        let mut r = Reader { b: bytes, at: 0 };
        if r.take(4)? != LAW_REF_MAGIC {
            return Err(LawRefError::BadMagic);
        }
        let version = u16::from_be_bytes(r.array()?);
        if version != LAW_REF_VERSION {
            return Err(LawRefError::UnsupportedVersion(version));
        }
        let kind_len = usize::from(r.byte()?);
        let kind = r.take(kind_len)?;
        check_kind(kind)?;
        let semantics_id: [u8; 32] = r.array()?;
        let stated: [u8; 32] = r.array()?;
        let count = usize::from(r.byte()?);
        if count > MAX_ROWS {
            return Err(LawRefError::TooManyRows(count));
        }
        let mut rows: Vec<(String, Value)> = Vec::with_capacity(count);
        for _ in 0..count {
            let name_len = usize::from(r.byte()?);
            if name_len == 0 {
                return Err(LawRefError::EmptyName);
            }
            let name = r.utf8(name_len)?;
            check_name(&name, rows.last().map(|(n, _)| n.as_str()))?;
            let value = match r.byte()? {
                tag::BYTES => {
                    let n = r.len64()?;
                    Value::Bytes(r.take(n)?.to_vec())
                }
                tag::U32 => Value::U32(u32::from_be_bytes(r.array()?)),
                tag::U64 => Value::U64(u64::from_be_bytes(r.array()?)),
                tag::F64 => {
                    let v = f64::from_bits(u64::from_be_bytes(r.array()?));
                    f64_bits(v).map_err(LawRefError::NonCanonicalNan)?;
                    Value::F64(v)
                }
                tag::F32 => {
                    let v = f32::from_bits(u32::from_be_bytes(r.array()?));
                    f32_bits(v).map_err(LawRefError::NonCanonicalNan)?;
                    Value::F32(v)
                }
                tag::STR => {
                    let n = r.len64()?;
                    Value::Str(r.utf8(n)?)
                }
                tag::ID => Value::Id(r.array()?),
                other => return Err(LawRefError::UnknownTag(other)),
            };
            rows.push((name, value));
        }
        if r.at != bytes.len() {
            return Err(LawRefError::TrailingBytes);
        }
        let computed = compute_law_id(kind, &semantics_id, &rows)?;
        if computed != stated {
            return Err(LawRefError::LawIdMismatch { stated, computed });
        }
        Ok(Self {
            kind: kind.to_vec(),
            semantics_id,
            law_id: computed,
            rows,
        })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], LawRefError> {
        if n > self.b.len() - self.at {
            return Err(LawRefError::Truncated);
        }
        let out = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8, LawRefError> {
        Ok(self.take(1)?[0])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], LawRefError> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    /// A u64 length, refused when it is past the bytes left (so it never sizes anything; the
    /// `take` that follows compares again, this check keeps the conversion from `u64` honest)
    fn len64(&mut self) -> Result<usize, LawRefError> {
        let n = u64::from_be_bytes(self.array()?);
        usize::try_from(n)
            .ok()
            .filter(|&n| n <= self.b.len() - self.at)
            .ok_or(LawRefError::Truncated)
    }

    fn utf8(&mut self, n: usize) -> Result<String, LawRefError> {
        let raw = self.take(n)?;
        core::str::from_utf8(raw)
            .map(String::from)
            .map_err(|_| LawRefError::NotUtf8)
    }
}
