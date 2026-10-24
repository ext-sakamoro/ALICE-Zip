//! The identifier of a law, v2: one encoding for every ALICE crate
//!
//! A law's identifier names exactly the inputs its evaluation reads: what kind of law it is,
//! the semantics it is evaluated under (a 32-byte semantics identifier) and its values. Every
//! crate that identifies laws builds the identifier with [`LawIdHasher`], so the same law gives
//! the same identifier wherever it is computed, and a kind tag (not a crate's own domain) tells
//! one family of laws from another.
//!
//! The bytes hashed, in order, every length a big-endian `u64`:
//!
//! ```text
//! len(LAW_ID_DOMAIN_V2) LAW_ID_DOMAIN_V2          b"alice-law-id/v2"
//! len(kind)             kind                      e.g. b"signal-law/polynomial/v1"
//!                       semantics_id              32 bytes
//! then each value, a one-byte type tag first:
//!   0x01 bytes  len data      0x02 u32  4 bytes BE      0x03 u64  8 bytes BE
//!   0x04 f64    bits 8 BE     0x05 str  len UTF-8       0x06 id   32 bytes
//!   0x07 f32    bits 4 BE
//! ```
//!
//! The tag keeps `u32 5` apart from `u64 5`; the length prefixes keep `("ab", "c")` apart from
//! `("a", "bc")`; the explicit byte order keeps the value independent of the host. A float enters
//! by its bits ([`alice_det_math::law_bits`]): `-0.0` and `+0.0` are different values, an `f32` is
//! never widened to `f64`, and a NaN other than the canonical one is refused, because a
//! platform's default NaN differs by target.
//! `scripts/law_id_v2_reference.py` is an independent implementation (standard-library Python)
//! and `tests/law_id_v2.rs` checks that both give the identifiers in
//! `tests/data/law_id/v2_probes.txt`.
//!
//! Author: Moroya Sakamoto

use alice_det_math::law_bits::{f32_bits, f64_bits, NonCanonicalNan};
use sha2::{Digest, Sha256};

/// The domain of the v2 law identifier, hashed first (it names the encoding's version)
pub const LAW_ID_DOMAIN_V2: &[u8] = b"alice-law-id/v2";

/// The type tags of the values (one byte before each value)
pub(crate) mod tag {
    pub(crate) const BYTES: u8 = 0x01;
    pub(crate) const U32: u8 = 0x02;
    pub(crate) const U64: u8 = 0x03;
    pub(crate) const F64: u8 = 0x04;
    pub(crate) const STR: u8 = 0x05;
    pub(crate) const ID: u8 = 0x06;
    pub(crate) const F32: u8 = 0x07;
}

/// Builds a law identifier: [`Self::new`] with the kind and the semantics, a value method per
/// input in the order the law reads them, then [`Self::finish`]
#[derive(Clone)]
pub struct LawIdHasher(Sha256);

impl core::fmt::Debug for LawIdHasher {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("LawIdHasher")
    }
}

fn len(h: &mut Sha256, n: usize) {
    h.update((n as u64).to_be_bytes());
}

impl LawIdHasher {
    /// Starts the identifier of a law of `kind` evaluated under `semantics_id`
    #[must_use]
    pub fn new(kind: &[u8], semantics_id: &[u8; 32]) -> Self {
        let mut h = Sha256::new();
        len(&mut h, LAW_ID_DOMAIN_V2.len());
        h.update(LAW_ID_DOMAIN_V2);
        len(&mut h, kind.len());
        h.update(kind);
        h.update(semantics_id);
        Self(h)
    }

    /// A byte string
    #[must_use]
    pub fn bytes(mut self, v: &[u8]) -> Self {
        self.0.update([tag::BYTES]);
        len(&mut self.0, v.len());
        self.0.update(v);
        self
    }

    /// A 32-bit unsigned integer
    #[must_use]
    pub fn u32(mut self, v: u32) -> Self {
        self.0.update([tag::U32]);
        self.0.update(v.to_be_bytes());
        self
    }

    /// A 64-bit unsigned integer
    #[must_use]
    pub fn u64(mut self, v: u64) -> Self {
        self.0.update([tag::U64]);
        self.0.update(v.to_be_bytes());
        self
    }

    /// A 64-bit float, by its bits
    ///
    /// # Errors
    /// [`NonCanonicalNan`] for a NaN other than the canonical one (no identifier names it).
    pub fn f64(mut self, v: f64) -> Result<Self, NonCanonicalNan> {
        let bits = f64_bits(v)?;
        self.0.update([tag::F64]);
        self.0.update(bits.to_be_bytes());
        Ok(self)
    }

    /// A 32-bit float, by its bits (not widened to `f64`)
    ///
    /// # Errors
    /// [`NonCanonicalNan`] for a NaN other than the canonical one (no identifier names it).
    pub fn f32(mut self, v: f32) -> Result<Self, NonCanonicalNan> {
        let bits = f32_bits(v)?;
        self.0.update([tag::F32]);
        self.0.update(bits.to_be_bytes());
        Ok(self)
    }

    /// A string (its UTF-8 bytes, under another tag than [`Self::bytes`])
    #[must_use]
    pub fn str(mut self, v: &str) -> Self {
        self.0.update([tag::STR]);
        len(&mut self.0, v.len());
        self.0.update(v.as_bytes());
        self
    }

    /// A 32-byte identifier (another law's, a semantics identifier, a digest)
    #[must_use]
    pub fn id(mut self, v: &[u8; 32]) -> Self {
        self.0.update([tag::ID]);
        self.0.update(v);
        self
    }

    /// The identifier
    #[must_use]
    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}
