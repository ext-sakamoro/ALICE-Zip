//! Content hash of what this crate computes: [`SEMANTICS_ID`](crate::law::SEMANTICS_ID).
//!
//! `alice-det-math`'s identifier names the transcendental functions; this crate's identifier
//! names what alice-zip itself computes with them. It is the fold of [`ZIP_SEMANTICS_PINS`]:
//! the recorded bit-pattern goldens of `tests/determinism_golden.rs` (the generators, the
//! spectrum analysis, the entropy and the polynomial generation), the SHA-256 of each
//! identifier tag a law identifier is built from (`LAW_ID_DOMAIN` and every law kind), and
//! `alice_det_math::SEMANTICS_ID`. Two builds with the same identifier compute every pinned
//! output to the same bits and build law identifiers the same way; a build that differs in
//! any of them has a different value.
//!
//! The identifier moves only when one of those moves (a golden re-recorded, a tag changed, a
//! det-math generation), not with the crate's version. The law identifier goldens are not
//! entries: they are computed with this identifier, so they follow it.
//!
//! # Hex form (with `alice-det-math` 0.6.1; other generations in [`ZIP_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! 06c11067caed2f27b3323c25b023b1237fb5f5936bf86624b2dfb33759183461
//! ```
//!
//! # Fold
//!
//! The procedure of `alice-det-math`'s `SEMANTICS_ID` fold: sort the entries by name
//! (ascending byte order), reject an empty table, a repeated name, an empty name or a name
//! that is not ASCII, then SHA-256 over, for each entry in that order, the name length as 4
//! bytes big-endian (`u32`), the name bytes, and the 32 digest bytes.
//!
//! # Checked by
//!
//! `tests/zip_semantics_id.rs` reads every golden digest from `tests/determinism_golden.rs`
//! (the source, not a copy), checks the tag entries against the tags themselves and the
//! det-math entry against `alice_det_math::SEMANTICS_ID`, and recomputes every row of
//! [`ZIP_SEMANTICS_BY_DET_MATH`]. Re-recording order after an intended change: the golden
//! first, then the entry here, then the identifier.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a compile error.
pub(crate) const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The `(name, digest)` entries [`SEMANTICS_ID`](crate::law::SEMANTICS_ID) folds. The golden
/// entries are named after the test of `tests/determinism_golden.rs` that records them, the
/// tag entries hold the SHA-256 of the tag bytes, and `"alice-det-math"` holds the resolved
/// `alice_det_math::SEMANTICS_ID`.
pub const ZIP_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "golden sinusoid_generators_are_the_recorded_bits",
        hex32("96898e3223e8c8d49cdd2a6b49df652ab48347c45df5922237783e10a73a290d"),
    ),
    (
        "golden spectrum_and_reconstruction_are_the_recorded_bits",
        hex32("f354146d062704960e408e1e5c8c54872a636d8deb42d8f1704caf6822e5ec4a"),
    ),
    (
        "golden fft_spectrum_is_the_recorded_bits",
        hex32("a83cfb9c62c3c79379fa1d0df8b5d7aa2b5d98ec978ed7a064cd95524cb5c0cc"),
    ),
    (
        "golden entropy_is_the_recorded_bits",
        hex32("202c3247f87652a8eff9ed9c7c1f9e4ff7d598dda7e3934960469f829cd0e99e"),
    ),
    (
        "golden polynomial_generation_is_the_recorded_bits",
        hex32("45d33355d39cf0632cf5a5faa5cc92dcbf8ff7d65b9192dbb5e0b7a4c8b60727"),
    ),
    (
        "tag law::LAW_ID_DOMAIN",
        hex32("ef5947ee24f227228e3e72932c0bae747bc260df5a96429c8be92782ae2ac546"),
    ),
    (
        "tag law::SIGNAL_LAW_KIND",
        hex32("51ccf9fe2060d78e48d8a4a17d12abcc6585ae2440079a2b5fbb0e75a84d7ce0"),
    ),
    (
        "tag int2d::ZERO_LAW_KIND",
        hex32("bd25074040013f3693c679801ab6fc740c1a290c3877870b9ab560b831bc765e"),
    ),
    (
        "tag int2d::PLANE_PER_TILE_KIND",
        hex32("45c0198206340b88811ca3fedd0358ce6663075d6926140e1b4abccc53d38118"),
    ),
    (
        "tag int2d::DOMAIN_DIGEST_DOMAIN",
        hex32("f0f79bbdb5b1c0ff925f94c5f80fb1db90171522fbf9d33772328310288932dd"),
    ),
    // the resolved alice-det-math's own identifier, not a copy of it: the identifier is
    // looked up by it in ZIP_SEMANTICS_BY_DET_MATH
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// [`SEMANTICS_ID`](crate::law::SEMANTICS_ID) for each `alice-det-math` generation this crate
/// knows: `(alice_det_math::SEMANTICS_ID, fold of ZIP_SEMANTICS_PINS with that det-math
/// entry)`. A build against a generation with no row does not compile ([`semantics_id_for`]
/// panics in a const context), rather than reporting an identifier of another generation.
pub const ZIP_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.0
    (
        hex32("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
        hex32("2f46abb0b360203cb630d7453d4065070759255231ee84e2d0d4f53cd566d925"),
    ),
    // alice-det-math 0.5.1 (adds max / min / max64 / min64)
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
    ),
    // alice-det-math 0.5.2 at 39507d1b (clamp / clamp64 follow the G2 order: NaN in gives the
    // default NaN, equal zeros in the G2 order, a NaN bound is taken as the bound)
    (
        hex32("a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf"),
        hex32("50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406"),
    ),
    // alice-det-math 0.5.2 at 2d01c63f and 0.6.0, the same identifier (sin / cos at |x| >= 2^13
    // through sin64 / cos64, powf of a negative base, powf64 as fdlibm e_pow, hypot of two
    // subnormals; erf / erf64 added)
    (
        hex32("a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9"),
        hex32("17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c"),
    ),
    // alice-det-math 0.6.1 (adds tanh64 / hypot64; the bits of the kernels this crate calls are unchanged)
    (
        hex32("6fc214e924c53557f33775df51f1f8dfe2f246d67a9e493d6fb5837d999022f2"),
        hex32("06c11067caed2f27b3323c25b023b1237fb5f5936bf86624b2dfb33759183461"),
    ),
];

/// The identifier of a build that resolved the `alice-det-math` with identifier `det_math`
///
/// # Panics
///
/// When [`ZIP_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const context this
/// stops the compile). A new `alice-det-math` generation needs its row, recomputed from the
/// pins.
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < ZIP_SEMANTICS_BY_DET_MATH.len() {
        if same32(&ZIP_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return ZIP_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in ZIP_SEMANTICS_BY_DET_MATH: add its fold");
}

const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Earlier values of [`SEMANTICS_ID`](crate::law::SEMANTICS_ID) under which every pinned
/// output has the same bits as in this build, newest first
///
/// Up to 0.10 this crate published `alice-det-math`'s identifier as its own, so data stamped
/// by those releases carries `85e08846…` (det-math 0.5.1) or `135c407d…` (0.5.0); the outputs
/// they computed are the pinned ones, so they stay reproducible here.
pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[
    // with alice-det-math 0.6.0 (`a2a11b9b…`): 0.6.1 adds tanh64 / hypot64, which this crate does not call;
    // the identifier moves because it folds det-math's whole identifier, every pinned output has the same bits
    hex32("17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c"),
    // with alice-det-math 0.5.2 at 39507d1b (`a2e46794…`): the generation before det-math 0.6's
    // changes; alice-zip's library calls none of the functions whose bits moved (it uses sin64 /
    // cos64 / sin_cos64, atan2, log2_64 / ln64 and max / min), so every pinned output has the same bits
    hex32("50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406"),
    // with alice-det-math 0.5.1 (`85e08846…`): the generation before det-math's clamp followed the G2
    // order; alice-zip calls none of the functions that moved, so every pinned output has the same bits
    hex32("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
    hex32("65ba78c17a0cab6d1717270ad7ceca8856c293425f92289de943767ecb1ae883"),
    hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
    hex32("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
];

/// What a stored identifier says about data stamped with it: this build's identifier
/// ([`Verified`](alice_det_math::SemanticsCheck::Verified)), one of
/// [`PREVIOUS_SEMANTICS_IDS`] (`VerifiedPrevious`), or neither (`Mismatch`)
pub fn check_semantics(stored: &[u8; 32]) -> alice_det_math::SemanticsCheck {
    if *stored == crate::law::SEMANTICS_ID {
        alice_det_math::SemanticsCheck::Verified
    } else if PREVIOUS_SEMANTICS_IDS.contains(stored) {
        alice_det_math::SemanticsCheck::VerifiedPrevious(*stored)
    } else {
        alice_det_math::SemanticsCheck::Mismatch
    }
}
