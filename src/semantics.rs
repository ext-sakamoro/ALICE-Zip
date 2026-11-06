//! Content hash of what this crate computes: [`SEMANTICS_ID`](crate::law::SEMANTICS_ID).
//!
//! `alice-det-math`'s identifiers name the functions it computes; this crate's identifier
//! names what alice-zip itself computes with them. It is the fold of [`ZIP_SEMANTICS_PINS`]:
//! the recorded bit-pattern goldens of `tests/determinism_golden.rs` (the generators, the
//! spectrum analysis, the entropy and the polynomial generation), the SHA-256 of each
//! identifier tag a law identifier is built from (`LAW_ID_DOMAIN` and every law kind), and
//! [`DET_MATH_PINS`], the identifier of each `alice-det-math` function this crate calls
//! ([`DET_MATH_FUNCTIONS`]). Two builds with the same identifier compute every pinned output to
//! the same bits and build law identifiers the same way; a build that differs in any of them
//! has a different value.
//!
//! The identifier moves only when one of those moves (a golden re-recorded, a tag changed, a
//! function this crate calls), not with the crate's version, and not with a function of
//! `alice-det-math` this crate does not call. The law identifier goldens are not entries: they
//! are computed with this identifier, so they follow it.
//!
//! # Hex form
//!
//! ```text
//! 0ea7b226c52aa4a71b51d41697bf26cf26440280949a2095161fd775f16ca7df
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
//! (the source, not a copy), checks the tag entries against the tags themselves, and recomputes
//! [`ZIP_SEMANTICS_ID`]. Each [`DET_MATH_PINS`] entry is checked at compile time against
//! `alice_det_math::function_pin`. Re-recording order after an intended change: the golden or
//! the function's entry first, then the entry here, then the identifier.

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
/// tag entries hold the SHA-256 of the tag bytes. [`DET_MATH_PINS`] are folded with these, each
/// under the name `det-math <function>`.
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
];

/// The functions of `alice-det-math` this crate calls (ALICE-Law's `scripts/fold_set_check.py`
/// compares this with the calls in the source and writes it)
pub const DET_MATH_FUNCTIONS: &[&str] = &[
    "atan2",
    "cos",
    "cos64",
    "hypot",
    "ln64",
    "log2_64",
    "max",
    "max64",
    "min",
    "min64",
    "sin",
    "sin64",
    "sin_cos64",
];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order, as recorded when
/// [`ZIP_SEMANTICS_ID`] was computed. A function whose identifier moves in the resolved
/// `alice-det-math` stops the compile, rather than reporting an identifier computed from
/// other bits; a function this crate does not call moves nothing.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "atan2",
        hex32("f1227770b42015176d620dba415653be1a14b9708624f4f70594011c54fef376"),
    ),
    (
        "cos",
        hex32("f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b"),
    ),
    (
        "cos64",
        hex32("d9cf0bbeda9dbf12a0c98e913ce75b6f5a404c8ff6bac44aee88269f8f05bb75"),
    ),
    (
        "hypot",
        hex32("c97aba4deb475308ea0c63c46ad778172c0cbd8fe0844fb23546d5df854f99b6"),
    ),
    (
        "ln64",
        hex32("51c11488d360bb972761d2cecf192c173249f846f8127ef950e580b99a590cdc"),
    ),
    (
        "log2_64",
        hex32("b17861199f6cc751a8621dee28fe8a03f8181a02222b12de7dbd5133eaaa4823"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "max64",
        hex32("aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
    (
        "min64",
        hex32("1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb"),
    ),
    (
        "sin",
        hex32("ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec"),
    ),
    (
        "sin64",
        hex32("9d31aa70c38253a2ca1d1c7849f0a650c7ac1a15f8a594d10ad2c9b69c62b2ea"),
    ),
    (
        "sin_cos64",
        hex32("8563bcc81daec5dd7cbcdde5049385b79bc1a497ea6d9ad0ecd5cbd820457fc5"),
    ),
];

const _: () = {
    assert!(DET_MATH_FUNCTIONS.len() == DET_MATH_PINS.len());
    let mut i = 0;
    while i < DET_MATH_PINS.len() {
        assert!(
            same_bytes(
                DET_MATH_FUNCTIONS[i].as_bytes(),
                DET_MATH_PINS[i].0.as_bytes()
            ),
            "DET_MATH_PINS names another function than DET_MATH_FUNCTIONS"
        );
        assert!(
            same_bytes(
                &DET_MATH_PINS[i].1,
                &alice_det_math::function_pin(DET_MATH_PINS[i].0)
            ),
            "a function this crate calls moved in alice-det-math: re-record DET_MATH_PINS and ZIP_SEMANTICS_ID"
        );
        i += 1;
    }
};

/// [`SEMANTICS_ID`](crate::law::SEMANTICS_ID): the fold of [`ZIP_SEMANTICS_PINS`] and
/// [`DET_MATH_PINS`] (`tests/zip_semantics_id.rs` recomputes it)
pub const ZIP_SEMANTICS_ID: [u8; 32] =
    hex32("0ea7b226c52aa4a71b51d41697bf26cf26440280949a2095161fd775f16ca7df");

const fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
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
    // folding alice-det-math 0.6.1's whole identifier (`6fc214e9…`): the identifier now folds the
    // functions this crate calls instead; every pinned output has the same bits
    hex32("06c11067caed2f27b3323c25b023b1237fb5f5936bf86624b2dfb33759183461"),
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
