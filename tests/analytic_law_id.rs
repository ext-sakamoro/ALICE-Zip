//! Oracles for the content identifier of a [`SignalLaw`].
//!
//! The identifier answers one question: *will two machines, two
//! implementations, or the same code twenty years apart compute the same
//! numbers from this law?* It is therefore a hash over exactly the inputs that
//! `SignalLaw::evaluate` reads, plus an identifier for the numeric semantics
//! used to evaluate it.
//!
//! `evaluate` reads exactly three things (measured, not assumed):
//!
//! ```text
//! fn evaluate(&self, x) -> Result<f64, LawError> {
//!     if !self.domain.contains(x) { return Err(OutOfRange); }
//!     let u = (x - self.domain.lo) / (self.domain.hi - self.domain.lo);
//!     horner_ascending(&self.coeffs, u)
//! }
//! ```
//!
//! so `domain.lo`, `domain.hi` and `coeffs` are in the identifier and
//! `evidence` / `residual` / `provenance` / `oracles` are not: the same law
//! obtained from different measurements evaluates identically and must share
//! an identifier.
//!
//! # What is guaranteed, and what is not
//!
//! **Guaranteed (and asserted below):** equal identifier implies
//! bit-identical `evaluate` over the whole domain. This is the direction that
//! reproducibility needs.
//!
//! **Not guaranteed:** the converse. Two laws that evaluate identically may
//! still get different identifiers — a trailing zero coefficient is the
//! simplest example, pinned in
//! [`evaluation_equivalent_laws_may_still_differ_in_id`]. Full semantic
//! normalisation (same meaning, same identifier) is a separate, unsolved
//! problem and this module does not claim it. Do not use the identifier for
//! deduplication.

use alice_zip::law::{
    OracleCase, Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange,
};

/// Stand-in for the numeric semantics identifier published by the arithmetic
/// crate. Any 32 bytes work here: these oracles only care that the value is
/// mixed in, not what it is.
const SEMANTICS_A: [u8; 32] = [0x11; 32];
const SEMANTICS_B: [u8; 32] = [0x22; 32];

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

fn parts(coefficients: Vec<f64>, lo: f64, hi: f64, evidence: Vec<(f64, f64)>) -> SignalLawParts {
    SignalLawParts {
        coefficients,
        domain: ValidRange { lo, hi },
        evidence,
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new("oracle", "hand-written"),
        oracles: Vec::new(),
    }
}

/// A law with the given coefficients over `[0, 1]`, with just enough evidence
/// to satisfy `from_parts`.
fn law(coefficients: Vec<f64>) -> SignalLaw {
    let n = coefficients.len().max(1);
    let evidence: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / n as f64;
            (t, 0.0)
        })
        .collect();
    SignalLaw::from_parts(parts(coefficients, 0.0, 1.0, evidence))
        .expect("the fixture satisfies every from_parts precondition")
}

/// Every `x` the oracles sample, including both endpoints of the domain.
fn grid() -> Vec<f64> {
    let mut xs = vec![0.0, 1.0];
    for i in 1..16_u32 {
        xs.push(f64::from(i) / 16.0);
    }
    xs
}

/// `evaluate` over [`grid`], as raw bits so that `-0.0` and `+0.0` are
/// distinguished and `NaN` compares equal to itself.
fn evaluation_bits(l: &SignalLaw) -> Vec<u64> {
    grid()
        .into_iter()
        .map(|x| match l.evaluate(x) {
            Ok(y) => y.to_bits(),
            Err(_) => u64::MAX,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. The guaranteed direction: equal id implies identical evaluation
// ---------------------------------------------------------------------------

#[test]
fn equal_law_id_implies_bit_identical_evaluation() {
    // Same coefficients and domain, deliberately different metadata.
    let a = SignalLaw::from_parts(parts(
        vec![0.25, -1.5, 2.0],
        0.0,
        1.0,
        vec![(0.0, 0.25), (0.5, 0.0), (1.0, 0.75)],
    ))
    .expect("fixture a is valid");

    let mut p = parts(
        vec![0.25, -1.5, 2.0],
        0.0,
        1.0,
        // different evidence (and therefore a different recomputed residual)
        vec![(0.1, 0.0), (0.2, 0.0), (0.9, 0.0), (1.0, 0.0)],
    );
    p.provenance = Provenance::new("a different source", "a different method");
    p.oracles = vec![OracleCase::new(0.5, 0.0, 1e-9, "some paper")];
    let b = SignalLaw::from_parts(p).expect("fixture b is valid");

    assert_eq!(
        a.law_id(&SEMANTICS_A),
        b.law_id(&SEMANTICS_A),
        "metadata must not enter the identifier"
    );
    assert_eq!(
        evaluation_bits(&a),
        evaluation_bits(&b),
        "equal identifier must imply bit-identical evaluation"
    );
}

#[test]
fn law_id_ignores_evidence_provenance_and_oracles() {
    let base = law(vec![1.0, 2.0]);
    let mut p = parts(vec![1.0, 2.0], 0.0, 1.0, vec![(0.3, 9.0), (0.7, -9.0)]);
    p.provenance = Provenance::new("unrelated", "unrelated");
    p.oracles = vec![
        OracleCase::new(0.1, 0.0, 1.0, "one"),
        OracleCase::new(0.2, 0.0, 1.0, "two"),
    ];
    let other = SignalLaw::from_parts(p).expect("fixture is valid");

    assert_eq!(base.law_id(&SEMANTICS_A), other.law_id(&SEMANTICS_A));
}

// ---------------------------------------------------------------------------
// 2. Injectivity over the evaluation inputs — this is the tooth
//
// One mutation per input that `evaluate` reads. A field that is missing from
// the identifier shows up here and nowhere else.
// ---------------------------------------------------------------------------

#[test]
fn law_id_changes_when_any_coefficient_changes() {
    let coefficients = vec![0.5, -2.0, 3.25, 7.0];
    let base = law(coefficients.clone()).law_id(&SEMANTICS_A);

    for i in 0..coefficients.len() {
        let mut mutated = coefficients.clone();
        mutated[i] = f64::from_bits(mutated[i].to_bits() ^ 1); // one ulp
        assert_ne!(
            law(mutated).law_id(&SEMANTICS_A),
            base,
            "coefficient {i} does not reach the identifier"
        );
    }
}

#[test]
fn law_id_changes_when_the_coefficient_count_changes() {
    let short = law(vec![1.0, 2.0]).law_id(&SEMANTICS_A);
    let long = law(vec![1.0, 2.0, 0.0]).law_id(&SEMANTICS_A);
    assert_ne!(
        short, long,
        "a law with one more coefficient must get a different identifier"
    );
}
// Note on what the previous oracle does *not* prove: the coefficients are the
// last field in the encoding, so dropping their length prefix would still give
// these two laws different digests (16 bytes of coefficients against 24). The
// prefixes are load-bearing only where one field is followed by another, which
// the unit test `length_prefix_prevents_field_confusion` in `src/law.rs`
// covers directly. Keeping that distinction visible here so nobody reads this
// test as evidence for the prefixes.

#[test]
fn law_id_changes_when_the_domain_changes() {
    let base = SignalLaw::from_parts(parts(
        vec![1.0, 2.0],
        0.0,
        1.0,
        vec![(0.0, 0.0), (1.0, 0.0)],
    ))
    .expect("valid")
    .law_id(&SEMANTICS_A);

    let lo_moved = SignalLaw::from_parts(parts(
        vec![1.0, 2.0],
        f64::from_bits(0.0_f64.to_bits() + 1), // smallest subnormal above 0
        1.0,
        vec![(0.5, 0.0), (1.0, 0.0)],
    ))
    .expect("valid")
    .law_id(&SEMANTICS_A);

    let hi_moved = SignalLaw::from_parts(parts(
        vec![1.0, 2.0],
        0.0,
        f64::from_bits(1.0_f64.to_bits() + 1), // one ulp above 1.0
        vec![(0.0, 0.0), (1.0, 0.0)],
    ))
    .expect("valid")
    .law_id(&SEMANTICS_A);

    assert_ne!(base, lo_moved, "domain.lo does not reach the identifier");
    assert_ne!(base, hi_moved, "domain.hi does not reach the identifier");
}

#[test]
fn law_id_changes_when_the_numeric_semantics_change() {
    let l = law(vec![1.0, 2.0]);
    assert_ne!(
        l.law_id(&SEMANTICS_A),
        l.law_id(&SEMANTICS_B),
        "the numeric semantics identifier must reach the identifier, otherwise \
         a change in the arithmetic silently reuses an old identifier"
    );
}

// ---------------------------------------------------------------------------
// 3. Zero signs are observable, so they are not normalised away
//
// `-0.0` and `+0.0` compare equal, which makes normalising them look harmless.
// It is not: with a negative linear coefficient, `acc * u` is `-0.0` at the
// lower end of the domain, and `-0.0 + (-0.0)` is `-0.0` while
// `-0.0 + (+0.0)` is `+0.0`. Folding the two together would hand the same
// identifier to two laws whose evaluation differs in bits.
// ---------------------------------------------------------------------------

#[test]
fn negative_zero_is_observable_in_evaluation_so_it_changes_the_id() {
    let plus = SignalLaw::from_parts(parts(
        vec![0.0, -1.0],
        0.0,
        1.0,
        vec![(0.0, 0.0), (1.0, -1.0)],
    ))
    .expect("valid");
    let minus = SignalLaw::from_parts(parts(
        vec![-0.0, -1.0],
        0.0,
        1.0,
        vec![(0.0, 0.0), (1.0, -1.0)],
    ))
    .expect("valid");

    let y_plus = plus.evaluate(0.0).expect("inside the domain");
    let y_minus = minus.evaluate(0.0).expect("inside the domain");

    // The premise of this oracle: the two really do evaluate to different bits.
    assert_eq!(y_plus.to_bits(), 0.0_f64.to_bits(), "expected +0.0");
    assert_eq!(y_minus.to_bits(), (-0.0_f64).to_bits(), "expected -0.0");

    assert_ne!(
        plus.law_id(&SEMANTICS_A),
        minus.law_id(&SEMANTICS_A),
        "the sign of a zero coefficient is observable, so it must not be \
         normalised away"
    );
}

// ---------------------------------------------------------------------------
// 4. The direction that is deliberately NOT guaranteed
// ---------------------------------------------------------------------------

#[test]
fn evaluation_equivalent_laws_may_still_differ_in_id() {
    // Appending a zero coefficient cannot change Horner's result: the fold
    // starts at 0.0 and `0.0 * u + 0.0` is `0.0` for every `u` in `[0, 1]`.
    let short = law(vec![1.0]);
    let long = law(vec![1.0, 0.0]);

    assert_eq!(
        evaluation_bits(&short),
        evaluation_bits(&long),
        "premise: a trailing zero coefficient does not change evaluation"
    );
    assert_ne!(
        short.law_id(&SEMANTICS_A),
        long.law_id(&SEMANTICS_A),
        "the identifier is over the law as written, not over a normal form; \
         this is deliberate and documented, do not `fix` it by stripping \
         trailing zeros without deciding what `degree()` then means"
    );
}

// ---------------------------------------------------------------------------
// 5. Degenerate and extreme inputs
//
// `from_parts` rejects non-finite coefficients and degenerate domains, so the
// identifier never sees them. These oracles pin that guard and check that the
// surviving extremes do not panic.
// ---------------------------------------------------------------------------

#[test]
fn from_parts_rejects_the_inputs_the_identifier_cannot_canonicalise() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            SignalLaw::from_parts(parts(vec![bad], 0.0, 1.0, vec![(0.0, 0.0)])).is_err(),
            "a non-finite coefficient must be rejected before it reaches the identifier"
        );
        assert!(
            SignalLaw::from_parts(parts(vec![1.0], bad, 1.0, vec![(0.0, 0.0)])).is_err(),
            "a non-finite domain bound must be rejected"
        );
    }
    assert!(
        SignalLaw::from_parts(parts(vec![1.0], 1.0, 1.0, vec![(1.0, 0.0)])).is_err(),
        "a degenerate domain must be rejected"
    );
    assert!(
        SignalLaw::from_parts(parts(Vec::new(), 0.0, 1.0, vec![(0.0, 0.0)])).is_err(),
        "an empty coefficient vector must be rejected"
    );
}

#[test]
fn law_id_is_stable_and_total_over_extreme_finite_values() {
    let extremes = vec![f64::MAX, f64::MIN, f64::MIN_POSITIVE, -f64::MIN_POSITIVE];
    let l = SignalLaw::from_parts(parts(
        extremes,
        -f64::MAX,
        f64::MAX,
        vec![(-f64::MAX, 0.0), (0.0, 0.0), (f64::MAX, 0.0), (1.0, 0.0)],
    ))
    .expect("extreme but finite values are accepted");

    let first = l.law_id(&SEMANTICS_A);
    assert_eq!(first, l.law_id(&SEMANTICS_A), "the identifier must be pure");
    assert_ne!(
        first, [0_u8; 32],
        "an all-zero digest means nothing was hashed"
    );
}

#[test]
fn law_id_of_the_smallest_possible_law() {
    let l = law(vec![1.0]);
    let id = l.law_id(&SEMANTICS_A);
    assert_ne!(id, [0_u8; 32]);
    assert_eq!(id, law(vec![1.0]).law_id(&SEMANTICS_A));
}

// ---------------------------------------------------------------------------
// 6. Cross-platform golden
//
// The constant below pins the whole encoding — the domain separation tag, the
// kind tag, the field order, the length prefixes and the byte order — against
// one recorded value. CI reproduces it on macOS, Linux (x86 and ARM), Windows
// and wasm; a mismatch on any of them means a platform-dependent operation
// entered the encoder.
//
// Updating it (only after an intentional encoding change): run the failing
// test, copy the "actual" hex in, and record the change in CHANGELOG. An
// encoding change invalidates every identifier ever published, so it also
// needs a new kind tag.
// ---------------------------------------------------------------------------

const GOLDEN_LAW_ID: &str = "582558584743c0888676d96bedb5441ebd87b7d4001d3b902d6e3af789b6ae55";

#[test]
fn law_id_golden() {
    let l = SignalLaw::from_parts(parts(
        vec![0.5, -2.0, 3.25],
        -1.0,
        2.0,
        vec![(-1.0, 0.0), (0.0, 0.5), (2.0, -1.0)],
    ))
    .expect("valid");

    let hex = l
        .law_id(&SEMANTICS_A)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    assert_eq!(
        hex, GOLDEN_LAW_ID,
        "\n\nLaw identifier golden mismatch.\n\
         actual:   {hex}\n\
         expected: {GOLDEN_LAW_ID}\n\n\
         If the encoding changed on purpose, bump the kind tag, update this \
         constant and record both in CHANGELOG. If it did not, a \
         platform-dependent operation entered the encoder.\n"
    );
}
