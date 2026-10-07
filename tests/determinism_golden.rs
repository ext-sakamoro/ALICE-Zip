//! Bit-level golden for everything a law identifier rests on
//!
//! [`alice_zip::law::SignalLaw::law_id`] promises that two laws share an
//! identifier only if evaluating them returns the same bits for every `x`. That
//! promise is about *this* arithmetic, so it is only worth anything if the
//! arithmetic itself returns the same bits on every target. The analytic
//! oracles in `tests/analytic_oracle.rs` cannot see this: they compare against
//! closed forms with tolerances, so a one-ulp difference between two platforms
//! passes there. Nothing in the crate pinned a single bit before this file —
//! measured 2026-10-08, when moving the transcendentals to `alice-det-math`
//! changed the output of `generate_multi_sine` and `analyze_signal` and all
//! 193 existing tests stayed green.
//!
//! This file therefore records bit patterns. It is written to run **in both
//! feature sets** (`cargo test` and `cargo test --no-default-features`) and on
//! every target of the CI matrix: the expected digests below are the same
//! everywhere, so a disagreement between any two of those runs is a red test
//! rather than something nobody looks at.
//!
//! What the layers catch is deliberately disjoint:
//!
//! | layer | catches |
//! |-------|---------|
//! | `tests/analytic_oracle.rs` | the law computing the wrong thing |
//! | this file | the same law computing different bits in different builds |
//! | `clippy.toml` `disallowed-methods` | a platform transcendental coming back |
//!
//! Updating a digest here is a deliberate act: it means stored identifiers
//! computed under the old arithmetic no longer describe what the crate now
//! computes, which is a breaking change.

use alice_zip::generators::{
    analyze_signal, generate_from_coefficients, generate_multi_sine, generate_polynomial,
    generate_sine_wave,
};
use alice_zip::law::{Provenance, SignalLaw, SEMANTICS_ID};
use alice_zip::{shannon_entropy, theoretical_min_size};
use sha2::{Digest, Sha256};

/// Collects the bits that a scenario produced, so the comparison is over the
/// exact IEEE 754 encodings rather than over decimal text
#[derive(Default)]
struct Bits {
    hasher: Sha256,
    bytes: usize,
}

impl Bits {
    fn f32(&mut self, v: f32) {
        self.raw(&v.to_bits().to_be_bytes());
    }

    fn f64(&mut self, v: f64) {
        self.raw(&v.to_bits().to_be_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.raw(&v.to_be_bytes());
    }

    fn raw(&mut self, b: &[u8]) {
        self.hasher.update(b);
        self.bytes += b.len();
    }

    fn f32s(&mut self, vs: &[f32]) {
        self.u64(vs.len() as u64);
        for &v in vs {
            self.f32(v);
        }
    }
}

/// Compares a scenario against its recorded digest.
///
/// `min_bytes` is the emptiness gate: a comparison of nothing against nothing
/// succeeds, so a scenario that silently stops producing values would keep
/// this file green without it. The bound is checked before the digest so the
/// failure says which of the two went wrong.
fn assert_golden(scenario: &str, bits: Bits, min_bytes: usize, expected: &str) {
    assert!(
        bits.bytes >= min_bytes,
        "{scenario}: hashed {} bytes, expected at least {min_bytes} — \
         the scenario stopped producing values, so the digest below proves nothing",
        bits.bytes
    );
    let got = hex(&bits.hasher.finalize());
    assert_eq!(
        got, expected,
        "{scenario}: {} bytes hashed to {got}, recorded {expected}",
        bits.bytes
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A signal built without any transcendental, so a difference in the digests
/// below can be attributed to the function under test rather than to its input
fn integer_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * i + 7 * i) % 23) as f32) - 11.0)
        .collect()
}

// ---------------------------------------------------------------------------
// 1. The arithmetic identifier itself
//
// Every identifier this crate produces mixes this in, so a change to it
// changes all of them. Pinning it here means such a change cannot arrive as a
// silent dependency bump.
// ---------------------------------------------------------------------------

#[test]
fn semantics_id_is_the_recorded_value() {
    assert_eq!(
        hex(&SEMANTICS_ID),
        "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e",
        "the numeric semantics changed: every law identifier published under \
         the old value now describes different arithmetic"
    );
}

// ---------------------------------------------------------------------------
// 2. The law identifier and the evaluation it stands for
// ---------------------------------------------------------------------------

/// `y = 1 + 2x` fitted over `x = 0..=4`, the same law the module example uses
fn line_law() -> SignalLaw {
    let points: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    SignalLaw::fit_polynomial(&points, 1, Provenance::new("golden", "least squares"))
        .expect("the fixture satisfies every fit precondition")
}

#[test]
fn law_id_and_evaluation_are_the_recorded_bits() {
    let law = line_law();
    let mut bits = Bits::default();
    bits.raw(&law.law_id(&SEMANTICS_ID));
    // The identifier claims these bits; record them next to it, so a change to
    // either one without the other is visible here.
    for i in 0..=40_u32 {
        let x = f64::from(i) / 10.0;
        bits.f64(law.evaluate(x).expect("inside the domain"));
    }
    assert_golden(
        "law_id and evaluate",
        bits,
        32 + 41 * 8,
        "06f9beb9dcfaca1786bb8c6b195471bd8049d0159bd8355d92caf94e8464cb28",
    );
}

// ---------------------------------------------------------------------------
// 3. The transcendental paths
//
// These are the ones that actually differed between builds before the move to
// `alice-det-math`.
// ---------------------------------------------------------------------------

#[test]
fn sinusoid_generators_are_the_recorded_bits() {
    let mut bits = Bits::default();
    bits.f32s(&generate_sine_wave(64, 3.0, 2.0, 0.25, 0.5));
    bits.f32s(&generate_multi_sine(
        64,
        &[(1.0, 1.0, 0.0), (5.0, 0.3, 1.0), (7.5, 1.25, -2.0)],
        -0.75,
    ));
    // Sweeps the argument far from zero, where range reduction decides the
    // result and implementations differ most
    bits.f32s(&generate_sine_wave(32, 97.0, 1.0, 1e4, 0.0));
    assert_golden(
        "sine wave and multi sine",
        bits,
        (64 + 64 + 32) * 4 + 3 * 8,
        "d4a18c7d47f85e10504e716f49ae98c5079051588564173ca4de5803a8dfc7d2",
    );
}

#[test]
fn spectrum_and_reconstruction_are_the_recorded_bits() {
    let mut bits = Bits::default();
    for n in [8usize, 31, 64] {
        let signal = integer_signal(n);
        let (coefficients, dc) = analyze_signal(&signal, 8, 1.0);
        bits.u64(coefficients.len() as u64);
        for &(k, magnitude, phase) in &coefficients {
            bits.u64(k as u64);
            bits.f32(magnitude);
            bits.f32(phase);
        }
        bits.f32(dc);
        bits.f32s(&generate_from_coefficients(n, &coefficients, dc));
    }
    assert_golden(
        "analyze_signal and generate_from_coefficients",
        bits,
        (8 + 31 + 64) * 4,
        "249bb501b397591df38f2673a2944fd4ea28eec9627b9b8089ab9a4b88a0fb25",
    );
}

#[test]
fn entropy_is_the_recorded_bits() {
    let mut bits = Bits::default();
    let inputs: [Vec<u8>; 4] = [
        (0..=255_u16).map(|i| (i % 251) as u8).collect(),
        b"aaaaaaaaabbbbbbccdddddddddddddeeeeeeeeeeeeeeeeefffg".to_vec(),
        (0..=255_u8).collect(),
        (0..1000_u32).map(|i| ((i * i) % 7) as u8).collect(),
    ];
    for data in &inputs {
        bits.f64(shannon_entropy(data));
        bits.u64(theoretical_min_size(data) as u64);
    }
    assert_golden(
        "shannon_entropy",
        bits,
        inputs.len() * 16,
        "202c3247f87652a8eff9ed9c7c1f9e4ff7d598dda7e3934960469f829cd0e99e",
    );
}

// ---------------------------------------------------------------------------
// 4. A path with no transcendental on it
//
// Pinned so that a digest change elsewhere can be read as "the transcendentals
// moved" rather than "something about f32 arithmetic moved": this one has to
// stay put through any such change.
// ---------------------------------------------------------------------------

#[test]
fn polynomial_generation_is_the_recorded_bits() {
    let mut bits = Bits::default();
    for n in [1usize, 2, 17, 64] {
        bits.f32s(&generate_polynomial(n, &[0.5, -1.25, 0.125, 2.0]));
    }
    assert_golden(
        "generate_polynomial",
        bits,
        (1 + 2 + 17 + 64) * 4,
        "45d33355d39cf0632cf5a5faa5cc92dcbf8ff7d65b9192dbb5e0b7a4c8b60727",
    );
}
