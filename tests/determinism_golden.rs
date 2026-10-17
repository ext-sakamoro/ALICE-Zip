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
        "51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e",
        "the semantics changed: every law identifier published under \
         the old value now describes something else"
    );
    // alice-zip's identifier folds alice-det-math's, pinned separately
    assert_eq!(
        hex(&alice_zip::law::DET_MATH_SEMANTICS_ID),
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
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
        "330cc74e2fed5d92891e0db1ffed224772041fa425dc4e8aa3c471987aaf4344",
    );
}

/// The digest recorded before the `fft` spectrum joined the pins (0.11.0, when the
/// in-crate transform replaced rustfft): passing the identifier of that build
/// (`65ba78c1…`, now in `PREVIOUS_SEMANTICS_IDS`) still gives it, so adding the pin
/// changed the identifier input only
#[test]
fn law_id_digest_under_the_identifier_before_the_fft_pin_is_unchanged() {
    let previous = alice_zip::law::PREVIOUS_SEMANTICS_IDS
        .iter()
        .find(|id| {
            hex(&id[..]) == "65ba78c17a0cab6d1717270ad7ceca8856c293425f92289de943767ecb1ae883"
        })
        .expect("alice-zip keeps the identifier before the fft pin as a previous one");
    let law = line_law();
    let mut bits = Bits::default();
    bits.raw(&law.law_id(previous));
    for i in 0..=40_u32 {
        let x = f64::from(i) / 10.0;
        bits.f64(law.evaluate(x).expect("inside the domain"));
    }
    assert_golden(
        "law_id and evaluate under the identifier before the fft pin",
        bits,
        32 + 41 * 8,
        "b8a5064b51c71bf4388082efe3ccfce7e73e1989accb4a0f07c4e38dbe11f022",
    );
}

/// The digest recorded up to 0.10, when this crate published alice-det-math 0.5.1's
/// identifier (`85e08846…`, now in `alice_zip::law::PREVIOUS_SEMANTICS_IDS`) as its own:
/// passing that identifier to `law_id` still gives it, so the move to alice-zip's own
/// identifier changed the identifier input only
#[test]
fn law_id_digest_under_the_previous_zip_identifier_is_unchanged() {
    let previous = alice_zip::law::PREVIOUS_SEMANTICS_IDS
        .iter()
        .find(|id| {
            hex(&id[..]) == "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"
        })
        .expect("alice-zip keeps the identifier it published up to 0.10 as a previous one");
    let law = line_law();
    let mut bits = Bits::default();
    bits.raw(&law.law_id(previous));
    for i in 0..=40_u32 {
        let x = f64::from(i) / 10.0;
        bits.f64(law.evaluate(x).expect("inside the domain"));
    }
    assert_golden(
        "law_id and evaluate under the previous alice-zip identifier",
        bits,
        32 + 41 * 8,
        "7269c5962d23f95f4488a3e316630d880dc58446d7d085d10ebcae7b1ddb3bb8",
    );
}

/// The digest recorded before alice-det-math 0.5.1, reproduced with the
/// identifier of 0.5.0 (`135c407d…`, kept in `PREVIOUS_SEMANTICS_IDS`): the
/// re-recording of the pin above moved the identifier input only
#[test]
fn law_id_digest_under_the_previous_identifier_is_unchanged() {
    let previous = alice_det_math::PREVIOUS_SEMANTICS_IDS
        .iter()
        .find(|id| {
            hex(&id[..]) == "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"
        })
        .expect("alice-det-math keeps the identifier of 0.5.0 as a previous one");
    let law = line_law();
    let mut bits = Bits::default();
    bits.raw(&law.law_id(previous));
    for i in 0..=40_u32 {
        let x = f64::from(i) / 10.0;
        bits.f64(law.evaluate(x).expect("inside the domain"));
    }
    assert_golden(
        "law_id and evaluate under the previous identifier",
        bits,
        32 + 41 * 8,
        "1943c25822640df80db50589039bced433cf6a55f02df5862e0f7a414eb5e7b7",
    );
}

// ---------------------------------------------------------------------------
// 3. The transcendental paths
//
// These are the ones that actually differed between builds before the move to
// `alice-det-math`.
//
// The two sinusoid digests below were re-recorded in 0.7.0, when the
// reconstruction laws moved from `f32` accumulation to `f64` accumulation
// rounded once on return. A digest is a change detector, so re-recording one
// needs a reason outside the digest itself: the laws were implemented twice
// (here over arrays, and again in a consumer that answered single-sample
// queries), the two disagreed on 788 / 1024 samples for a sine and 962 / 1024
// for a Fourier reconstruction, and against the `f64` closed form the `f32`
// accumulation was off by 9.16e-7 against the other's 5.95e-8. The array
// generators are now a `map` over the point evaluator, which
// `tests/law_single_source.rs` pins bit for bit alongside the error against
// the closed form.
//
// alice-det-math 0.5.0 changed `SEMANTICS_ID` (d2209b30… → 135c407d…), so
// the two pins above that hold it were re-recorded in 0.10.0. The evaluation
// bits of `law_id_and_evaluation_are_the_recorded_bits` did not move: under
// 0.5.0 the same scenario with the old identifier passed to `law_id` hashes to
// the old digest 06f9beb9…; the digests of the transcendental paths below are
// unchanged as well.
//
// alice-det-math 0.5.1 changed `SEMANTICS_ID` again (135c407d… → 85e08846…;
// 0.5.1 lists 135c407d… in `PREVIOUS_SEMANTICS_IDS`), so the same two pins were
// re-recorded. Again only the identifier input moved: under 0.5.1 the scenario
// with 135c407d… passed to `law_id` still hashes to the old digest 1943c258…
// (`law_id_digest_under_the_previous_identifier_is_unchanged` below), and
// the fitted domain, coefficients and evaluation bits of `line_law` are the
// same bits as before.
//
// 0.11.0 replaced the re-exported alice-det-math identifier by alice-zip's own
// (`src/semantics.rs`, 85e08846… -> 65ba78c1…), so the law identifier pin was
// re-recorded once more (7269c596… -> b8a5064b…); with 85e08846… passed to `law_id`
// the old digest comes back (`law_id_digest_under_the_previous_zip_identifier_is_unchanged`).
//
// `polynomial_generation_is_the_recorded_bits` and
// `law_id_and_evaluation_are_the_recorded_bits` are unchanged: the polynomial
// law already evaluated in `f64` through `horner_ascending`, so making it a
// `map` over `polynomial_at` moved no bits and left every `law_id` intact.
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
        "96898e3223e8c8d49cdd2a6b49df652ab48347c45df5922237783e10a73a290d",
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
        "f354146d062704960e408e1e5c8c54872a636d8deb42d8f1704caf6822e5ec4a",
    );
}

/// The `fft` feature's spectrum: the in-crate `f64` transform (radix-2 for 8, 64 and 1024,
/// Bluestein for 31, 97 and 1000), each bin rounded to `f32` and turned into magnitude and
/// phase. Recorded in 0.11.0, when the transform replaced rustfft, whose butterflies differ
/// per target; this digest is the same on every target of the CI matrix.
#[cfg(feature = "fft")]
#[test]
fn fft_spectrum_is_the_recorded_bits() {
    use alice_zip::generators::analyze_signal_fft;
    let mut bits = Bits::default();
    let lengths = [8usize, 31, 64, 97, 1000, 1024];
    for n in lengths {
        let (coefficients, dc) = analyze_signal_fft(&integer_signal(n), n, 1.0);
        bits.u64(coefficients.len() as u64);
        for &(k, magnitude, phase) in &coefficients {
            bits.u64(k as u64);
            bits.f32(magnitude);
            bits.f32(phase);
        }
        bits.f32(dc);
    }
    assert_golden(
        "analyze_signal_fft",
        bits,
        lengths.iter().sum::<usize>() * 4,
        "a83cfb9c62c3c79379fa1d0df8b5d7aa2b5d98ec978ed7a064cd95524cb5c0cc",
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

// ---------------------------------------------------------------------------
// The checker's own canaries: `assert_golden` must reject a run that hashed
// fewer bytes than its floor and a digest that differs from the recorded one.
// The accepting case comes first, so the two rejections cannot pass by
// panicking for another reason.
// ---------------------------------------------------------------------------

/// SHA-256 of eight zero bytes (`Bits::u64(0)` writes 0 as 8 big-endian
/// bytes), computed outside this crate:
/// `head -c 8 /dev/zero | shasum -a 256`
const SHA256_OF_8_ZERO_BYTES: &str =
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc";

fn eight_zero_bytes() -> Bits {
    let mut bits = Bits::default();
    bits.u64(0);
    bits
}

#[test]
fn checker_accepts_the_reference_bits() {
    assert_golden("canary", eight_zero_bytes(), 8, SHA256_OF_8_ZERO_BYTES);
}

#[test]
#[should_panic(expected = "canary: hashed 8 bytes, expected at least 9")]
fn checker_rejects_too_few_bytes() {
    assert_golden("canary", eight_zero_bytes(), 9, SHA256_OF_8_ZERO_BYTES);
}

#[test]
#[should_panic(expected = "canary: 8 bytes hashed to af5570f5a1810b7a")]
fn checker_rejects_a_wrong_digest() {
    let wrong = "0".repeat(64);
    assert_golden("canary", eight_zero_bytes(), 8, &wrong);
}
