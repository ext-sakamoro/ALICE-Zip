//! Bit-level golden for the sign of a zero picked by a minimum or maximum
//!
//! `f64::min` / `f64::max` / `f32::min` / `f32::max` leave the result
//! unspecified when the two operands are `-0.0` and `+0.0`, and targets
//! differ: measured on 2026-10-15 with `x = [-0.0, 0.0, 1.0]`, the fitted
//! `domain.lo` was `-0.0` on aarch64 and `+0.0` on x86_64, which changed
//! `law_id` and the sign of `evaluate(-0.0)`. `quantize::range` had the same
//! split for `[-0.0, 0.0]`.
//!
//! Both now take the minimum and maximum from `alice-det-math`, whose zero sign
//! is defined (`min(-0.0, 0.0) = -0.0`, `max(-0.0, 0.0) = +0.0` in either
//! order). The expected values below are the same on every target and in both
//! operand orders, so reverting one site to the platform comparison turns this
//! file red on the target whose choice differs.
//!
//! `law_id` is pinned under a fixed semantics identifier, so a change of the
//! arithmetic identifier of `alice-det-math` does not move these pins; the
//! pins that hold that identifier live in `tests/determinism_golden.rs`.
//!
//! Every input goes through `std::hint::black_box` (`zero_law` and `opaque`).
//! The inputs are literals, so without it a release build may constant-fold
//! the comparison and never run the `x86_64` `minsd`/`maxsd` instruction
//! (which returns the second operand on a zero tie, unlike the aarch64
//! `fminnm`/`fmaxnm`); a revert to the platform comparison would then pass in
//! release.

use alice_zip::law::{Provenance, SignalLaw};
use alice_zip::quantize::{quantize_16bit, quantize_8bit};
use std::hint::black_box;

const SEMANTICS: [u8; 32] = [0x11; 32];
const NEG_ZERO: u64 = 0x8000_0000_0000_0000;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `y = 0` fitted with degree 1 over `xs`, so the coefficients are zeros and
/// the sign of `evaluate(-0.0)` depends on `domain.lo`
fn zero_law(xs: &[f64]) -> SignalLaw {
    let points: Vec<(f64, f64)> = xs.iter().map(|&x| (black_box(x), 0.0)).collect();
    SignalLaw::fit_polynomial(&points, 1, Provenance::new("golden", "least squares"))
        .expect("the fixture satisfies every fit precondition")
}

/// The samples, each passed through `black_box`, so a release build cannot
/// see the literal values through the quantiser
fn opaque(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&v| black_box(v)).collect()
}

const LAW_ID_LO_ZERO: &str = "57eebce292f044cc965024d36ad39abd1a88f6dc4c9f9dedfcddcd93128c9b01";
const LAW_ID_HI_ZERO: &str = "7e55ad5f856be004d92a7942a9a57417c5e1b0900d4ed95ec39ca4a236410213";

#[test]
fn fitted_lo_of_mixed_zeros_is_negative_zero_in_both_orders() {
    for xs in [[-0.0, 0.0, 1.0], [0.0, -0.0, 1.0]] {
        let law = zero_law(&xs);
        let d = law.domain();
        assert_eq!(d.lo.to_bits(), NEG_ZERO, "lo for {xs:?}");
        assert_eq!(d.hi.to_bits(), 1.0_f64.to_bits(), "hi for {xs:?}");
        assert_eq!(
            hex(&law.law_id(&SEMANTICS)),
            LAW_ID_LO_ZERO,
            "law_id for {xs:?}"
        );
        let y = law.evaluate(-0.0).expect("-0.0 is inside the domain");
        assert_eq!(y.to_bits(), 0, "evaluate(-0.0) for {xs:?}");
    }
}

#[test]
fn fitted_hi_of_mixed_zeros_is_positive_zero_in_both_orders() {
    for xs in [[-1.0, -0.0, 0.0], [-1.0, 0.0, -0.0]] {
        let law = zero_law(&xs);
        let d = law.domain();
        assert_eq!(d.lo.to_bits(), (-1.0_f64).to_bits(), "lo for {xs:?}");
        assert_eq!(d.hi.to_bits(), 0, "hi for {xs:?}");
        assert_eq!(
            hex(&law.law_id(&SEMANTICS)),
            LAW_ID_HI_ZERO,
            "law_id for {xs:?}"
        );
        let y = law.evaluate(-0.0).expect("-0.0 is inside the domain");
        assert_eq!(y.to_bits(), 0, "evaluate(-0.0) for {xs:?}");
    }
}

#[test]
fn quantize_min_of_mixed_zeros_is_negative_zero_in_both_orders() {
    for data in [
        &[-0.0_f32, 0.0][..],
        &[0.0, -0.0][..],
        &[-0.0, 0.0, 1.0][..],
        &[0.0, -0.0, 1.0][..],
    ] {
        let (_, min8, scale8) = quantize_8bit(&opaque(data));
        let (_, min16, scale16) = quantize_16bit(&opaque(data));
        assert_eq!(min8.to_bits(), NEG_ZERO, "8-bit min for {data:?}");
        assert_eq!(min16.to_bits(), NEG_ZERO, "16-bit min for {data:?}");
        assert_eq!(
            scale8.to_bits(),
            1.0_f64.to_bits(),
            "8-bit scale for {data:?}"
        );
        assert_eq!(
            scale16.to_bits(),
            1.0_f64.to_bits(),
            "16-bit scale for {data:?}"
        );
    }
}

#[test]
fn quantize_bytes_of_mixed_zeros_are_unchanged() {
    assert_eq!(hex(&quantize_8bit(&opaque(&[-0.0, 0.0, 1.0])).0), "0000ff");
    assert_eq!(hex(&quantize_8bit(&opaque(&[0.0, -0.0, 1.0])).0), "0000ff");
    assert_eq!(
        hex(&quantize_16bit(&opaque(&[-0.0, 0.0, 1.0])).0),
        "00000000ffff"
    );
    assert_eq!(
        hex(&quantize_16bit(&opaque(&[0.0, -0.0, 1.0])).0),
        "00000000ffff"
    );
}
