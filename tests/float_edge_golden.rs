//! Bit-exact goldens for the float edge cases of the public law functions:
//! signed zero and catastrophic cancellation.
//!
//! A tolerance-based oracle cannot see these: `-0.0 == +0.0`, and a
//! cancellation that loses its last bit is still "close". The law is the bits,
//! so each case is pinned with `to_bits()`. Every expected value was computed
//! once, then checked by hand against the exact real-number result (and, where
//! the exact result is not representable, against IEEE 754 round-to-nearest,
//! ties-to-even). Every case runs in both the debug and the release profile
//! (`.github/workflows/release-profile.yml`), so an optimisation that
//! reassociates, contracts to FMA or folds a signed zero turns this red.
//!
//! Functions covered: `polynomial_at`, `fit_polynomial`, `multi_sine_at`,
//! `fourier_at`, `quantize_8bit` / `quantize_16bit` (+ dequantisation),
//! `SignalLaw::evaluate`, `SignalLaw::residual` and `SignalLaw::ingest`.

use alice_zip::generators::{fit_polynomial, fourier_at, multi_sine_at, polynomial_at};
use alice_zip::law::{
    IngestPolicy, Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange, Verdict,
};
use alice_zip::quantize::{dequantize_16bit, dequantize_8bit, quantize_16bit, quantize_8bit};

const F32_POS_ZERO: u32 = 0x0000_0000;
const F32_NEG_ZERO: u32 = 0x8000_0000;
const F64_POS_ZERO: u64 = 0x0000_0000_0000_0000;
const F64_NEG_ZERO: u64 = 0x8000_0000_0000_0000;
/// `1.0_f64`
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;

/// A law over `[lo, 1]` restored from its parts, with just enough evidence
fn law(coefficients: Vec<f64>, lo: f64) -> SignalLaw {
    law_with_evidence(coefficients, lo, vec![(lo, 0.0), (1.0, 0.0)])
}

fn law_with_evidence(coefficients: Vec<f64>, lo: f64, evidence: Vec<(f64, f64)>) -> SignalLaw {
    SignalLaw::from_parts(SignalLawParts {
        coefficients,
        domain: ValidRange { lo, hi: 1.0 },
        evidence,
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new("float edge golden", "hand-written"),
        oracles: Vec::new(),
    })
    .expect("the fixture satisfies every from_parts precondition")
}

/// Distance in units in the last place between two finite `f64` of the same
/// sign: an independent bound for the pins below that record a QR rounding
/// "as computed" (the bound comes from the exact answer, not from the
/// implementation)
fn ulps_apart(a: f64, b: f64) -> u64 {
    assert!(a.is_finite() && b.is_finite() && a.is_sign_negative() == b.is_sign_negative());
    a.to_bits().abs_diff(b.to_bits())
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// polynomial_at (Horner in f64, one rounding to f32)
// ---------------------------------------------------------------------------

/// `-3 + 1.5·2` and `3 + (-1.5)·2`: the last step adds two exactly opposite
/// values, which IEEE round-to-nearest defines as `+0.0` whichever operand is
/// negative.
#[test]
fn polynomial_at_exact_cancellation_is_positive_zero_in_both_orders() {
    assert_eq!(polynomial_at(&[-3.0, 1.5], 2.0).to_bits(), F32_POS_ZERO);
    assert_eq!(polynomial_at(&[3.0, -1.5], 2.0).to_bits(), F32_POS_ZERO);
}

/// At `x = -0.0`, `1·(-0.0) = -0.0`; adding the constant keeps `-0.0` only if
/// the constant is `-0.0` too (`-0.0 + -0.0 = -0.0`, `-0.0 + +0.0 = +0.0`).
#[test]
fn polynomial_at_negative_zero_input_keeps_sign_only_with_negative_zero_constant() {
    assert_eq!(polynomial_at(&[-0.0, 1.0], -0.0).to_bits(), F32_NEG_ZERO);
    assert_eq!(polynomial_at(&[0.0, 1.0], -0.0).to_bits(), F32_POS_ZERO);
}

/// `-1e16 + 1·(1e16 + 2)`: both terms are exact in f64 (spacing 2 above
/// 2^53), so the survivor of the cancellation is exactly `2.0`.
#[test]
fn polynomial_at_large_opposite_terms_leave_exact_survivor_two() {
    assert_eq!(
        polynomial_at(&[-1e16, 1.0], 1e16 + 2.0).to_bits(),
        0x4000_0000
    );
}

// ---------------------------------------------------------------------------
// SignalLaw::evaluate (f64, u = (x - lo) / (hi - lo), then Horner)
// ---------------------------------------------------------------------------

/// `1 + 1e16·1 = 10000000000000001` lies exactly halfway between the f64
/// neighbours `1e16` and `1e16 + 2`; ties-to-even gives `1e16` (even
/// significand), and symmetrically `-1e16` for `1 - 1e16`. The `1` is lost by
/// the rounding rule, not by an error.
#[test]
fn law_evaluate_absorbs_unit_term_by_ties_to_even() {
    let up = law(vec![1.0, 1e16], 0.0).evaluate(1.0).expect("in domain");
    let down = law(vec![1.0, -1e16], 0.0).evaluate(1.0).expect("in domain");
    assert_eq!(up.to_bits(), 0x4341_c379_37e0_8000); // 1e16
    assert_eq!(down.to_bits(), 0xc341_c379_37e0_8000); // -1e16
}

/// `c0 = -fl(0.1·0.3)`, `c1 = 0.1`, `u = 0.3`: separate multiply and add round
/// `0.1·0.3` to exactly `-c0`, so the law's value is `+0.0`. A fused
/// multiply-add would keep the product's rounding error and return
/// `1.6653345369377347e-18` instead — this is the case that tells the law's
/// arithmetic from FMA (in f64 via `evaluate`, and in f32 via `polynomial_at`).
#[test]
fn horner_cancellation_is_positive_zero_not_the_fma_rounding_error() {
    let product = 0.1_f64 * 0.3;
    assert_eq!(product.to_bits(), 0x3f9e_b851_eb85_1eb8);
    let coeffs = vec![-product, 0.1];
    let y = law(coeffs.clone(), 0.0).evaluate(0.3).expect("in domain");
    assert_eq!(y.to_bits(), F64_POS_ZERO);
    assert_eq!(polynomial_at(&coeffs, 0.3).to_bits(), F32_POS_ZERO);
}

/// `1e8 - 1e8·1` and `0.1 - 0.1·1`: exact cancellation to `+0.0` (the second
/// one also for a coefficient that is not exactly representable).
#[test]
fn law_evaluate_exact_cancellation_is_positive_zero() {
    let a = law(vec![1e8, -1e8], 0.0).evaluate(1.0).expect("in domain");
    let b = law(vec![0.1, -0.1], 0.0).evaluate(1.0).expect("in domain");
    assert_eq!(a.to_bits(), F64_POS_ZERO);
    assert_eq!(b.to_bits(), F64_POS_ZERO);
}

/// `x = -0.0` on the domain `[+0.0, 1]`: `u = (-0.0 - 0.0) / 1 = -0.0`.
/// - `[-0.0, 1]`: `1·u = -0.0`, `+ (-0.0)` = `-0.0`
/// - `[+0.0, 1]`: `-0.0 + 0.0` = `+0.0`
/// - `[-0.0, -1]`: `(-1)·(-0.0) = +0.0`, `+ (-0.0)` = `+0.0`
#[test]
fn law_evaluate_at_negative_zero_input_follows_ieee_sign_rules() {
    let x = -0.0;
    let neg = law(vec![-0.0, 1.0], 0.0).evaluate(x).expect("in domain");
    let pos = law(vec![0.0, 1.0], 0.0).evaluate(x).expect("in domain");
    let flipped = law(vec![-0.0, -1.0], 0.0).evaluate(x).expect("in domain");
    assert_eq!(neg.to_bits(), F64_NEG_ZERO);
    assert_eq!(pos.to_bits(), F64_POS_ZERO);
    assert_eq!(flipped.to_bits(), F64_POS_ZERO);
}

/// With the domain starting at `-0.0`, `u = -0.0 - (-0.0) = +0.0`, so the same
/// law `[-0.0, 1]` that gave `-0.0` above now gives `+0.0` at `x = -0.0`:
/// the sign of `domain.lo` reaches the result, which is why it is in `law_id`.
#[test]
fn law_evaluate_negative_zero_domain_start_turns_input_sign_positive() {
    let y = law(vec![-0.0, 1.0], -0.0)
        .evaluate(-0.0)
        .expect("in domain");
    assert_eq!(y.to_bits(), F64_POS_ZERO);
}

// ---------------------------------------------------------------------------
// fit_polynomial (Householder QR on x/(n-1), coefficients mapped back)
// ---------------------------------------------------------------------------

/// `y = 1e8 + 8·i` for `i = 0..16` (every value exact in f32): the slope is
/// `8e-8` of the offset. The fit recovers the offset exactly and the slope
/// 2 ulp below `8` (rounding of the QR, relative error 2.2e-16), with zero
/// normalised error at degree 1.
#[test]
fn fit_polynomial_large_offset_tiny_slope_bits() {
    #[allow(clippy::cast_precision_loss)]
    let ys: Vec<f32> = (0..16).map(|i| 1e8_f32 + 8.0 * i as f32).collect();
    let (coeffs, degree, err) = fit_polynomial(&ys, 3, 1e-9).expect("a line fits");
    assert_eq!(degree, 1);
    assert_eq!(err.to_bits(), F64_POS_ZERO);
    let bits: Vec<u64> = coeffs.iter().map(|c| c.to_bits()).collect();
    assert_eq!(bits, [0x4197_d784_0000_0000, 0x401f_ffff_ffff_fffe]); // 1e8, 8 - 2 ulp
                                                                      // the exact fit is 1e8 + 8x: the offset is exact and the slope within a few ulp of 8
    assert_eq!(coeffs[0], 1e8);
    assert!(
        ulps_apart(coeffs[1], 8.0) <= 4,
        "slope {} is not within 4 ulp of 8",
        coeffs[1]
    );
}

/// A constant `1e8` series takes the constant branch: the mean is summed in
/// f64 where `16·1e8` is exact, so it is exactly `1e8`, degree 0, error `+0.0`.
#[test]
fn fit_polynomial_constant_large_offset_is_exact_mean() {
    let (coeffs, degree, err) = fit_polynomial(&[1e8_f32; 16], 3, 1e-9).expect("constant");
    assert_eq!(degree, 0);
    assert_eq!(err.to_bits(), F64_POS_ZERO);
    let bits: Vec<u64> = coeffs.iter().map(|c| c.to_bits()).collect();
    assert_eq!(bits, [0x4197_d784_0000_0000]);
}

/// `[1e8, 1e8, 1e8, 1e8 + 8]` needs the interpolating cubic
/// `1e8 + 8/3·x - 4·x² + 4/3·x³`: the signal is `8e-8` of the offset, so the
/// coefficients carry the cancellation of the offset (relative error 1e-9 to
/// 7e-9, consistent with the data's own f64 resolution `1e8·2^-53 / 8`, and
/// within the module's documented `1e-6`). Pinned as computed.
#[test]
fn fit_polynomial_cubic_on_large_offset_carries_cancellation_bits() {
    let (coeffs, degree, err) =
        fit_polynomial(&[1e8, 1e8, 1e8, 1e8 + 8.0], 3, 1e-9).expect("cubic interpolates");
    assert_eq!(degree, 3);
    let exact = [1e8, 8.0 / 3.0, -4.0, 4.0 / 3.0];
    for (c, e) in coeffs.iter().zip(exact) {
        assert!(((c - e) / e).abs() < 1e-8, "{c} vs {e}");
    }
    let bits: Vec<u64> = coeffs.iter().map(|c| c.to_bits()).collect();
    assert_eq!(
        bits,
        [
            0x4197_d783_ffff_ffff, // 1e8 - 1 ulp
            0x4005_5555_57c7_1c50, // 2.666666684879182
            0xc010_0000_00aa_aa9c, // -4.000000009934094
            0x3ff5_5555_55c7_1c66, // 1.3333333349890153
        ]
    );
    assert_eq!(err.to_bits(), 0x3c55_5555_5555_5555); // 4.6e-18
}

// ---------------------------------------------------------------------------
// multi_sine_at / fourier_at (det-math sin64 / cos64, f64 sum, one f32 rounding)
// ---------------------------------------------------------------------------

/// The DC term starts the sum, so with no component a `-0.0` offset survives;
/// `-1·sin(+0.0) = -0.0` keeps it; `sin(-0.0) = -0.0` (odd function, at
/// `i = -0.0` and phase `-0.0`) keeps it. A `+0.0` term flips it to `+0.0`.
#[test]
fn multi_sine_at_signed_zero_offset_and_odd_sine_at_zero() {
    assert_eq!(multi_sine_at(4, &[], -0.0, 0.0).to_bits(), F32_NEG_ZERO);
    assert_eq!(
        multi_sine_at(4, &[(1.0, -1.0, 0.0)], -0.0, 0.0).to_bits(),
        F32_NEG_ZERO
    );
    assert_eq!(
        multi_sine_at(4, &[(1.0, 1.0, -0.0)], -0.0, -0.0).to_bits(),
        F32_NEG_ZERO
    );
    assert_eq!(
        multi_sine_at(4, &[(1.0, 1.0, 0.0)], -0.0, 0.0).to_bits(),
        F32_POS_ZERO
    );
}

/// The zero crossing at `θ = 2π·2/4`: f64 `π` is not the real π, so
/// `sin(fl(π)) = 1.2246467991473532e-16` (the correctly rounded value), not
/// `0`; it rounds to f32 `1.2246469e-16`. Two components that cancel at
/// `θ = π/2` (`sin = 1` exactly) give `+0.0`.
#[test]
fn multi_sine_at_zero_crossing_is_sin_of_rounded_pi() {
    assert_eq!(
        multi_sine_at(4, &[(1.0, 1.0, 0.0)], 0.0, 2.0).to_bits(),
        0x250d_3132
    );
    assert_eq!(
        multi_sine_at(4, &[(1.0, 1.0, 0.0), (1.0, -1.0, 0.0)], 0.0, 1.0).to_bits(),
        F32_POS_ZERO
    );
}

/// `fourier_at` at the zero crossing `θ = π/2` of bin 1 (weight 2, magnitude
/// 4, n = 4): `2·4/4·cos(fl(π/2)) = 2·6.123233995736766e-17`, the correctly
/// rounded cosine, not `0`. `-1 + 2·2/4·cos(0) = -1 + 1` cancels to `+0.0`.
#[test]
fn fourier_at_zero_crossing_and_cancellation_bits() {
    assert_eq!(
        fourier_at(4, &[(1, 4.0, 0.0)], 0.0, 1.0).to_bits(),
        0x250d_3132
    );
    assert_eq!(
        fourier_at(4, &[(1, 2.0, 0.0)], -1.0, 0.0).to_bits(),
        F32_POS_ZERO
    );
}

/// No coefficient (and `n == 0`, which returns the DC term directly) keeps a
/// `-0.0` offset.
#[test]
fn fourier_at_negative_zero_offset_survives_empty_and_degenerate_length() {
    assert_eq!(fourier_at(4, &[], -0.0, 0.0).to_bits(), F32_NEG_ZERO);
    assert_eq!(
        fourier_at(0, &[(1, 4.0, 0.0)], -0.0, 0.0).to_bits(),
        F32_NEG_ZERO
    );
}

// ---------------------------------------------------------------------------
// quantize (max - min = 0 takes the constant-data guard, scale = 1.0)
// ---------------------------------------------------------------------------

/// All-equal data: every sample quantises to 0, `min = 2.5`, `scale = 1.0`,
/// and dequantisation returns exactly `2.5` (8 and 16 bit).
#[test]
fn quantize_all_equal_data_uses_unit_scale_and_round_trips_exactly() {
    let data = [2.5_f32; 4];
    let (q8, min8, scale8) = quantize_8bit(&data);
    let (q16, min16, scale16) = quantize_16bit(&data);
    assert_eq!(q8, [0; 4]);
    assert_eq!(q16, [0; 8]);
    for (min, scale) in [(min8, scale8), (min16, scale16)] {
        assert_eq!(min.to_bits(), 0x4004_0000_0000_0000); // 2.5
        assert_eq!(scale.to_bits(), F64_ONE);
    }
    assert_eq!(
        f32_bits(&dequantize_8bit(&q8, min8, scale8)),
        [0x4020_0000; 4]
    );
    assert_eq!(
        f32_bits(&dequantize_16bit(&q16, min16, scale16)),
        [0x4020_0000; 4]
    );
}

/// All `-0.0`: `min = max = -0.0` (no `+0.0` to compare with), `scale = 1.0`,
/// and dequantisation gives `+0.0` (`0·1 + (-0.0) = +0.0 + -0.0 = +0.0`).
#[test]
fn quantize_all_negative_zero_keeps_min_sign_and_dequantises_to_positive_zero() {
    let data = [-0.0_f32; 3];
    let (q8, min8, scale8) = quantize_8bit(&data);
    let (q16, min16, scale16) = quantize_16bit(&data);
    assert_eq!(q8, [0; 3]);
    assert_eq!(q16, [0; 6]);
    for (min, scale) in [(min8, scale8), (min16, scale16)] {
        assert_eq!(min.to_bits(), F64_NEG_ZERO);
        assert_eq!(scale.to_bits(), F64_ONE);
    }
    assert_eq!(
        f32_bits(&dequantize_8bit(&q8, min8, scale8)),
        [F32_POS_ZERO; 3]
    );
    assert_eq!(
        f32_bits(&dequantize_16bit(&q16, min16, scale16)),
        [F32_POS_ZERO; 3]
    );
}

/// `[-0.0, +0.0]` in both orders: bytes, scale and the dequantised samples are
/// pinned. The **sign of the returned `min` is not pinned**: it comes from
/// `f32::min` on two equal zeros, for which Rust does not specify which
/// operand is returned, so only its value (`0`) is asserted here.
#[test]
fn quantize_mixed_signed_zeros_pins_bytes_scale_and_reconstruction() {
    for data in [[-0.0_f32, 0.0], [0.0, -0.0]] {
        let (q8, min8, scale8) = quantize_8bit(&data);
        let (q16, min16, scale16) = quantize_16bit(&data);
        assert_eq!(q8, [0; 2]);
        assert_eq!(q16, [0; 4]);
        for (min, scale) in [(min8, scale8), (min16, scale16)] {
            assert_eq!(min, 0.0);
            assert_eq!(scale.to_bits(), F64_ONE);
        }
        assert_eq!(
            f32_bits(&dequantize_8bit(&q8, min8, scale8)),
            [F32_POS_ZERO; 2]
        );
        assert_eq!(
            f32_bits(&dequantize_16bit(&q16, min16, scale16)),
            [F32_POS_ZERO; 2]
        );
    }
}

// ---------------------------------------------------------------------------
// Residual RMS
// ---------------------------------------------------------------------------

/// A law `1 + 2u` restored with evidence it reproduces exactly: every residual
/// is `+0.0`, so `rms = sqrt(0/3) = +0.0` and `max_abs = +0.0`; new exact
/// evidence is judged `Supports { rms: +0.0 }`.
#[test]
fn residual_rms_of_exact_evidence_is_positive_zero() {
    let l = law_with_evidence(
        vec![1.0, 2.0],
        0.0,
        vec![(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)],
    );
    assert_eq!(l.residual().rms.to_bits(), F64_POS_ZERO);
    assert_eq!(l.residual().max_abs.to_bits(), F64_POS_ZERO);
    let policy = IngestPolicy {
        abs_tolerance: 0.0,
        break_factor: 2.0,
    };
    match l.ingest(&[(0.25, 1.5)], &policy) {
        Verdict::Supports { rms } => assert_eq!(rms.to_bits(), F64_POS_ZERO),
        other => panic!("expected Supports, got {other:?}"),
    }
}

/// The same line fitted by least squares: the constant comes out 1 ulp below
/// `1` (QR rounding), so the residual is not zero but a pinned `6.4e-17` RMS
/// with `max_abs = 2^-53`. A fit far from the origin (`x ≈ 1e8`) works in
/// `u ∈ [0, 1]` and gives the same bits.
#[test]
fn residual_rms_of_least_squares_line_is_pinned_rounding() {
    let near = SignalLaw::fit_polynomial(
        &[(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)],
        1,
        Provenance::new("a", "b"),
    )
    .expect("fits");
    let far = SignalLaw::fit_polynomial(
        &[(1e8, 1.0), (1e8 + 1.0, 2.0), (1e8 + 2.0, 3.0)],
        1,
        Provenance::new("a", "b"),
    )
    .expect("fits");
    for l in [&near, &far] {
        let bits: Vec<u64> = l.coefficients().iter().map(|c| c.to_bits()).collect();
        assert_eq!(bits, [0x3fef_ffff_ffff_ffff, 0x4000_0000_0000_0000]);
        assert_eq!(l.residual().rms.to_bits(), 0x3c92_79a7_4590_331c);
        assert_eq!(l.residual().max_abs.to_bits(), 0x3ca0_0000_0000_0000);
        // the exact fit is 1 + 2u: each coefficient within 2 ulp of it, and the
        // residual of 3 exact samples bounded by 2 ulp of the largest value 3
        let c = l.coefficients();
        assert!(
            ulps_apart(c[0], 1.0) <= 2,
            "constant {} is not within 2 ulp of 1",
            c[0]
        );
        assert!(
            ulps_apart(c[1], 2.0) <= 2,
            "slope {} is not within 2 ulp of 2",
            c[1]
        );
        let bound = 2.0 * f64::EPSILON * 3.0;
        assert!(l.residual().max_abs <= bound && l.residual().rms <= l.residual().max_abs);
    }
    assert_eq!(
        far.evaluate(1e8 + 1.0).expect("in domain").to_bits(),
        0x4000_0000_0000_0000
    );
}
