//! Oracle tests for `alice_zip::law` — a fitted law plus its residual, valid
//! range, provenance and oracle cases, and the verdict on new evidence
//!
//! Every expected value comes from a closed form written in the test (the
//! polynomial the samples were drawn from), never from the implementation.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use alice_zip::law::{
    IngestPolicy, LawError, OracleCase, Provenance, SignalLaw, ValidRange, Verdict,
};

fn quad(x: f64) -> f64 {
    // oracle: y = 2 - 3x + 0.5x²
    2.0 - 3.0 * x + 0.5 * x * x
}

fn samples(f: fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            (x, f(x))
        })
        .collect()
}

fn prov() -> Provenance {
    Provenance::new("synthetic: y = 2 - 3x + 0.5x^2", "least squares, degree 2")
}

#[test]
fn exact_polynomial_is_recovered_and_evaluated_at_new_conditions() {
    let law = SignalLaw::fit_polynomial(&samples(quad, -2.0, 5.0, 37), 2, prov()).unwrap();
    assert_eq!(law.degree(), 2);
    assert_eq!(law.domain(), ValidRange { lo: -2.0, hi: 5.0 });
    // re-computation at conditions that were never sampled
    for x in [-1.97, -0.3, 1.3, 2.71, 4.999] {
        let y = law.evaluate(x).unwrap();
        assert!(
            (y - quad(x)).abs() <= 1e-11 * (1.0 + quad(x).abs()),
            "x={x} y={y}"
        );
    }
    let r = law.residual();
    assert_eq!(r.n, 37);
    assert!(r.rms <= 1e-12 && r.max_abs <= 1e-11, "{r:?}");
}

#[test]
fn residual_is_measured_against_the_evidence_not_reported_by_the_fit() {
    // y = x on [0, 1] fitted with degree 0: the best constant is 0.5 and the
    // residual of a straight line about its mean is known in closed form
    let pts = samples(|x| x, 0.0, 1.0, 101);
    let law = SignalLaw::fit_polynomial(&pts, 0, prov()).unwrap();
    assert!((law.evaluate(0.2).unwrap() - 0.5).abs() < 1e-12);
    let r = law.residual();
    assert!((r.max_abs - 0.5).abs() < 1e-12, "{r:?}");
    // rms of x - 0.5 over 101 evenly spaced points: sqrt(Σ(i/100 - 0.5)² / 101)
    let expect = ((0..=100)
        .map(|i| (i as f64 / 100.0 - 0.5).powi(2))
        .sum::<f64>()
        / 101.0)
        .sqrt();
    assert!((r.rms - expect).abs() < 1e-12, "{} vs {expect}", r.rms);
}

#[test]
fn evaluation_outside_the_valid_range_is_refused_not_extrapolated() {
    let law = SignalLaw::fit_polynomial(&samples(quad, -2.0, 5.0, 37), 2, prov()).unwrap();
    assert_eq!(law.evaluate(5.000_001), Err(LawError::OutOfRange));
    assert_eq!(law.evaluate(-2.000_001), Err(LawError::OutOfRange));
    assert_eq!(law.evaluate(f64::NAN), Err(LawError::OutOfRange));
    assert_eq!(law.evaluate(f64::INFINITY), Err(LawError::OutOfRange));
    // the closed interval includes both ends
    assert!(law.evaluate(-2.0).is_ok() && law.evaluate(5.0).is_ok());
}

#[test]
fn oracle_cases_report_pass_fail_and_out_of_range() {
    let law = SignalLaw::fit_polynomial(&samples(quad, -2.0, 5.0, 37), 2, prov())
        .unwrap()
        .with_oracle(OracleCase::new(
            1.0,
            quad(1.0),
            1e-9,
            "closed form at x = 1",
        ))
        .with_oracle(OracleCase::new(
            3.0,
            quad(3.0) + 0.1,
            1e-3,
            "a value 0.1 off",
        ))
        .with_oracle(OracleCase::new(
            9.0,
            quad(9.0),
            1e-9,
            "outside the fitted range",
        ));
    let out = law.check_oracles();
    assert_eq!(out.len(), 3);
    assert!(out[0].passed && out[0].value.is_ok());
    assert!(!out[1].passed);
    assert!((out[1].error.unwrap() - 0.1).abs() < 1e-9);
    assert!(!out[2].passed && out[2].value == Err(LawError::OutOfRange));
}

// ---- verdicts on new evidence -------------------------------------------------

/// A law fitted to three points of y = x², one of them measured 0.3 too high:
/// the interpolant is y = 0.7x² + 0.3
fn law_from_three_points() -> SignalLaw {
    SignalLaw::fit_polynomial(&[(-1.0, 1.0), (0.0, 0.3), (1.0, 1.0)], 2, prov()).unwrap()
}

fn policy() -> IngestPolicy {
    IngestPolicy {
        abs_tolerance: 0.05,
        break_factor: 4.0,
    }
}

#[test]
fn three_point_law_is_the_interpolant() {
    let law = law_from_three_points();
    for x in [-1.0, -0.4, 0.0, 0.25, 1.0] {
        assert!((law.evaluate(x).unwrap() - (0.7 * x * x + 0.3)).abs() < 1e-12);
    }
}

#[test]
fn evidence_that_agrees_supports_the_law() {
    let law = law_from_three_points();
    let v = law.ingest(&samples(|x| 0.7 * x * x + 0.3, -1.0, 1.0, 50), &policy());
    match v {
        Verdict::Supports { rms } => assert!(rms < 1e-12, "{rms}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn evidence_from_the_same_form_with_other_parameters_updates_the_parameters() {
    // 200 exact samples of y = x²: the old law misses them by 0.3 - 0.3x²
    // (rms ≈ 0.22 > 0.05), but one quadratic fits all 203 points with
    // rms ≈ 0.3 / sqrt(203)·(...) < 0.05
    let law = law_from_three_points();
    let v = law.ingest(&samples(|x| x * x, -1.0, 1.0, 200), &policy());
    match v {
        Verdict::ParameterUpdate {
            previous_rms,
            updated,
        } => {
            assert!(previous_rms > 0.05, "{previous_rms}");
            assert!(updated.residual().rms <= 0.05, "{:?}", updated.residual());
            assert_eq!(updated.residual().n, 203);
            // the updated law is close to y = x² away from the bad point
            assert!((updated.evaluate(0.9).unwrap() - 0.81).abs() < 0.01);
            assert_eq!(updated.domain(), ValidRange { lo: -1.0, hi: 1.0 });
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn noise_beyond_the_band_but_not_a_new_form_is_a_grown_residual() {
    // ±0.08 alternating about the law: rms 0.08 ∈ (0.05, 0.2], and a refit of
    // the same form cannot remove alternating noise
    let law = law_from_three_points();
    let pts: Vec<(f64, f64)> = samples(|x| 0.7 * x * x + 0.3, -1.0, 1.0, 200)
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| (x, if i % 2 == 0 { y + 0.08 } else { y - 0.08 }))
        .collect();
    match law.ingest(&pts, &policy()) {
        Verdict::ResidualGrew { rms } => assert!((rms - 0.08).abs() < 1e-9, "{rms}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn evidence_of_another_form_breaks_the_law() {
    // y = 3x³ cannot be expressed by any quadratic on [-1, 1]
    let law = law_from_three_points();
    match law.ingest(&samples(|x| 3.0 * x * x * x, -1.0, 1.0, 100), &policy()) {
        Verdict::Breaks { rms } => assert!(rms > 0.2, "{rms}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn evidence_outside_the_valid_range_is_not_judged() {
    let law = law_from_three_points();
    let v = law.ingest(&[(0.5, 0.475), (2.0, 4.0), (-3.0, 9.0)], &policy());
    assert_eq!(v, Verdict::OutOfRange { outside: 2 });
}

// ---- degenerate input ---------------------------------------------------------

#[test]
fn degenerate_input_is_an_error_not_a_panic() {
    // empty, too few points for the degree, a single x, non-finite values
    assert_eq!(
        SignalLaw::fit_polynomial(&[], 1, prov()).unwrap_err(),
        LawError::TooFewPoints
    );
    assert_eq!(
        SignalLaw::fit_polynomial(&[(0.0, 1.0), (1.0, 2.0)], 2, prov()).unwrap_err(),
        LawError::TooFewPoints
    );
    assert_eq!(
        SignalLaw::fit_polynomial(&[(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)], 1, prov()).unwrap_err(),
        LawError::DegenerateDomain
    );
    assert_eq!(
        SignalLaw::fit_polynomial(&[(0.0, 1.0), (f64::NAN, 2.0), (1.0, 3.0)], 1, prov())
            .unwrap_err(),
        LawError::NonFinite
    );
    assert_eq!(
        SignalLaw::fit_polynomial(&[(0.0, 1.0), (0.5, f64::INFINITY), (1.0, 3.0)], 1, prov())
            .unwrap_err(),
        LawError::NonFinite
    );
    // three distinct x cannot determine a quadratic if two of them coincide
    assert_eq!(
        SignalLaw::fit_polynomial(&[(0.0, 1.0), (0.0, 2.0), (1.0, 3.0)], 2, prov()).unwrap_err(),
        LawError::RankDeficient
    );
    // empty evidence is out of range of nothing: it supports nothing either
    let law = law_from_three_points();
    assert_eq!(law.ingest(&[], &policy()), Verdict::NoEvidence);
}

#[test]
fn provenance_is_kept_with_the_law() {
    let law = law_from_three_points();
    assert_eq!(law.provenance().source, "synthetic: y = 2 - 3x + 0.5x^2");
    assert_eq!(law.provenance().method, "least squares, degree 2");
    assert_eq!(law.evidence().len(), 3);
}

// ---- parts: storing and restoring a law ---------------------------------------

#[test]
fn a_law_survives_a_round_trip_through_its_parts() {
    let law = law_from_three_points().with_oracle(OracleCase::new(
        0.0,
        0.3,
        1e-9,
        "the measured middle point",
    ));
    let parts = law.to_parts();
    assert_eq!(parts.coefficients.len(), 3);
    assert_eq!(parts.domain, ValidRange { lo: -1.0, hi: 1.0 });
    let back = SignalLaw::from_parts(parts.clone()).unwrap();
    assert_eq!(back, law);
    // the residual is re-measured from the evidence, not trusted from the parts
    let mut forged = parts;
    forged.residual.rms = 0.0;
    forged.evidence[1].1 = 0.9;
    let back = SignalLaw::from_parts(forged).unwrap();
    assert!(back.residual().rms > 0.1, "{:?}", back.residual());
    assert_eq!(law.coefficients(), law.to_parts().coefficients.as_slice());
}

#[test]
fn parts_that_do_not_describe_a_law_are_rejected() {
    let good = law_from_three_points().to_parts();
    let mut p = good.clone();
    p.coefficients.clear();
    assert_eq!(
        SignalLaw::from_parts(p).unwrap_err(),
        LawError::TooFewPoints
    );
    let mut p = good.clone();
    p.domain = ValidRange { lo: 1.0, hi: 1.0 };
    assert_eq!(
        SignalLaw::from_parts(p).unwrap_err(),
        LawError::DegenerateDomain
    );
    let mut p = good.clone();
    p.coefficients[0] = f64::NAN;
    assert_eq!(SignalLaw::from_parts(p).unwrap_err(), LawError::NonFinite);
    let mut p = good.clone();
    p.evidence.push((5.0, 1.0)); // evidence outside the stated domain
    assert_eq!(SignalLaw::from_parts(p).unwrap_err(), LawError::OutOfRange);
    let mut p = good;
    p.evidence.clear(); // a law must keep the evidence it stands on
    assert_eq!(
        SignalLaw::from_parts(p).unwrap_err(),
        LawError::TooFewPoints
    );
}
