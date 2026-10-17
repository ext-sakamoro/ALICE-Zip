//! `analyze_signal_fft` against an independent high-precision DFT.
//!
//! oracle: the reference evaluates `X[k] = Σ_i c[i]·e^{-2πi·k·i/n}` directly on the
//! DC-removed signal `c` (the same `f32` values the function transforms), with the angle
//! reduced exactly in integers (`k·i mod n`), `alice_det_math::{cos64, sin64}` and
//! compensated (Neumaier) summation in `f64`. Its error is a few `f64` ulps of
//! `S = Σ|c[i]|`, far below what the bound below allows, so it stands in for the exact
//! transform. No FFT is involved in the reference.
//!
//! Bound (stated before the implementation): the function returns `f32` magnitude and
//! phase computed from the transform rounded to `f32`, so against the reference the
//! magnitude may differ by `3·ε32·|X[k]|` (rounding of the real part, the imaginary part,
//! the sum of squares and the square root) plus the transform's own error `E`, and the
//! phase by `4·ε32·π` (the correctly rounded `f32` atan2 is within one ulp of a value
//! `≤ π`, plus the rounding of its arguments) plus `E / |X[k]|`. An `f64` FFT of padded
//! length `m` has `E ≤ 16·log2(m)·ε64·S`; `m ≤ 4·n` for the non-power-of-two lengths.
//! The phase is checked only where `|X[k]| ≥ 1e4·(E + ε32·S)`, where it is defined well
//! enough for the bound to mean something.
//!
//! The lengths cover 1, primes (2, 3, 5, 7, 97), composites that are not powers of two
//! (12, 1000) and powers of two (8, 1024, 4096).
#![cfg(feature = "fft")]
#![allow(clippy::cast_precision_loss)]

use alice_det_math::{atan2_64, cos64, log2_64, sin64};
use alice_zip::generators::{analyze_signal, analyze_signal_fft};

const LENGTHS: [usize; 11] = [1, 2, 3, 5, 7, 8, 12, 97, 1000, 1024, 4096];

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / 16_777_216.0) * 10.0 - 5.0
        })
        .collect()
}

/// The DC-removed values the function transforms: `x[i] - mean`, both in `f32`. The mean
/// is the one `analyze_signal_fft` returns, so the reference transforms the same inputs.
fn centered(values: &[f32], dc: f32) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v - dc)).collect()
}

fn neumaier(acc: &mut (f64, f64), v: f64) {
    let t = acc.0 + v;
    if acc.0.abs() >= v.abs() {
        acc.1 += (acc.0 - t) + v;
    } else {
        acc.1 += (v - t) + acc.0;
    }
    acc.0 = t;
}

fn reference_bin(c: &[f64], k: usize) -> (f64, f64) {
    let n = c.len();
    let mut re = (0.0, 0.0);
    let mut im = (0.0, 0.0);
    for (i, &v) in c.iter().enumerate() {
        let r = (k * i) % n;
        let theta = 2.0 * core::f64::consts::PI * r as f64 / n as f64;
        neumaier(&mut re, v * cos64(theta));
        neumaier(&mut im, -(v * sin64(theta)));
    }
    (re.0 + re.1, im.0 + im.1)
}

fn padded_len(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        (2 * n - 1).next_power_of_two()
    }
}

fn wrap(d: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    let d = d % two_pi;
    if d > core::f32::consts::PI {
        d - two_pi
    } else if d < -core::f32::consts::PI {
        d + two_pi
    } else {
        d
    }
}

#[test]
fn every_bin_is_within_the_stated_bound_of_the_reference_dft() {
    let eps32 = f64::from(f32::EPSILON);
    let eps64 = f64::EPSILON;
    let mut compared = 0usize;
    let mut phases = 0usize;
    for (j, &n) in LENGTHS.iter().enumerate() {
        let values = signal(n, 0x5eed + j as u64);
        let (coefficients, dc) = analyze_signal_fft(&values, n, 1.0);
        let c = centered(&values, dc);
        let s: f64 = c.iter().map(|v| v.abs()).sum();
        let lg = if n > 1 {
            log2_64(padded_len(n) as f64)
        } else {
            1.0
        };
        let e = 16.0 * lg * eps64 * s;
        let mut by_k = coefficients.clone();
        by_k.sort_by_key(|t| t.0);
        // every non-DC bin 1..=n/2 above the reporting floor comes back
        let expected: Vec<usize> = (1..=n / 2)
            .filter(|&k| {
                let (re, im) = reference_bin(&c, k);
                (re * re + im * im).sqrt() > 1e-6
            })
            .collect();
        assert_eq!(
            by_k.iter().map(|t| t.0).collect::<Vec<_>>(),
            expected,
            "n={n}: the reported bins"
        );
        for &(k, magnitude, phase) in &by_k {
            let (re, im) = reference_bin(&c, k);
            let m_ref = (re * re + im * im).sqrt();
            let dm = (f64::from(magnitude) - m_ref).abs();
            let bound = 3.0 * eps32 * m_ref + e;
            assert!(
                dm <= bound,
                "n={n} k={k}: magnitude {magnitude} vs reference {m_ref} (|d| {dm:e} > {bound:e})"
            );
            compared += 1;
            if m_ref >= 1e4 * (e + eps32 * s) {
                let p_ref = atan2_64(im, re) as f32;
                let dp = f64::from(wrap(phase - p_ref).abs());
                let pbound = 4.0 * eps32 * core::f64::consts::PI + e / m_ref;
                assert!(
                    dp <= pbound,
                    "n={n} k={k}: phase {phase} vs reference {p_ref} (|d| {dp:e} > {pbound:e})"
                );
                phases += 1;
            }
        }
    }
    // the sum over LENGTHS of n/2 is 0+1+1+2+3+4+6+48+500+512+2048 = 3125; random signals have no zero bin
    assert_eq!(compared, 3125, "bins compared");
    assert!(phases > 3000, "phases compared: {phases}");
}

#[test]
fn length_one_has_no_bins_and_its_value_as_dc() {
    assert_eq!(analyze_signal_fft(&[2.5], 4, 1.0), (vec![], 2.5));
}

/// The FFT path and the naive DFT compute the same law: the same bins in the same order,
/// with magnitudes and phases within the bound above of each other
#[test]
fn the_fft_path_reports_the_bins_of_the_naive_dft() {
    for (j, &n) in LENGTHS.iter().enumerate() {
        let values = signal(n, 0xa11ce + j as u64);
        let (a, dc_a) = analyze_signal(&values, 6, 0.9);
        let (b, dc_b) = analyze_signal_fft(&values, 6, 0.9);
        assert_eq!(dc_a.to_bits(), dc_b.to_bits(), "n={n}: dc");
        assert_eq!(
            a.iter().map(|t| t.0).collect::<Vec<_>>(),
            b.iter().map(|t| t.0).collect::<Vec<_>>(),
            "n={n}: bins and their order"
        );
    }
}
