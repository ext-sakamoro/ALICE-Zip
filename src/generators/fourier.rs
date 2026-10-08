//! Fourier analysis / reconstruction and sinusoid generators
//!
//! Law (`X[k] = Σ_i x[i]·e^{-2πi·k·i/n}` on the DC-removed signal):
//!
//! - [`analyze_signal`] returns the strongest non-DC bins `k ∈ 1..=n/2` as
//!   `(k, |X[k]|, atan2(Im X[k], Re X[k]))` plus the DC offset (mean)
//! - [`generate_from_coefficients`] inverts it: sample `i` is
//!   `dc + Σ_k w_k · |X[k]| / n · cos(2π k i / n + φ_k)` with `w_k = 2` for
//!   `0 < k < n/2` (the mirrored negative-frequency bin) and `w_k = 1` for
//!   `k == 0` or `k == n/2` (self-conjugate bins) Versions ≤ 0.3 used
//!   `w = 2` for the Nyquist bin, doubling it; `tests/analytic_oracle.rs`
//!   pins the exact reconstruction of every bin including `k = n/2`
//! - `analyze_signal_fft` (`fft` feature) computes the same bins with
//!   rustfft in O(n log n); the coefficient selection is shared code and the
//!   oracle test pins naive-vs-FFT parity
//!
//! Sinusoid helpers evaluate `A · sin(2π f i / n + φ) + dc`
//!
//! # One law, one implementation
//!
//! Every reconstruction law has exactly one evaluator — the **point** evaluator
//! ([`sine_at`], [`multi_sine_at`], [`fourier_at`]) — and the array generators
//! are a `map` over it. Consumers that answer single-sample queries call the
//! point evaluator instead of writing their own copy of the law.
//!
//! This is not a style preference. The laws used to be implemented twice: here
//! over whole arrays, and again in the consumer that needed one sample at a
//! time. On the same input at integer positions with `n = 1024` the two
//! disagreed on **788 / 1024** samples for a sine, **881 / 1024** for a
//! multi-sine and **962 / 1024** for a Fourier reconstruction, because the
//! array side accumulated in `f32` and added the DC term last while the copy
//! accumulated in `f64` starting from the DC term. Against the `f64` closed
//! form the `f32` accumulation was off by at most 9.16e-7 and the `f64` one by
//! 5.95e-8, so the laws now accumulate in `f64` and round once on the way out.
//! `tests/law_single_source.rs` pins both halves: the error against the closed
//! form stays inside one `f32` step, and the array output equals the point
//! output bit for bit.

use alloc::vec::Vec;
use core::f64::consts::PI;

use alice_det_math::{atan2, cos64, sin64};

// `sqrt` only: IEEE 754 specifies it exactly, so the shim and the platform
// version agree bit for bit. The transcendentals above do not go through it.
// (test builds link std, whose inherent methods shadow the trait → allow)
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::FloatExt;

/// Bins with magnitude below this are never reported (numerical zero)
const MIN_MAGNITUDE: f32 = 1e-10;

/// Naive DFT of `values` returning the top-`max_coefficients` non-DC
/// frequency bins (ordered by descending magnitude, ties by ascending bin) as
/// `(freq_index, magnitude, phase)` plus the DC offset (mean).
///
/// Bins are dropped once their cumulative energy reaches `energy_threshold`
/// of the **total** non-DC spectral energy (all bins `1..=n/2`); thresholds
/// outside `(0, 1)` disable the trimming. The returned vector always holds at
/// most `max_coefficients` entries. Empty input or `max_coefficients == 0`
/// yields `(vec![], 0.0)`.
#[must_use]
pub fn analyze_signal(
    values: &[f32],
    max_coefficients: usize,
    energy_threshold: f32,
) -> (Vec<(usize, f32, f32)>, f32) {
    if values.is_empty() || max_coefficients == 0 {
        return (Vec::new(), 0.0);
    }
    let n = values.len();
    let dc_offset = mean(values);
    let centered: Vec<f32> = values.iter().map(|v| v - dc_offset).collect();

    let spectrum: Vec<(usize, f32, f32)> = (1..=n / 2)
        .map(|k| {
            let (re, im) = dft_bin(&centered, k);
            (k, (re * re + im * im).sqrt(), atan2(im, re))
        })
        .collect();

    (
        select_coefficients(spectrum, max_coefficients, energy_threshold),
        dc_offset,
    )
}

/// [`analyze_signal`] computed with rustfft (`fft` feature) Same output
/// contract; magnitudes / phases agree with the naive DFT to floating-point
/// round-off (see `tests/analytic_oracle.rs`)
#[cfg(feature = "fft")]
#[must_use]
pub fn analyze_signal_fft(
    values: &[f32],
    max_coefficients: usize,
    energy_threshold: f32,
) -> (Vec<(usize, f32, f32)>, f32) {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    if values.is_empty() || max_coefficients == 0 {
        return (Vec::new(), 0.0);
    }
    let n = values.len();
    let dc_offset = mean(values);
    let mut buffer: Vec<Complex<f32>> = values
        .iter()
        .map(|&x| Complex::new(x - dc_offset, 0.0))
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);

    let spectrum: Vec<(usize, f32, f32)> = buffer[1..=n / 2]
        .iter()
        .enumerate()
        .map(|(i, c)| (i + 1, c.norm(), c.arg()))
        .collect();

    (
        select_coefficients(spectrum, max_coefficients, energy_threshold),
        dc_offset,
    )
}

/// Sample `i` of the Fourier reconstruction — **the law**
///
/// Each coefficient `(k, magnitude, phase)` with `k < n` contributes
/// `w_k · magnitude / n · cos(2π k i / n + phase)`, where `w_k = 2` for
/// `0 < k < n/2` and `w_k = 1` for `k == 0` or `k == n/2` (self-conjugate
/// bins). Coefficients with `k >= n` are ignored (a DFT bin index of a
/// length-`n` transform is always `< n`).
///
/// `i` is a position in samples and may be fractional, which is what lets a
/// consumer answer a point query without materialising the whole segment.
/// `n == 0` has no positions, so the law degenerates to the DC term.
///
/// Accumulation is `f64` from the DC term outwards and rounds to `f32` once on
/// return — see the module docs for the measurement behind that choice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn fourier_at(n: usize, coefficients: &[(usize, f32, f32)], dc_offset: f32, i: f64) -> f32 {
    if n == 0 {
        return dc_offset;
    }
    let inv_n = 1.0 / n as f64;
    let mut sum = f64::from(dc_offset);
    for &(k, mag, phase) in coefficients {
        if k >= n {
            continue;
        }
        let weight = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
        let theta = 2.0 * PI * k as f64 * i * inv_n + f64::from(phase);
        sum += weight * f64::from(mag) * inv_n * cos64(theta);
    }
    sum as f32
}

/// Sample `i` of `amplitude · sin(2π·frequency·i/n + phase) + dc_offset` — **the law**
///
/// `i` may be fractional (see [`fourier_at`] for why). `n == 0` degenerates to
/// `amplitude · sin(phase) + dc_offset`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn sine_at(
    n: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    dc_offset: f32,
    i: f64,
) -> f32 {
    let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
    let theta = 2.0 * PI * f64::from(frequency) * i * inv_n + f64::from(phase);
    (f64::from(dc_offset) + f64::from(amplitude) * sin64(theta)) as f32
}

/// Sample `i` of a sum of sinusoids plus one shared `dc_offset` — **the law**
///
/// `dc_offset + Σ_j amplitude_j · sin(2π·frequency_j·i/n + phase_j)`, i.e. the
/// sum of the corresponding [`sine_at`] values with a single DC term.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn multi_sine_at(n: usize, components: &[(f32, f32, f32)], dc_offset: f32, i: f64) -> f32 {
    let inv_n = if n == 0 { 0.0 } else { 1.0 / n as f64 };
    let mut sum = f64::from(dc_offset);
    for &(freq, amp, phase) in components {
        let theta = 2.0 * PI * f64::from(freq) * i * inv_n + f64::from(phase);
        sum += f64::from(amp) * sin64(theta);
    }
    sum as f32
}

/// Reconstruct a signal of length `n` from the Fourier coefficients produced
/// by [`analyze_signal`] / `analyze_signal_fft` plus the DC offset.
///
/// This is [`fourier_at`] evaluated at `i = 0, 1, …, n-1` — the array and the
/// point query cannot drift apart because there is only one law.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_from_coefficients(
    n: usize,
    coefficients: &[(usize, f32, f32)],
    dc_offset: f32,
) -> Vec<f32> {
    (0..n)
        .map(|i| fourier_at(n, coefficients, dc_offset, i as f64))
        .collect()
}

/// Emit `n` samples of `amplitude * sin(2*pi*frequency*i/n + phase) +
/// dc_offset` for `i = 0, 1, …, n-1`.
///
/// This is [`sine_at`] evaluated at the integer positions.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_sine_wave(
    n: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    dc_offset: f32,
) -> Vec<f32> {
    (0..n)
        .map(|i| sine_at(n, frequency, amplitude, phase, dc_offset, i as f64))
        .collect()
}

/// Emit `n` samples that are the sum of multiple sinusoids `(frequency,
/// amplitude, phase)` plus a shared `dc_offset`.
///
/// This is [`multi_sine_at`] evaluated at the integer positions.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_multi_sine(n: usize, components: &[(f32, f32, f32)], dc_offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| multi_sine_at(n, components, dc_offset, i as f64))
        .collect()
}

// ---------- internals ----------

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// Shared coefficient selection: drop numerical zeros, order by magnitude
/// (descending, ties by bin), keep at most `max_coefficients`, then trim once
/// the cumulative energy reaches `energy_threshold` of the total
fn select_coefficients(
    mut spectrum: Vec<(usize, f32, f32)>,
    max_coefficients: usize,
    energy_threshold: f32,
) -> Vec<(usize, f32, f32)> {
    spectrum.retain(|&(_, mag, _)| mag >= MIN_MAGNITUDE);
    let total_energy: f32 = spectrum.iter().map(|&(_, m, _)| m * m).sum();
    // Stable sort keeps ascending-bin order among exact ties
    spectrum.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    spectrum.truncate(max_coefficients);

    if total_energy > 0.0 && energy_threshold > 0.0 && energy_threshold < 1.0 {
        let cutoff = total_energy * energy_threshold;
        let mut cum = 0.0_f32;
        let mut keep = 0;
        for (i, &(_, m, _)) in spectrum.iter().enumerate() {
            cum += m * m;
            keep = i + 1;
            if cum >= cutoff {
                break;
            }
        }
        spectrum.truncate(keep);
    }
    spectrum
}

/// One DFT bin of the (already DC-removed) signal, accumulated in `f64` so the
/// naive path is the precision reference for the FFT path
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn dft_bin(values: &[f32], k: usize) -> (f32, f32) {
    let omega = 2.0 * core::f64::consts::PI * k as f64 / values.len() as f64;
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for (i, &v) in values.iter().enumerate() {
        let theta = omega * i as f64;
        re += f64::from(v) * cos64(theta);
        im -= f64::from(v) * sin64(theta);
    }
    (re as f32, im as f32)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn generate_sine_wave_shape() {
        let out = generate_sine_wave(4, 1.0, 1.0, 0.0, 0.0);
        // sin(0), sin(pi/2), sin(pi), sin(3pi/2) = 0, 1, 0, -1
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
        assert!((out[3] + 1.0).abs() < 1e-5);
        assert!(generate_sine_wave(0, 1.0, 1.0, 0.0, 0.0).is_empty());
        assert!(generate_multi_sine(0, &[(1.0, 1.0, 0.0)], 0.0).is_empty());
    }

    #[test]
    fn analyze_signal_pure_tone() {
        let n = 32usize;
        let samples = generate_sine_wave(n, 1.0, 2.0, 0.0, 0.0);
        let (coefs, dc) = analyze_signal(&samples, 4, 0.99);
        assert!(dc.abs() < 1e-4, "DC offset should be ~0");
        assert_eq!(coefs.len(), 1, "one tone → one bin: {coefs:?}");
        assert_eq!(coefs[0].0, 1);
        // |X[1]| = A n / 2
        assert!((coefs[0].1 - 32.0).abs() < 1e-3, "{}", coefs[0].1);
    }

    #[test]
    fn round_trip_pure_tone() {
        let n = 32usize;
        let samples = generate_sine_wave(n, 1.0, 2.0, 0.0, 3.0);
        let (coefs, dc) = analyze_signal(&samples, 4, 0.99);
        let recon = generate_from_coefficients(n, &coefs, dc);
        assert_eq!(recon.len(), n);
        for (a, b) in samples.iter().zip(&recon) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn analyze_signal_empty_and_zero_max() {
        let (c, dc) = analyze_signal(&[], 3, 0.9);
        assert!(c.is_empty() && dc == 0.0);
        let (c, _) = analyze_signal(&[1.0, 2.0, 3.0], 0, 0.9);
        assert!(c.is_empty());
        // Constant signal: every non-DC bin is a numerical zero → no coefficients
        let (c, dc) = analyze_signal(&[5.0; 16], 3, 0.9);
        assert!(c.is_empty());
        assert!((dc - 5.0).abs() < 1e-6);
    }

    #[test]
    fn generate_from_coefficients_edge_cases() {
        assert!(generate_from_coefficients(0, &[], 0.0).is_empty());
        // k >= n is ignored, DC only
        let out = generate_from_coefficients(4, &[(4, 100.0, 0.0), (9, 1.0, 0.0)], 2.0);
        assert_eq!(out, vec![2.0; 4]);
    }

    #[test]
    fn energy_threshold_uses_total_energy() {
        // Two tones, energies 16 : 9 → the first bin is 0.64 of the total, so
        // 0.9 needs both bins and 0.5 is reached by the first alone
        let n = 64usize;
        let s = generate_multi_sine(n, &[(3.0, 4.0, 0.0), (7.0, 3.0, 0.0)], 0.0);
        let (c, _) = analyze_signal(&s, 8, 0.9);
        assert_eq!(c.len(), 2, "{c:?}");
        // 0.5 of the total is reached by the first bin alone
        let (c, _) = analyze_signal(&s, 8, 0.5);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].0, 3);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn fft_matches_naive_dft() {
        let n = 128usize;
        let s = generate_multi_sine(n, &[(3.0, 4.0, 0.3), (17.0, 1.5, -1.0)], 0.7);
        // The two real tones: identical bins, magnitudes and phases
        let (a, dc_a) = analyze_signal(&s, 2, 1.0);
        let (b, dc_b) = analyze_signal_fft(&s, 2, 1.0);
        assert!((dc_a - dc_b).abs() < 1e-5);
        assert_eq!(a.len(), 2);
        assert_eq!(a.len(), b.len());
        for ((ka, ma, pa), (kb, mb, pb)) in a.iter().zip(&b) {
            assert_eq!(ka, kb);
            assert!((ma - mb).abs() < 1e-2 * ma.max(1.0), "{ma} vs {mb}");
            assert!((pa - pb).abs() < 1e-3, "{pa} vs {pb}");
        }
        // With every bin kept, both spectra reconstruct the same signal (the
        // leakage bins below 1e-4 may be ordered differently, the sum is not)
        let (a, dc_a) = analyze_signal(&s, n / 2, 1.0);
        let (b, dc_b) = analyze_signal_fft(&s, n / 2, 1.0);
        let ra = generate_from_coefficients(n, &a, dc_a);
        let rb = generate_from_coefficients(n, &b, dc_b);
        for ((x, y), orig) in ra.iter().zip(&rb).zip(&s) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
            assert!((x - orig).abs() < 1e-3, "{x} vs {orig}");
        }
    }
}
