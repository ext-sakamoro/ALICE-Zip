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
//! Sinusoid helpers evaluate `A · sin(2π f i / n + φ) + dc` in `f32`

use alloc::vec::Vec;
use core::f32::consts::PI;

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
            (k, (re * re + im * im).sqrt(), im.atan2(re))
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

/// Reconstruct a signal of length `n` from the Fourier coefficients produced
/// by [`analyze_signal`] / `analyze_signal_fft` plus the DC offset.
///
/// Each coefficient `(k, magnitude, phase)` with `k < n` contributes
/// `w_k · magnitude / n · cos(2π k i / n + phase)` to sample `i`, where
/// `w_k = 2` for `0 < k < n/2` and `w_k = 1` for `k == 0` or `k == n/2`
/// (self-conjugate bins). Coefficients with `k >= n` are ignored (a DFT bin
/// index of a length-`n` transform is always `< n`).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_from_coefficients(
    n: usize,
    coefficients: &[(usize, f32, f32)],
    dc_offset: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let inv_n = 1.0 / n as f32;
    let terms: Vec<(f32, f32, f32)> = coefficients
        .iter()
        .filter(|&&(k, _, _)| k < n)
        .map(|&(k, mag, phase)| {
            let weight = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
            (weight * mag * inv_n, 2.0 * PI * k as f32 * inv_n, phase)
        })
        .collect();
    (0..n)
        .map(|i| {
            let x = i as f32;
            let sum: f32 = terms
                .iter()
                .map(|&(amp, omega, phase)| amp * (omega * x + phase).cos())
                .sum();
            sum + dc_offset
        })
        .collect()
}

/// Emit `n` samples of `amplitude * sin(2*pi*frequency*i/n + phase) +
/// dc_offset` for `i = 0, 1, …, n-1`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_sine_wave(
    n: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    dc_offset: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * frequency * i as f32 * inv_n + phase;
            amplitude * theta.sin() + dc_offset
        })
        .collect()
}

/// Emit `n` samples that are the sum of multiple sinusoids `(frequency,
/// amplitude, phase)` plus a shared `dc_offset`.
///
/// Each sample `i` evaluates to
/// `dc_offset + sum_j(amplitude_j * sin(2*pi*frequency_j*i/n + phase_j))`,
/// i.e. exactly the sum of the corresponding [`generate_sine_wave`] outputs
/// (with a single DC term).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_multi_sine(n: usize, components: &[(f32, f32, f32)], dc_offset: f32) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            let x = i as f32;
            let sum: f32 = components
                .iter()
                .map(|&(freq, amp, phase)| {
                    let theta = 2.0 * PI * freq * x * inv_n + phase;
                    amp * theta.sin()
                })
                .sum();
            sum + dc_offset
        })
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
        re += f64::from(v) * theta.cos();
        im -= f64::from(v) * theta.sin();
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
