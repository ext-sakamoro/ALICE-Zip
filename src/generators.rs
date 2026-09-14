//! Procedural signal generators — Fourier reconstruction and polynomial regression
//! primitives that let downstream crates store *the function that produces the
//! signal* instead of the raw samples.
//!
//! This is the "Send laws, not data" principle applied to time-series and
//! numerical arrays: instead of persisting `n` samples verbatim, fit a compact
//! set of coefficients (polynomial or Fourier) that regenerates them.
//!
//! # API
//!
//! - [`fit_polynomial`] — least-squares polynomial regression, tries degrees
//!   ascending until the relative error falls below the caller-supplied
//!   threshold. Returns `(coefficients, degree, error)` when a fit is found.
//! - [`generate_polynomial`] — Horner-form evaluator for a polynomial with
//!   `coefficients[0] + coefficients[1] * x + coefficients[2] * x^2 + …` at
//!   the integer indices `x = 0, 1, …, n-1`.
//! - [`analyze_signal`] — naive DFT that returns the top-`k` frequency bins
//!   (by magnitude, with an energy threshold) as `(freq_idx, magnitude,
//!   phase)` plus the DC offset.
//! - [`generate_from_coefficients`] — inverse partial-sum reconstruction from
//!   the [`analyze_signal`] output.
//! - [`generate_sine_wave`] — single-tone helper for the common case of a
//!   pure sinusoid + DC offset.
//!
//! Implementations are dependency-free (naive DFT and Gaussian elimination
//! over `f64`). For typical compression payloads (a few hundred samples the
//! naive `O(n²)` DFT is adequate; large arrays should call a specialised FFT
//! crate directly and feed its coefficients into
//! [`generate_from_coefficients`].

use core::f32::consts::PI;

/// Fit a polynomial `c[0] + c[1]*x + … + c[k]*x^k` to `values` at integer
/// indices `x = 0, 1, …, values.len() - 1`. Tries degrees from `1` up to
/// `max_degree` and returns the *first* fit whose normalised MSE is below
/// `error_threshold`. Returns `None` if no polynomial up to `max_degree`
/// satisfies the threshold.
///
/// The error metric is `mean((f(i) - values[i])^2) / variance(values)`, i.e.
/// 1 - R², so a value of `0.01` means the polynomial explains 99% of the
/// variance.
#[must_use]
pub fn fit_polynomial(
    values: &[f32],
    max_degree: usize,
    error_threshold: f64,
) -> Option<(Vec<f64>, usize, f64)> {
    if values.len() < 2 {
        return None;
    }
    let variance = variance_f32(values);
    if variance == 0.0 {
        // Perfectly constant series — treat as degree-0 fit with a single
        // coefficient. Callers who care about constant series should use
        // their own fast path; we still return a well-formed result.
        return Some((vec![f64::from(values[0])], 0, 0.0));
    }

    let n = values.len();
    let ys: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();

    for degree in 1..=max_degree {
        let Some(coeffs) = solve_polynomial_ls(n, &ys, degree) else {
            continue;
        };

        let mse = ys
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let yhat = eval_poly(&coeffs, i as f64);
                (y - yhat).powi(2)
            })
            .sum::<f64>()
            / (n as f64);

        let rel = mse / variance;
        if rel < error_threshold {
            return Some((coeffs, degree, rel));
        }
    }
    None
}

/// Evaluate a polynomial at integer indices `x = 0, 1, …, n-1` and return
/// the samples as `f32`.
#[must_use]
pub fn generate_polynomial(n: usize, coefficients: &[f64]) -> Vec<f32> {
    (0..n)
        .map(|i| eval_poly(coefficients, i as f64) as f32)
        .collect()
}

/// Naive DFT of `values` returning the top-`max_coefficients` non-DC
/// frequency bins (ordered by descending magnitude) as
/// `(freq_index, magnitude, phase)` plus the DC offset (mean).
///
/// Bins are dropped once their cumulative energy exceeds `energy_threshold`
/// of the total non-DC spectral energy — the returned vector always contains
/// at most `max_coefficients` entries, but may contain fewer when the signal
/// concentrates its energy in the first few bins.
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
    let dc_offset = values.iter().sum::<f32>() / (n as f32);

    // Only unique bins matter — the DFT is Hermitian for real input, so
    // bins > n/2 mirror bins < n/2. We iterate 1..=n/2 which covers the
    // full unique frequency range.
    let half = n / 2;
    let mut spectrum: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(half); // (idx, mag, phase, energy)
    for k in 1..=half {
        let (re, im) = dft_bin(values, k);
        let mag = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        spectrum.push((k, mag, phase, mag * mag));
    }

    // Sort by descending magnitude, keep top max_coefficients.
    spectrum.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    spectrum.truncate(max_coefficients);

    // Energy-threshold trimming: drop the tail once cumulative energy
    // exceeds threshold * total_energy.
    let total_energy: f32 = spectrum.iter().map(|(_, _, _, e)| *e).sum();
    if total_energy > 0.0 && energy_threshold > 0.0 && energy_threshold < 1.0 {
        let cutoff = total_energy * energy_threshold;
        let mut cum = 0.0;
        let mut keep = 0;
        for (i, &(_, _, _, e)) in spectrum.iter().enumerate() {
            cum += e;
            keep = i + 1;
            if cum >= cutoff {
                break;
            }
        }
        spectrum.truncate(keep);
    }

    let coefs = spectrum.into_iter().map(|(k, m, p, _)| (k, m, p)).collect();
    (coefs, dc_offset)
}

/// Reconstruct a signal of length `n` from the top-k Fourier coefficients
/// produced by [`analyze_signal`] plus the DC offset.
///
/// Each coefficient `(k, magnitude, phase)` contributes
/// `(2 * magnitude / n) * cos(2*pi*k*i/n + phase)` to sample `i`. The `2/n`
/// scaling matches the amplitude convention used by callers that treat the
/// DFT bin magnitude as `n/2 * amplitude` for a real sinusoid, and the
/// `+ phase` sign matches the inverse of the forward transform used by
/// [`analyze_signal`] (which computes `phase = atan2(Im(X[k]), Re(X[k]))`
/// against the standard `X[k] = sum(x[n] * exp(-2*pi*i*k*n/N))` DFT
/// convention).
#[must_use]
pub fn generate_from_coefficients(
    n: usize,
    coefficients: &[(usize, f32, f32)],
    dc_offset: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            let x = i as f32;
            let sum: f32 = coefficients
                .iter()
                .map(|&(k, mag, phase)| {
                    let amp = 2.0 * mag * inv_n;
                    let theta = 2.0 * PI * k as f32 * x * inv_n + phase;
                    amp * theta.cos()
                })
                .sum();
            sum + dc_offset
        })
        .collect()
}

/// Emit `n` samples of `amplitude * sin(2*pi*frequency*i/n + phase) +
/// dc_offset` for `i = 0, 1, …, n-1`.
#[must_use]
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
/// `dc_offset + sum_j(amplitude_j * sin(2*pi*frequency_j*i/n + phase_j))`.
#[must_use]
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

/// 1D fractional-Brownian-motion Perlin-like value noise with configurable
/// octaves, persistence and lacunarity — deterministic given `seed`.
///
/// The output range is roughly `[-1.0, 1.0]` before `octaves * persistence`
/// amplification.
///
/// # Parameters
///
/// - `n`: number of samples.
/// - `dimension`: currently accepts `1` (1D slice); values > 1 are treated as
///   1D for forward-compatibility.
/// - `seed`: deterministic PRNG seed for the gradient lattice.
/// - `scale`: base spatial frequency (larger = finer detail).
/// - `octaves`: number of octaves summed (`>= 1`).
/// - `persistence`: amplitude multiplier between octaves (typically `0.5`).
/// - `lacunarity`: frequency multiplier between octaves (typically `2.0`).
#[must_use]
pub fn generate_perlin_advanced(
    n: usize,
    _dimension: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let octaves = octaves.max(1);
    let inv_n = 1.0 / n as f32;

    (0..n)
        .map(|i| {
            let mut sum = 0.0_f32;
            let mut amplitude = 1.0_f32;
            let mut frequency = scale.max(1e-6);
            let mut max_amp = 0.0_f32;
            let x = i as f32 * inv_n;

            for _ in 0..octaves {
                let px = x * frequency;
                sum += amplitude * value_noise_1d(px, seed);
                max_amp += amplitude;
                amplitude *= persistence;
                frequency *= lacunarity;
            }

            if max_amp > 0.0 {
                sum / max_amp
            } else {
                sum
            }
        })
        .collect()
}

// ---------- internals ----------

fn variance_f32(values: &[f32]) -> f64 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    values
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / n
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    // Horner's method: c0 + x*(c1 + x*(c2 + …))
    coeffs.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c)
}

/// Solve the least-squares polynomial fit by forming the normal equations
/// `V^T V c = V^T y` where `V` is the Vandermonde matrix.
fn solve_polynomial_ls(n: usize, ys: &[f64], degree: usize) -> Option<Vec<f64>> {
    let k = degree + 1; // number of coefficients
    if n < k {
        return None;
    }

    // Precompute power sums S[j] = sum_{i=0}^{n-1} i^j for j = 0..=2*degree.
    let mut s = vec![0.0_f64; 2 * degree + 1];
    for i in 0..n {
        let xi = i as f64;
        let mut pow = 1.0_f64;
        for sj in s.iter_mut() {
            *sj += pow;
            pow *= xi;
        }
    }

    // Precompute t[j] = sum_{i=0}^{n-1} y_i * i^j for j = 0..=degree.
    let mut t = vec![0.0_f64; k];
    for (i, &y) in ys.iter().enumerate() {
        let xi = i as f64;
        let mut pow = 1.0_f64;
        for tj in t.iter_mut() {
            *tj += y * pow;
            pow *= xi;
        }
    }

    // Build the (k x (k+1)) augmented matrix.
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|r| {
            let mut row = Vec::with_capacity(k + 1);
            for c in 0..k {
                row.push(s[r + c]);
            }
            row.push(t[r]);
            row
        })
        .collect();

    // Gaussian elimination with partial pivoting.
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        // Pivot: find row with max |a[r][i]|.
        let mut pivot = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..k {
            let v = a[r][i].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(i, pivot);

        // Eliminate.
        for r in (i + 1)..k {
            let factor = a[r][i] / a[i][i];
            for c in i..=k {
                a[r][c] -= factor * a[i][c];
            }
        }
    }

    // Back-substitute.
    let mut coeffs = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = a[i][k];
        for c in (i + 1)..k {
            sum -= a[i][c] * coeffs[c];
        }
        coeffs[i] = sum / a[i][i];
    }

    Some(coeffs)
}

/// 1D value noise with cubic Hermite (smoothstep) interpolation between
/// hashed integer lattice points. Deterministic given the seed.
fn value_noise_1d(x: f32, seed: u64) -> f32 {
    let x0 = x.floor();
    let x1 = x0 + 1.0;
    let t = x - x0;
    // Smoothstep: 3t² - 2t³
    let s = t * t * (3.0 - 2.0 * t);
    let g0 = hash_to_unit(x0 as i32, seed);
    let g1 = hash_to_unit(x1 as i32, seed);
    g0 * (1.0 - s) + g1 * s
}

fn hash_to_unit(x: i32, seed: u64) -> f32 {
    // Small mixed-word hash (Wang-style) mapped to `[-1.0, 1.0]`.
    let mut h = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= seed.wrapping_mul(0x85eb_ca6b_c2b2_ae35);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc2b2_ae35_9e37_79b9);
    h ^= h >> 33;
    // Convert to [-1, 1]
    let f = (h & 0x00ff_ffff) as f32 / 8_388_608.0; // 2^23
    f - 1.0
}

fn dft_bin(values: &[f32], k: usize) -> (f32, f32) {
    let n = values.len() as f32;
    let inv_n = 1.0 / n;
    let mut re = 0.0_f32;
    let mut im = 0.0_f32;
    for (i, &v) in values.iter().enumerate() {
        let theta = 2.0 * PI * k as f32 * i as f32 * inv_n;
        re += v * theta.cos();
        im -= v * theta.sin();
    }
    (re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_polynomial_linear() {
        // f(x) = 3 + 2x
        let out = generate_polynomial(5, &[3.0, 2.0]);
        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn generate_polynomial_quadratic() {
        // f(x) = 1 + x + x^2
        let out = generate_polynomial(4, &[1.0, 1.0, 1.0]);
        assert_eq!(out, vec![1.0, 3.0, 7.0, 13.0]);
    }

    #[test]
    fn generate_polynomial_empty() {
        assert!(generate_polynomial(0, &[1.0, 2.0]).is_empty());
    }

    #[test]
    fn fit_polynomial_linear_series() {
        let values: Vec<f32> = (0..20).map(|i| 3.0 + 2.0 * i as f32).collect();
        let (coefs, deg, err) = fit_polynomial(&values, 3, 1e-6).expect("linear fit");
        assert_eq!(deg, 1);
        assert!((coefs[0] - 3.0).abs() < 1e-4);
        assert!((coefs[1] - 2.0).abs() < 1e-4);
        assert!(err < 1e-6);
    }

    #[test]
    fn fit_polynomial_quadratic_series() {
        let values: Vec<f32> = (0..20)
            .map(|i| 1.0 + i as f32 + (i as f32).powi(2))
            .collect();
        let (coefs, deg, _) = fit_polynomial(&values, 4, 1e-4).expect("quadratic fit");
        assert_eq!(deg, 2);
        assert!((coefs[0] - 1.0).abs() < 1e-3);
        assert!((coefs[1] - 1.0).abs() < 1e-3);
        assert!((coefs[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn fit_polynomial_noise_returns_none() {
        // Alternating series — no low-degree polynomial should fit within
        // the tight threshold.
        let values: Vec<f32> = (0..40)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(fit_polynomial(&values, 3, 0.0001).is_none());
    }

    #[test]
    fn fit_polynomial_too_short() {
        assert!(fit_polynomial(&[1.0], 2, 0.01).is_none());
        assert!(fit_polynomial(&[], 2, 0.01).is_none());
    }

    #[test]
    fn generate_sine_wave_shape() {
        let out = generate_sine_wave(4, 1.0, 1.0, 0.0, 0.0);
        // sin(0), sin(pi/2), sin(pi), sin(3pi/2) = 0, 1, 0, -1
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
        assert!((out[3] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn analyze_signal_pure_tone() {
        // 1 Hz over n=32 samples, amplitude 2.
        let n = 32usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 2.0 * (2.0 * PI * i as f32 / n as f32).sin())
            .collect();
        let (coefs, dc) = analyze_signal(&samples, 4, 0.99);
        assert!(dc.abs() < 1e-4, "DC offset should be ~0");
        assert!(!coefs.is_empty());
        // Top bin should be k=1.
        assert_eq!(coefs[0].0, 1);
    }

    #[test]
    fn round_trip_pure_tone() {
        // Encode → decode should recover the tone (approximately) via
        // generate_from_coefficients.
        let n = 32usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 3.0 + 2.0 * (2.0 * PI * i as f32 / n as f32).sin())
            .collect();
        let (coefs, dc) = analyze_signal(&samples, 4, 0.99);
        let recon = generate_from_coefficients(n, &coefs, dc);
        assert_eq!(recon.len(), n);
        let mse: f32 = samples
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32;
        assert!(mse < 0.05, "reconstruction MSE = {mse} too high");
    }

    #[test]
    fn analyze_signal_empty_and_zero_max() {
        let (c, dc) = analyze_signal(&[], 3, 0.9);
        assert!(c.is_empty() && dc == 0.0);
        let (c, _) = analyze_signal(&[1.0, 2.0, 3.0], 0, 0.9);
        assert!(c.is_empty());
    }

    #[test]
    fn generate_from_coefficients_empty() {
        assert!(generate_from_coefficients(0, &[], 0.0).is_empty());
    }
}
