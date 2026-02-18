//! Procedural Data Analyzer
//!
//! Ports the Python analyzers from alice_zip/analyzers.py to Rust.
//! Tries sine, Fourier, and polynomial fitting in order, then falls back
//! to LZMA if no procedural method achieves a good enough compression ratio.
//!
//! Reuses existing Rust generators:
//! - `crate::generators::fourier::analyze_signal`
//! - `crate::generators::polynomial::fit_polynomial`
//!
//! License: MIT
//! Author: Moroya Sakamoto

use std::f64::consts::PI;

use crate::generators::fourier::{analyze_signal, generate_sine_wave, generate_from_coefficients};
use crate::generators::polynomial::{fit_polynomial, generate_polynomial};
use crate::compression::{lzma_compress, lzma_decompress};

// ---------------------------------------------------------------------------
// Constants (mirrors Python analyzer constants)
// ---------------------------------------------------------------------------

/// Maximum polynomial degree for free version
const MAX_POLYNOMIAL_DEGREE: usize = 5;

/// Minimum compression ratio to prefer procedural fit over LZMA
const MIN_PROCEDURAL_RATIO: f32 = 1.5;

/// Maximum acceptable normalized error for procedural compression
const MAX_ACCEPTABLE_ERROR: f64 = 0.01;

/// Maximum Fourier coefficients used for compression
const MAX_FOURIER_COEFFICIENTS: usize = 20;

/// Energy threshold for Fourier fitting (captures this fraction of total energy)
const FOURIER_ENERGY_THRESHOLD: f32 = 0.99;

/// Compressed size in bytes for a pure-sine fit (4 f64 coefficients + overhead)
const SINE_COMPRESSED_SIZE: usize = 64;

/// Overhead bytes added to polynomial/Fourier compressed size estimates
const COMPRESSED_OVERHEAD: usize = 32;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which fitting method produced this result
#[derive(Debug, Clone, PartialEq)]
pub enum FitMethod {
    /// Single-frequency sine wave: A·sin(2π·f·x + φ) + dc
    Sine,
    /// Multi-frequency Fourier series (sum of sine/cosine components)
    Fourier,
    /// Polynomial of degree ≤ MAX_POLYNOMIAL_DEGREE
    Polynomial,
    /// LZMA byte-level compression (fallback for non-procedural data)
    LzmaFallback,
}

/// Result of a fitting attempt
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Which method produced this result
    pub method: FitMethod,
    /// Flat coefficient vector whose meaning depends on `method`:
    /// - Sine:       [frequency, amplitude, phase, dc_offset]
    /// - Fourier:    [freq_idx_0, magnitude_0, phase_0, freq_idx_1, …] (triples)
    ///               followed by dc_offset as the last element
    /// - Polynomial: coefficients highest-degree-first (Horner convention)
    /// - LzmaFallback: empty (payload lives in `CompressedPayload::residual`)
    pub coefficients: Vec<f64>,
    /// Normalized mean-squared error relative to data variance (0 = perfect)
    pub error: f64,
    /// Estimated compression ratio: original bytes / compressed bytes
    pub compression_ratio: f32,
}

/// Self-contained payload produced by `ProceduralCompressionDesigner::compress`
#[derive(Debug, Clone)]
pub struct CompressedPayload {
    /// Best fit result (determines how to reconstruct the signal)
    pub fit_result: FitResult,
    /// LZMA-compressed residual (present when the fit is lossy and we want
    /// near-lossless reconstruction, or when method == LzmaFallback)
    pub residual: Option<Vec<u8>>,
    /// Number of samples in the original signal
    pub original_len: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute variance of a slice (returns 0.0 for empty or single-element slice)
#[inline]
fn variance_f64(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let inv_n = 1.0 / n as f64;
    let mean = data.iter().sum::<f64>() * inv_n;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() * inv_n
}

/// Compute normalized MSE: MSE / variance (returns MSE when variance ≈ 0)
#[inline]
fn normalized_mse(data: &[f64], fitted: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mse = data.iter().zip(fitted.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>() / n;
    let var = variance_f64(data);
    if var > 1e-15 { mse / var } else { mse }
}

// ---------------------------------------------------------------------------
// try_sine_fit
// ---------------------------------------------------------------------------

/// Try to fit `data` to a single-frequency sine wave.
///
/// Model: `y = amplitude · sin(2π · frequency · (i/n) + phase) + dc_offset`
///
/// Uses FFT (via `analyze_signal`) to estimate the dominant frequency, then
/// does a 360-step grid search over phase to minimize MSE.
///
/// # Returns
/// `Some(FitResult)` when the normalized error is below `MAX_ACCEPTABLE_ERROR`,
/// `None` otherwise.
pub fn try_sine_fit(data: &[f32]) -> Option<FitResult> {
    let n = data.len();
    if n < 4 {
        return None;
    }

    // --- DC offset & amplitude ---
    let inv_n = 1.0 / n as f64;
    let dc_offset: f64 = data.iter().map(|&x| x as f64).sum::<f64>() * inv_n;

    let centered: Vec<f64> = data.iter().map(|&x| x as f64 - dc_offset).collect();

    let max_c = centered.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_c = centered.iter().cloned().fold(f64::INFINITY, f64::min);
    let amplitude = (max_c - min_c) * 0.5;

    if amplitude < 1e-10 {
        return None;
    }

    // --- Dominant frequency via FFT (reuse existing analyze_signal) ---
    let f32_data: Vec<f32> = data.to_vec();
    // Ask for 1 coefficient; the first (highest-magnitude) gives the dominant freq
    let (coeffs, _dc) = analyze_signal(&f32_data, 1, 1.0);

    let frequency: f64 = if let Some(&(freq_idx, _mag, _phase)) = coeffs.first() {
        // freq_idx is the FFT bin index; convert to cycles per n samples
        freq_idx as f64
    } else {
        // Fallback: estimate from zero crossings
        let crossings: Vec<usize> = centered.windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                if (w[0] < 0.0) != (w[1] < 0.0) { Some(i) } else { None }
            })
            .collect();

        if crossings.len() >= 2 {
            let diffs: Vec<f64> = crossings.windows(2)
                .map(|w| (w[1] - w[0]) as f64)
                .collect();
            let avg_half_period = diffs.iter().sum::<f64>() / diffs.len() as f64;
            if avg_half_period > 0.0 {
                n as f64 / (2.0 * avg_half_period)
            } else {
                1.0
            }
        } else {
            1.0
        }
    };

    // --- Grid search over phase (360 steps, 1-degree resolution) ---
    // Pre-compute angular step to avoid redundant multiplications inside loop
    let two_pi_f_inv_n = 2.0 * PI * frequency * inv_n;

    // Pre-compute sin argument base for each sample index: two_pi_f_inv_n * i
    let args: Vec<f64> = (0..n).map(|i| two_pi_f_inv_n * i as f64).collect();

    let phase_step = 2.0 * PI / 360.0;
    let mut best_phase = 0.0_f64;
    let mut best_error = f64::INFINITY;

    for step in 0..360usize {
        let phase = step as f64 * phase_step;
        let mse = args.iter()
            .zip(data.iter())
            .map(|(&arg, &y)| {
                let fitted = amplitude * (arg + phase).sin() + dc_offset;
                let diff = y as f64 - fitted;
                diff * diff
            })
            .sum::<f64>() * inv_n;

        // Branchless-style: update best using direct comparison
        if mse < best_error {
            best_error = mse;
            best_phase = phase;
        }
    }

    // --- Normalized error ---
    let fitted_vals: Vec<f64> = args.iter()
        .map(|&arg| amplitude * (arg + best_phase).sin() + dc_offset)
        .collect();
    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let norm_err = normalized_mse(&data_f64, &fitted_vals);

    if norm_err >= MAX_ACCEPTABLE_ERROR {
        return None;
    }

    // Compressed size: 4 f64 coefficients + overhead = 64 bytes
    let original_bytes = (n * 4) as f32;
    let compression_ratio = original_bytes / SINE_COMPRESSED_SIZE as f32;

    Some(FitResult {
        method: FitMethod::Sine,
        coefficients: vec![frequency, amplitude, best_phase, dc_offset],
        error: norm_err,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// try_fourier_fit
// ---------------------------------------------------------------------------

/// Try to fit `data` to a Fourier series (sum of sine/cosine components).
///
/// Delegates to `analyze_signal` for FFT analysis, then stores
/// `(freq_idx, magnitude, phase)` triples as flat f64 coefficients, with
/// `dc_offset` appended as the final element.
///
/// # Returns
/// `Some(FitResult)` when the normalized error is below `MAX_ACCEPTABLE_ERROR`,
/// `None` otherwise.
pub fn try_fourier_fit(data: &[f32], max_coeffs: usize) -> Option<FitResult> {
    let n = data.len();
    if n < 8 {
        return None;
    }

    let (coefficients, dc_offset) =
        analyze_signal(data, max_coeffs.min(MAX_FOURIER_COEFFICIENTS), FOURIER_ENERGY_THRESHOLD);

    if coefficients.is_empty() {
        return None;
    }

    // Reconstruct signal to measure error
    let reconstructed = generate_from_coefficients(n, &coefficients, dc_offset);

    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let recon_f64: Vec<f64> = reconstructed.iter().map(|&x| x as f64).collect();
    let norm_err = normalized_mse(&data_f64, &recon_f64);

    if norm_err >= MAX_ACCEPTABLE_ERROR {
        return None;
    }

    // Flatten (freq_idx, magnitude, phase) triples into f64 coefficients,
    // appending dc_offset as the last element.
    // Layout: [freq_idx_0, magnitude_0, phase_0, freq_idx_1, …, dc_offset]
    let num_coeffs = coefficients.len();
    let mut flat: Vec<f64> = Vec::with_capacity(num_coeffs * 3 + 1);
    for (idx, mag, phase) in &coefficients {
        flat.push(*idx as f64);
        flat.push(*mag as f64);
        flat.push(*phase as f64);
    }
    flat.push(dc_offset as f64);

    // Compressed size estimate:
    // Each (freq_idx: 4 bytes, magnitude: 8 bytes, phase: 8 bytes) = 20 bytes per coeff
    // + dc_offset: 8 bytes + overhead: 32 bytes
    let compressed_bytes = (num_coeffs * 20 + 8 + COMPRESSED_OVERHEAD) as f32;
    let original_bytes = (n * 4) as f32;
    let compression_ratio = if compressed_bytes > 0.0 {
        original_bytes / compressed_bytes
    } else {
        0.0
    };

    Some(FitResult {
        method: FitMethod::Fourier,
        coefficients: flat,
        error: norm_err,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// try_polynomial_fit
// ---------------------------------------------------------------------------

/// Try to fit `data` to a polynomial of degree 1..=`max_degree`.
///
/// Delegates entirely to `fit_polynomial` (least-squares via Gaussian
/// elimination with partial pivoting, Horner evaluation).
///
/// # Returns
/// `Some(FitResult)` when the fit succeeds and normalized error is below
/// `MAX_ACCEPTABLE_ERROR`, `None` otherwise.
pub fn try_polynomial_fit(data: &[f32], max_degree: usize) -> Option<FitResult> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let degree = max_degree.min(MAX_POLYNOMIAL_DEGREE);
    // Use MAX_ACCEPTABLE_ERROR as error threshold; fit_polynomial returns None
    // if no degree achieves this level.
    let result = fit_polynomial(data, degree, MAX_ACCEPTABLE_ERROR)?;

    let (coeffs, _deg, rel_err) = result;

    if rel_err >= MAX_ACCEPTABLE_ERROR {
        return None;
    }

    // Compressed size: (degree+1) f64 values + overhead
    let compressed_bytes = (coeffs.len() * 8 + COMPRESSED_OVERHEAD) as f32;
    let original_bytes = (n * 4) as f32;
    let compression_ratio = if compressed_bytes > 0.0 {
        original_bytes / compressed_bytes
    } else {
        0.0
    };

    Some(FitResult {
        method: FitMethod::Polynomial,
        coefficients: coeffs,
        error: rel_err,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// analyze_data
// ---------------------------------------------------------------------------

/// Analyze `data` and return the best `FitResult`.
///
/// Strategy (mirrors Python `analyze_data`):
/// 1. Try sine fit.
/// 2. Try Fourier fit.
/// 3. Try polynomial fit.
/// 4. Pick the candidate with the highest `compression_ratio`.
/// 5. If the best ratio is below `MIN_PROCEDURAL_RATIO`, return an
///    `LzmaFallback` result instead.
pub fn analyze_data(data: &[f32]) -> FitResult {
    let mut best: Option<FitResult> = None;

    // Helper: replace `best` if `candidate` has a higher compression_ratio
    let mut update_best = |candidate: Option<FitResult>| {
        if let Some(c) = candidate {
            let replace = best.as_ref().map_or(true, |b| c.compression_ratio > b.compression_ratio);
            if replace {
                best = Some(c);
            }
        }
    };

    update_best(try_sine_fit(data));
    update_best(try_fourier_fit(data, MAX_FOURIER_COEFFICIENTS));
    update_best(try_polynomial_fit(data, MAX_POLYNOMIAL_DEGREE));

    // Accept procedural result only when ratio >= threshold
    if let Some(b) = best {
        if b.compression_ratio >= MIN_PROCEDURAL_RATIO {
            return b;
        }
    }

    // Fallback: LZMA (ratio computed from actual compressed size)
    let raw_bytes: Vec<u8> = data.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let original_bytes = raw_bytes.len() as f32;

    let compression_ratio = lzma_compress(&raw_bytes, 6)
        .map(|compressed| {
            let cb = compressed.len() as f32;
            if cb > 0.0 { original_bytes / cb } else { 1.0 }
        })
        .unwrap_or(1.0);

    FitResult {
        method: FitMethod::LzmaFallback,
        coefficients: Vec::new(),
        error: 0.0,
        compression_ratio,
    }
}

// ---------------------------------------------------------------------------
// Reconstruct from FitResult
// ---------------------------------------------------------------------------

/// Reconstruct a signal from a `FitResult`.
///
/// Used internally by `ProceduralCompressionDesigner::decompress`.
fn reconstruct(fit: &FitResult, n: usize) -> Vec<f32> {
    match fit.method {
        FitMethod::Sine => {
            // coefficients: [frequency, amplitude, phase, dc_offset]
            if fit.coefficients.len() < 4 {
                return vec![0.0; n];
            }
            let freq = fit.coefficients[0] as f32;
            let amp  = fit.coefficients[1] as f32;
            let phase= fit.coefficients[2] as f32;
            let dc   = fit.coefficients[3] as f32;
            generate_sine_wave(n, freq, amp, phase, dc)
        }

        FitMethod::Fourier => {
            // coefficients: [freq_idx_0, magnitude_0, phase_0, …, dc_offset]
            let coeff_slice = &fit.coefficients;
            if coeff_slice.is_empty() {
                return vec![0.0; n];
            }
            // Last element is dc_offset; remainder are triples
            let dc_offset = *coeff_slice.last().unwrap() as f32;
            let triples_flat = &coeff_slice[..coeff_slice.len() - 1];

            let coefficients: Vec<(usize, f32, f32)> = triples_flat
                .chunks_exact(3)
                .map(|c| (c[0] as usize, c[1] as f32, c[2] as f32))
                .collect();

            generate_from_coefficients(n, &coefficients, dc_offset)
        }

        FitMethod::Polynomial => {
            generate_polynomial(n, &fit.coefficients)
        }

        FitMethod::LzmaFallback => {
            // Cannot reconstruct without compressed bytes; return zeros
            vec![0.0; n]
        }
    }
}

// ---------------------------------------------------------------------------
// ProceduralCompressionDesigner
// ---------------------------------------------------------------------------

/// High-level API for procedural compression and decompression.
///
/// ```rust,no_run
/// use alice_core::analyzer::ProceduralCompressionDesigner;
/// use std::f32::consts::PI;
///
/// let n = 512;
/// let data: Vec<f32> = (0..n)
///     .map(|i| (2.0 * PI * 5.0 * i as f32 / n as f32).sin())
///     .collect();
///
/// let designer = ProceduralCompressionDesigner::new();
/// let payload = designer.compress(&data);
/// let recovered = designer.decompress(&payload);
/// ```
pub struct ProceduralCompressionDesigner {
    /// Maximum acceptable normalized error for procedural fit
    pub max_error: f64,
}

impl ProceduralCompressionDesigner {
    /// Create a designer with default settings.
    pub fn new() -> Self {
        Self { max_error: MAX_ACCEPTABLE_ERROR }
    }

    /// Create a designer with a custom error threshold.
    pub fn with_max_error(max_error: f64) -> Self {
        Self { max_error }
    }

    /// Compress `data` into a `CompressedPayload`.
    ///
    /// For procedural methods (Sine/Fourier/Polynomial), the residual between
    /// the reconstructed signal and the original is computed and LZMA-compressed
    /// so that exact reconstruction is possible.
    ///
    /// For `LzmaFallback`, the entire raw bytes are LZMA-compressed and stored
    /// in `payload.residual`.
    pub fn compress(&self, data: &[f32]) -> CompressedPayload {
        let n = data.len();
        let fit_result = analyze_data(data);

        match fit_result.method {
            FitMethod::LzmaFallback => {
                // Compress raw f32 bytes with LZMA
                let raw_bytes: Vec<u8> = data.iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                let residual = lzma_compress(&raw_bytes, 6).ok();
                CompressedPayload {
                    fit_result,
                    residual,
                    original_len: n,
                }
            }

            _ => {
                // Reconstruct and compute residual for near-lossless storage
                let reconstructed = reconstruct(&fit_result, n);

                let residual_f32: Vec<f32> = data.iter()
                    .zip(reconstructed.iter())
                    .map(|(&orig, &rec)| orig - rec)
                    .collect();

                // Check whether residual is significant
                let max_abs_residual = residual_f32.iter()
                    .map(|&v| v.abs())
                    .fold(0.0_f32, f32::max);

                let residual = if max_abs_residual > 1e-10 {
                    let residual_bytes: Vec<u8> = residual_f32.iter()
                        .flat_map(|&v| v.to_le_bytes())
                        .collect();
                    lzma_compress(&residual_bytes, 6).ok()
                } else {
                    None
                };

                CompressedPayload {
                    fit_result,
                    residual,
                    original_len: n,
                }
            }
        }
    }

    /// Decompress a `CompressedPayload` back into the original signal.
    pub fn decompress(&self, payload: &CompressedPayload) -> Vec<f32> {
        let n = payload.original_len;

        match payload.fit_result.method {
            FitMethod::LzmaFallback => {
                // Decompress raw LZMA bytes back to f32
                if let Some(ref compressed) = payload.residual {
                    if let Ok(raw) = lzma_decompress(compressed) {
                        return raw.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                            .collect();
                    }
                }
                vec![0.0; n]
            }

            _ => {
                let mut signal = reconstruct(&payload.fit_result, n);

                // Add residual if present
                if let Some(ref compressed_residual) = payload.residual {
                    if let Ok(raw) = lzma_decompress(compressed_residual) {
                        let residual: Vec<f32> = raw.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                            .collect();
                        for (s, r) in signal.iter_mut().zip(residual.iter()) {
                            *s += r;
                        }
                    }
                }

                signal
            }
        }
    }
}

impl Default for ProceduralCompressionDesigner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a pure sine wave with known parameters
    fn make_sine(n: usize, freq: f32, amp: f32, phase: f32, dc: f32) -> Vec<f32> {
        (0..n)
            .map(|i| dc + amp * (2.0 * PI * freq * i as f32 / n as f32 + phase).sin())
            .collect()
    }

    // ------------------------------------------------------------------
    // test_sine_fit_pure_sine
    // ------------------------------------------------------------------

    #[test]
    fn test_sine_fit_pure_sine() {
        let n = 512;
        let freq = 8.0_f32;
        let amp = 2.0_f32;
        let phase = 0.3_f32;
        let dc = 1.5_f32;

        let data = make_sine(n, freq, amp, phase, dc);
        let result = try_sine_fit(&data);

        assert!(result.is_some(), "Sine fit should succeed for a pure sine wave");
        let fit = result.unwrap();
        assert_eq!(fit.method, FitMethod::Sine);
        assert!(fit.error < MAX_ACCEPTABLE_ERROR,
            "Normalized error should be below threshold: {}", fit.error);

        // Coefficients: [frequency, amplitude, phase, dc_offset]
        assert_eq!(fit.coefficients.len(), 4);

        // Amplitude should be close to the true amplitude
        let fitted_amp = fit.coefficients[1].abs();
        assert!((fitted_amp - amp as f64).abs() < 0.5,
            "Amplitude mismatch: fitted={:.4}, true={:.4}", fitted_amp, amp);
    }

    // ------------------------------------------------------------------
    // test_fourier_fit
    // ------------------------------------------------------------------

    #[test]
    fn test_fourier_fit() {
        let n = 512;
        // Multi-frequency signal: two sine components
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (2.0 * PI * 3.0 * t).sin() * 1.5
                    + (2.0 * PI * 7.0 * t).sin() * 0.8
            })
            .collect();

        let result = try_fourier_fit(&data, MAX_FOURIER_COEFFICIENTS);
        assert!(result.is_some(), "Fourier fit should succeed for multi-sine signal");

        let fit = result.unwrap();
        assert_eq!(fit.method, FitMethod::Fourier);
        assert!(fit.error < MAX_ACCEPTABLE_ERROR,
            "Normalized error should be below threshold: {}", fit.error);
        // Must have at least one triple + dc_offset
        assert!(fit.coefficients.len() >= 4,
            "Must have at least one (idx, mag, phase) triple + dc_offset");
    }

    // ------------------------------------------------------------------
    // test_polynomial_fit
    // ------------------------------------------------------------------

    #[test]
    fn test_polynomial_fit() {
        let n = 200;
        // Quadratic: y = 2x^2 - x + 0.5 on [0, 1]
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                2.0 * x * x - x + 0.5
            })
            .collect();

        let result = try_polynomial_fit(&data, MAX_POLYNOMIAL_DEGREE);
        assert!(result.is_some(), "Polynomial fit should succeed for quadratic data");

        let fit = result.unwrap();
        assert_eq!(fit.method, FitMethod::Polynomial);
        assert!(fit.error < MAX_ACCEPTABLE_ERROR,
            "Normalized error should be below threshold: {}", fit.error);
    }

    // ------------------------------------------------------------------
    // test_analyze_data_selects_best
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_data_selects_best() {
        let n = 512;
        // Pure sine — procedural fit should win
        let data = make_sine(n, 5.0, 1.0, 0.0, 0.0);
        let result = analyze_data(&data);

        // Should be a procedural method (Sine or Fourier), not LZMA
        assert_ne!(result.method, FitMethod::LzmaFallback,
            "Pure sine should not fall back to LZMA");
        assert!(result.compression_ratio >= MIN_PROCEDURAL_RATIO,
            "Compression ratio should meet minimum threshold: {}", result.compression_ratio);
    }

    #[test]
    fn test_analyze_data_lzma_for_noise() {
        // White noise: no procedural fit should work well
        // Use a deterministic pseudo-random sequence instead of rand
        let n = 512;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                // Simple LCG-based deterministic noise in [-1, 1]
                let v = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let normalized = (v as f32 / u64::MAX as f32) * 2.0 - 1.0;
                normalized
            })
            .collect();

        let result = analyze_data(&data);
        // For random data the procedural fits should fail; LZMA is the fallback
        // (This assertion is probabilistic but deterministic with fixed seed above)
        assert_eq!(result.method, FitMethod::LzmaFallback,
            "Random noise should fall back to LZMA; method={:?}", result.method);
    }

    // ------------------------------------------------------------------
    // test_compress_decompress_roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_compress_decompress_roundtrip_sine() {
        let n = 512;
        let data = make_sine(n, 6.0, 1.5, 0.1, 0.5);

        let designer = ProceduralCompressionDesigner::new();
        let payload = designer.compress(&data);
        let recovered = designer.decompress(&payload);

        assert_eq!(recovered.len(), n);

        let max_err = data.iter().zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_err < 1e-4,
            "Roundtrip max error too high for sine: {}", max_err);
    }

    #[test]
    fn test_compress_decompress_roundtrip_polynomial() {
        let n = 256;
        // Cubic polynomial
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                x * x * x - 2.0 * x * x + x + 0.1
            })
            .collect();

        let designer = ProceduralCompressionDesigner::new();
        let payload = designer.compress(&data);
        let recovered = designer.decompress(&payload);

        assert_eq!(recovered.len(), n);

        let max_err = data.iter().zip(recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_err < 1e-4,
            "Roundtrip max error too high for polynomial: {}", max_err);
    }

    #[test]
    fn test_compress_decompress_roundtrip_lzma_fallback() {
        // Use deterministic noise that won't be procedurally compressed
        let n = 512;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let v = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (v as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let designer = ProceduralCompressionDesigner::new();
        let payload = designer.compress(&data);

        // LZMA fallback path must reconstruct exactly
        if payload.fit_result.method == FitMethod::LzmaFallback {
            let recovered = designer.decompress(&payload);
            assert_eq!(recovered.len(), n);
            for (a, b) in data.iter().zip(recovered.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(),
                    "LZMA fallback must be bit-exact");
            }
        }
    }
}
