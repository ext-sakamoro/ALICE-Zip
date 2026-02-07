//! Fourier Series Generator
//!
//! Uses rustfft for high-performance FFT-based signal reconstruction.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::cell::RefCell;

// Thread-local FftPlanner cache for performance optimization.
// FftPlanner internally caches FFT plans, but creating a new planner each call
// loses that cache. This thread-local storage preserves the cache across calls.
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

/// Coefficient for Fourier reconstruction
#[derive(Debug, Clone, Copy)]
pub struct FourierCoefficient {
    pub freq_idx: usize,
    pub magnitude: f32,
    pub phase: f32,
}

/// Generate signal from Fourier coefficients using inverse FFT
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `coefficients` - List of (freq_idx, magnitude, phase) tuples
/// * `dc_offset` - DC component (mean value)
///
/// # Returns
/// Vec<f32> of generated signal
pub fn generate_from_coefficients(
    n: usize,
    coefficients: &[(usize, f32, f32)],
    dc_offset: f32,
) -> Vec<f32> {
    // Build complex FFT buffer
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n];

    for &(freq_idx, magnitude, phase) in coefficients {
        if freq_idx < n {
            // Set positive frequency
            buffer[freq_idx] = Complex::from_polar(magnitude, phase);

            // Set negative frequency (conjugate symmetry for real signal)
            if freq_idx > 0 && freq_idx < n / 2 {
                buffer[n - freq_idx] = Complex::from_polar(magnitude, -phase);
            }
        }
    }

    // Perform inverse FFT using cached planner
    FFT_PLANNER.with(|planner| {
        let ifft = planner.borrow_mut().plan_fft_inverse(n);
        ifft.process(&mut buffer);
    });

    // Extract real parts and add DC offset
    // Note: rustfft doesn't normalize, so we divide by n
    let inv_n = 1.0 / n as f32;
    buffer.iter()
        .map(|c| c.re * inv_n + dc_offset)
        .collect()
}

/// Generate a simple sine wave
///
/// # Arguments
/// * `n` - Number of samples
/// * `frequency` - Frequency (cycles per n samples)
/// * `amplitude` - Wave amplitude
/// * `phase` - Initial phase in radians
/// * `dc_offset` - DC offset
///
/// # Returns
/// Vec<f32> of sine wave samples
pub fn generate_sine_wave(
    n: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    dc_offset: f32,
) -> Vec<f32> {
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            let t = i as f32;
            dc_offset + amplitude * (2.0 * PI * frequency * t * inv_n + phase).sin()
        })
        .collect()
}

/// Generate a multi-tone signal (sum of sine waves)
///
/// # Arguments
/// * `n` - Number of samples
/// * `components` - List of (frequency, amplitude, phase) tuples
/// * `dc_offset` - DC offset
///
/// # Returns
/// Vec<f32> of generated signal
pub fn generate_multi_sine(
    n: usize,
    components: &[(f32, f32, f32)],
    dc_offset: f32,
) -> Vec<f32> {
    let mut result = vec![dc_offset; n];
    let inv_n = 1.0 / n as f32;

    for &(freq, amp, phase) in components {
        for (i, sample) in result.iter_mut().enumerate() {
            let t = i as f32;
            *sample += amp * (2.0 * PI * freq * t * inv_n + phase).sin();
        }
    }

    result
}

/// Analyze signal using FFT and extract dominant frequencies
///
/// # Arguments
/// * `signal` - Input signal
/// * `max_coefficients` - Maximum number of coefficients to return
/// * `energy_threshold` - Capture this fraction of total energy (0.0-1.0)
///
/// # Returns
/// Tuple of (coefficients, dc_offset)
pub fn analyze_signal(
    signal: &[f32],
    max_coefficients: usize,
    energy_threshold: f32,
) -> (Vec<(usize, f32, f32)>, f32) {
    let n = signal.len();
    let inv_n = 1.0 / n as f32;
    let dc_offset = signal.iter().sum::<f32>() * inv_n;

    // Remove DC and prepare for FFT
    let mut buffer: Vec<Complex<f32>> = signal
        .iter()
        .map(|&x| Complex::new(x - dc_offset, 0.0))
        .collect();

    // Forward FFT using cached planner
    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    // Calculate magnitudes and phases
    let half_n = n / 2;
    let mut freq_data: Vec<(usize, f32, f32)> = buffer[..half_n]
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mag = c.norm();
            let phase = c.arg();
            (i, mag, phase)
        })
        .collect();

    // Sort by magnitude (descending), handling NaN values safely
    // NaN values are treated as less than any real number
    freq_data.sort_by(|a, b| {
        match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,  // NaN goes to end
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    // Calculate total energy
    let total_energy: f32 = freq_data.iter().map(|(_, m, _)| m * m).sum();

    // Select coefficients until energy threshold is reached
    let mut coefficients = Vec::new();
    let mut captured_energy = 0.0f32;

    for (idx, mag, phase) in freq_data {
        if coefficients.len() >= max_coefficients {
            break;
        }
        if mag < 1e-10 {
            continue;
        }

        coefficients.push((idx, mag, phase));
        captured_energy += mag * mag;

        if total_energy > 0.0 && captured_energy / total_energy > energy_threshold {
            break;
        }
    }

    (coefficients, dc_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave() {
        let signal = generate_sine_wave(1000, 10.0, 1.0, 0.0, 0.0);
        assert_eq!(signal.len(), 1000);

        // Check that it's periodic
        let period = 100; // 1000 / 10 = 100 samples per cycle
        for i in 0..100 {
            let diff = (signal[i] - signal[i + period]).abs();
            assert!(diff < 0.01, "Not periodic at index {}", i);
        }
    }

    #[test]
    fn test_fourier_roundtrip() {
        // Create a signal
        let original = generate_sine_wave(256, 5.0, 1.0, 0.0, 0.5);

        // Analyze it
        let (coeffs, dc) = analyze_signal(&original, 10, 0.99);

        // Reconstruct it
        let reconstructed = generate_from_coefficients(256, &coeffs, dc);

        // Check similarity
        let mse: f32 = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;

        assert!(mse < 0.01, "MSE too high: {}", mse);
    }
}
