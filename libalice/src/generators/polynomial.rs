//! Polynomial Generator
//!
//! Evaluates polynomials for signal reconstruction.
//! Uses Horner's method for efficient evaluation.
//!
//! License: MIT
//! Author: Moroya Sakamoto

// Numerical precision constants
// Note: Input data is f32 (≈7 significant digits, epsilon ≈ 1.19e-7),
// but internal calculations use f64 (≈15 significant digits, epsilon ≈ 2.22e-16).
//
// SINGULAR_MATRIX_THRESHOLD rationale:
// - Too strict (1e-12): May reject valid fits due to accumulated round-off errors
// - Too lenient (1e-6): May accept numerically unstable solutions
// - 1e-8 is a balanced choice: accounts for f32 input precision while allowing
//   for numerical stability in higher-degree polynomial fitting (up to degree 9)
const SINGULAR_MATRIX_THRESHOLD: f64 = 1e-8;  // Balanced threshold for f32-derived data
const CONSTANT_DATA_VARIANCE_THRESHOLD: f64 = 1e-15;  // Threshold for detecting constant data

/// Generate signal from polynomial coefficients
///
/// The polynomial is evaluated as:
///   y = c[0] * x^n + c[1] * x^(n-1) + ... + c[n-1] * x + c[n]
///
/// Where x is normalized to [0, 1] range.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `coefficients` - Polynomial coefficients (highest degree first)
///
/// # Returns
/// Vec<f32> of generated signal
pub fn generate_polynomial(n: usize, coefficients: &[f64]) -> Vec<f32> {
    if coefficients.is_empty() {
        return vec![0.0; n];
    }

    (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1).max(1) as f64;
            horner_eval(x, coefficients) as f32
        })
        .collect()
}

/// Evaluate polynomial using Horner's method
///
/// More numerically stable and efficient than naive evaluation.
/// Coefficients are ordered from highest degree to lowest.
#[inline]
fn horner_eval(x: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Fit a polynomial to data using least squares
///
/// # Arguments
/// * `data` - Input data points
/// * `max_degree` - Maximum polynomial degree to try
/// * `error_threshold` - Stop if relative error is below this
///
/// # Returns
/// Tuple of (coefficients, degree, relative_error)
pub fn fit_polynomial(
    data: &[f32],
    max_degree: usize,
    error_threshold: f64,
) -> Option<(Vec<f64>, usize, f64)> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    // Normalize x to [0, 1]
    let x_data: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1).max(1) as f64)
        .collect();
    let y_data: Vec<f64> = data.iter().map(|&y| y as f64).collect();

    let data_variance = variance(&y_data);
    if data_variance < CONSTANT_DATA_VARIANCE_THRESHOLD {
        // Constant data - return mean as degree-0 polynomial
        let mean = y_data.iter().sum::<f64>() / n as f64;
        return Some((vec![mean], 0, 0.0));
    }

    for degree in 1..=max_degree.min(n - 1) {
        if let Some(coeffs) = least_squares_fit(&x_data, &y_data, degree) {
            let fitted: Vec<f64> = x_data.iter()
                .map(|&x| horner_eval(x, &coeffs))
                .collect();

            let mse = mean_squared_error(&y_data, &fitted);
            let relative_error = mse / data_variance;

            if relative_error < error_threshold {
                return Some((coeffs, degree, relative_error));
            }
        }
    }

    None
}

/// Least squares polynomial fitting
///
/// Solves the normal equations: (X^T X) c = X^T y
fn least_squares_fit(x: &[f64], y: &[f64], degree: usize) -> Option<Vec<f64>> {
    let n = x.len();
    let m = degree + 1;

    // Build Vandermonde matrix
    let mut xtx = vec![0.0f64; m * m];
    let mut xty = vec![0.0f64; m];

    for i in 0..n {
        let mut xi_pow = vec![1.0f64; m];
        for j in 1..m {
            xi_pow[j] = xi_pow[j - 1] * x[i];
        }

        // X^T X
        for j in 0..m {
            for k in 0..m {
                xtx[j * m + k] += xi_pow[m - 1 - j] * xi_pow[m - 1 - k];
            }
            // X^T y
            xty[j] += xi_pow[m - 1 - j] * y[i];
        }
    }

    // Solve using Gaussian elimination with partial pivoting
    solve_linear_system(&mut xtx, &mut xty, m)
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for k in (i + 1)..n {
            if a[k * n + i].abs() > max_val {
                max_val = a[k * n + i].abs();
                max_row = k;
            }
        }

        if max_val < SINGULAR_MATRIX_THRESHOLD {
            return None; // Singular or near-singular matrix
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                a.swap(i * n + j, max_row * n + j);
            }
            b.swap(i, max_row);
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = a[k * n + i] / a[i * n + i];
            for j in i..n {
                a[k * n + j] -= factor * a[i * n + j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let pivot = a[i * n + i];
        // Safety check: pivot should never be zero after pivoting, but verify
        if pivot.abs() < SINGULAR_MATRIX_THRESHOLD {
            return None;  // Numerical instability detected
        }

        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / pivot;

        // Check for NaN or Inf which indicates numerical instability
        if !x[i].is_finite() {
            return None;
        }
    }

    Some(x)
}

/// Calculate variance
fn variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Calculate mean squared error
fn mean_squared_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>() / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_generation() {
        // y = x^2 on [0, 1]
        let coeffs = vec![1.0, 0.0, 0.0]; // x^2 + 0*x + 0
        let signal = generate_polynomial(101, &coeffs);

        assert_eq!(signal.len(), 101);
        assert!((signal[0] - 0.0).abs() < 0.001);
        assert!((signal[50] - 0.25).abs() < 0.001); // (0.5)^2 = 0.25
        assert!((signal[100] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_polynomial_fitting() {
        // Generate quadratic data
        let data: Vec<f32> = (0..100)
            .map(|i| {
                let x = i as f32 / 99.0;
                x * x + 2.0 * x + 1.0
            })
            .collect();

        let result = fit_polynomial(&data, 5, 0.001);
        assert!(result.is_some());

        let (coeffs, degree, error) = result.unwrap();
        assert!(degree <= 3);
        assert!(error < 0.001);

        // Reconstruct and compare
        let reconstructed = generate_polynomial(100, &coeffs);
        let mse: f32 = data.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / data.len() as f32;

        assert!(mse < 0.001, "MSE too high: {}", mse);
    }
}
