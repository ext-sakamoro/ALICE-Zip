//! Polynomial regression / evaluation (one least-squares solver, two
//! coefficient conventions)
//!
//! - [`fit_polynomial`] / [`generate_polynomial`]: `x = 0, 1, …, n-1`,
//!   coefficients ascending (`c[0] + c[1]·x + c[2]·x² + …`) — the `alice-db`
//!   segment contract
//! - [`fit_polynomial_unit`] / [`generate_polynomial_unit`]: `x = i / (n-1)`
//!   normalised to `[0, 1]`, coefficients **descending** (`c[0]·x^k + … +
//!   c[k]`) — the `.alice` container contract (`libalice` ≤ 2.2 stored files
//!   this way)
//!
//! Both fit paths solve the same least-squares problem (Householder QR of the
//! Vandermonde matrix on `x ∈ [0, 1]`; the integer convention rescales the
//! coefficients afterwards) and both evaluate by Horner's rule Law pinned by `tests/analytic_oracle.rs`: data
//! sampled from an exact polynomial of degree `d ≤ max_degree` is recovered
//! with `degree == d` and coefficient error `≤ 1e-6` (relative)

use alloc::vec;
use alloc::vec::Vec;

// (test builds link std, whose inherent methods shadow the trait → allow)
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::FloatExt;

/// Relative size of an `R` diagonal entry (vs the largest) below which the
/// Vandermonde system is treated as rank-deficient
const RANK_TOLERANCE: f64 = 1e-13;
/// Variance below which the series is treated as constant (degree-0 fit)
const CONSTANT_VARIANCE: f64 = 1e-15;

/// Fit a polynomial `c[0] + c[1]*x + … + c[k]*x^k` to `values` at integer
/// indices `x = 0, 1, …, values.len() - 1`. Tries degrees from `1` up to
/// `max_degree` and returns the *first* fit whose normalised MSE is below
/// `error_threshold`. Returns `None` if no polynomial up to `max_degree`
/// satisfies the threshold (or `values.len() < 2`).
///
/// The error metric is `mean((f(i) - values[i])^2) / variance(values)`, i.e.
/// 1 - R², so a value of `0.01` means the polynomial explains 99% of the
/// variance. A constant series returns `Some((vec![mean], 0, 0.0))`.
#[must_use]
pub fn fit_polynomial(
    values: &[f32],
    max_degree: usize,
    error_threshold: f64,
) -> Option<(Vec<f64>, usize, f64)> {
    // Fit on x/(n-1) ∈ [0, 1] (well conditioned Vandermonde), then map the
    // coefficients back: Σ a_j (i h)^j = Σ (a_j h^j) i^j with h = 1/(n-1)
    let xs = unit_xs(values.len());
    let (mut coeffs, degree, err) = fit_generic(&xs, values, max_degree, error_threshold)?;
    #[allow(clippy::cast_precision_loss)]
    let h = 1.0 / (values.len().saturating_sub(1).max(1)) as f64;
    let mut scale = 1.0;
    for c in &mut coeffs {
        *c *= scale;
        scale *= h;
    }
    Some((coeffs, degree, err))
}

/// Evaluate a polynomial with **ascending** coefficients at integer indices
/// `x = 0, 1, …, n-1` and return the samples as `f32`.
#[must_use]
pub fn generate_polynomial(n: usize, coefficients: &[f64]) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    (0..n)
        .map(|i| horner_ascending(coefficients, i as f64) as f32)
        .collect()
}

/// Fit a polynomial to `values` with `x` normalised to `[0, 1]`
/// (`x_i = i / (n - 1)`) and return **descending** coefficients
/// (`c[0]·x^k + … + c[k]`) — the `.alice` container convention.
///
/// Same search / error metric as [`fit_polynomial`]; decode the result with
/// [`generate_polynomial_unit`].
#[must_use]
pub fn fit_polynomial_unit(
    values: &[f32],
    max_degree: usize,
    error_threshold: f64,
) -> Option<(Vec<f64>, usize, f64)> {
    let xs = unit_xs(values.len());
    let (mut coeffs, degree, err) = fit_generic(&xs, values, max_degree, error_threshold)?;
    coeffs.reverse();
    Some((coeffs, degree, err))
}

/// Evaluate a polynomial with **descending** coefficients at
/// `x_i = i / (n - 1)` (`x = 0` when `n == 1`) and return the samples as
/// `f32`. Empty `coefficients` yield `n` zeros.
#[must_use]
pub fn generate_polynomial_unit(n: usize, coefficients: &[f64]) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation)]
    unit_xs(n)
        .into_iter()
        .map(|x| horner_descending(coefficients, x) as f32)
        .collect()
}

// ---------- internals ----------

#[allow(clippy::cast_precision_loss)]
fn unit_xs(n: usize) -> Vec<f64> {
    let rcp = 1.0 / (n.saturating_sub(1).max(1)) as f64;
    (0..n).map(|i| i as f64 * rcp).collect()
}

/// Horner's rule for ascending coefficients: `c0 + x*(c1 + x*(c2 + …))`
#[inline]
fn horner_ascending(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c)
}

/// Horner's rule for descending coefficients
#[inline]
fn horner_descending(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0_f64, |acc, &c| acc * x + c)
}

/// Shared search: ascending coefficients for the sample positions `xs`
#[allow(clippy::cast_precision_loss)]
fn fit_generic(
    xs: &[f64],
    values: &[f32],
    max_degree: usize,
    error_threshold: f64,
) -> Option<(Vec<f64>, usize, f64)> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let ys: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
    let inv_n = 1.0 / n as f64;
    let mean = ys.iter().sum::<f64>() * inv_n;
    let variance = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() * inv_n;
    if variance < CONSTANT_VARIANCE {
        return Some((vec![mean], 0, 0.0));
    }

    for degree in 1..=max_degree.min(n - 1) {
        let Some(coeffs) = least_squares_fit(xs, &ys, degree) else {
            continue;
        };
        let mse = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| {
                let d = y - horner_ascending(&coeffs, x);
                d * d
            })
            .sum::<f64>()
            * inv_n;
        let rel = mse / variance;
        if rel < error_threshold {
            return Some((coeffs, degree, rel));
        }
    }
    None
}

/// Least-squares polynomial fit: Householder QR of the `n × m` Vandermonde
/// matrix (`m = degree + 1`), then back-substitution `R c = Qᵀ y`
///
/// QR works on the matrix itself, so the condition number is that of the
/// Vandermonde matrix and not its square (normal equations lost 4-5 digits at
/// degree 5 on `[0, 1]`) Returns ascending coefficients, or `None` when
/// `n < m` or the matrix is rank-deficient (repeated `x`)
fn least_squares_fit(xs: &[f64], ys: &[f64], degree: usize) -> Option<Vec<f64>> {
    let n = xs.len();
    let m = degree + 1;
    if n < m {
        return None;
    }
    // Column-major Vandermonde: a[j * n + i] = x_i^j
    let mut a = vec![0.0_f64; n * m];
    for (i, &x) in xs.iter().enumerate() {
        let mut pow = 1.0;
        for j in 0..m {
            a[j * n + i] = pow;
            pow *= x;
        }
    }
    let mut b = ys.to_vec();
    let mut diag = vec![0.0_f64; m];

    for k in 0..m {
        // Householder vector for column k, rows k..n
        let norm = a[k * n + k..(k + 1) * n]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return None;
        }
        let alpha = if a[k * n + k] > 0.0 { -norm } else { norm };
        // v = x - alpha e_k, stored in place of column k (rows k..n)
        a[k * n + k] -= alpha;
        let vnorm2: f64 = a[k * n + k..(k + 1) * n].iter().map(|v| v * v).sum();
        if vnorm2 == 0.0 {
            return None;
        }
        // Apply H = I - 2 v vᵀ / (vᵀ v) to the remaining columns and to b
        for j in (k + 1)..m {
            let dot: f64 = (k..n).map(|i| a[k * n + i] * a[j * n + i]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in k..n {
                a[j * n + i] -= f * a[k * n + i];
            }
        }
        let dot: f64 = (k..n).map(|i| a[k * n + i] * b[i]).sum();
        let f = 2.0 * dot / vnorm2;
        for i in k..n {
            b[i] -= f * a[k * n + i];
        }
        diag[k] = alpha;
    }

    let r_max = diag.iter().fold(0.0_f64, |acc, d| acc.max(d.abs()));
    if diag.iter().any(|d| d.abs() <= RANK_TOLERANCE * r_max) {
        return None;
    }

    // Back-substitution with R (upper triangle: R[k][j] = a[j*n + k] for j > k)
    let mut coeffs = vec![0.0_f64; m];
    for k in (0..m).rev() {
        let mut sum = b[k];
        for j in (k + 1)..m {
            sum -= a[j * n + k] * coeffs[j];
        }
        coeffs[k] = sum / diag[k];
        if !coeffs[k].is_finite() {
            return None;
        }
    }
    Some(coeffs)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
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
        assert_eq!(generate_polynomial(3, &[]), vec![0.0; 3]);
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
            .map(|i| 1.0 + i as f32 + (i as f32) * (i as f32))
            .collect();
        let (coefs, deg, _) = fit_polynomial(&values, 4, 1e-4).expect("quadratic fit");
        assert_eq!(deg, 2);
        assert!((coefs[0] - 1.0).abs() < 1e-3);
        assert!((coefs[1] - 1.0).abs() < 1e-3);
        assert!((coefs[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn fit_polynomial_noise_returns_none() {
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
    fn fit_polynomial_constant_is_degree_zero_mean() {
        let (c, d, e) = fit_polynomial(&[4.5; 10], 3, 0.01).unwrap();
        assert_eq!((c, d, e), (vec![4.5], 0, 0.0));
    }

    #[test]
    fn generate_polynomial_unit_quadratic() {
        // y = x^2 on [0, 1], descending coefficients
        let signal = generate_polynomial_unit(101, &[1.0, 0.0, 0.0]);
        assert_eq!(signal.len(), 101);
        assert!(signal[0].abs() < 1e-6);
        assert!((signal[50] - 0.25).abs() < 1e-6);
        assert!((signal[100] - 1.0).abs() < 1e-6);
        assert_eq!(generate_polynomial_unit(3, &[]), vec![0.0; 3]);
        // n == 1 evaluates at x = 0 → constant term
        assert_eq!(generate_polynomial_unit(1, &[7.0, 5.0]), vec![5.0]);
    }

    #[test]
    fn fit_polynomial_unit_recovers_descending_coefficients() {
        // y = x^2 + 2x + 1 on [0, 1]
        let data: Vec<f32> = (0..100)
            .map(|i| {
                let x = i as f32 / 99.0;
                x * x + 2.0 * x + 1.0
            })
            .collect();
        let (coeffs, degree, error) = fit_polynomial_unit(&data, 5, 1e-6).unwrap();
        assert_eq!(degree, 2);
        assert!(error < 1e-6);
        assert!((coeffs[0] - 1.0).abs() < 1e-3, "{coeffs:?}");
        assert!((coeffs[1] - 2.0).abs() < 1e-3, "{coeffs:?}");
        assert!((coeffs[2] - 1.0).abs() < 1e-3, "{coeffs:?}");
        let recon = generate_polynomial_unit(100, &coeffs);
        for (a, b) in data.iter().zip(&recon) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn horner_conventions_are_mirrors() {
        let asc = [1.0, 2.0, 3.0];
        let desc = [3.0, 2.0, 1.0];
        for x in [0.0, 0.5, 1.0, 2.0, -1.5] {
            assert_eq!(horner_ascending(&asc, x), horner_descending(&desc, x));
        }
    }
}
