//! Procedural signal generators — the laws that let downstream crates store
//! *the function that produces the signal* instead of the raw samples
//!
//! This is the "Send laws, not data" principle applied to time-series,
//! numerical arrays and textures: instead of persisting `n` samples verbatim,
//! fit a compact set of parameters (polynomial / Fourier coefficients, a
//! noise seed) that regenerates them
//!
//! Every law lives here exactly once The `libalice` CLI / FFI / Python crate
//! and `alice-db` both consume these functions; there is no second copy of
//! any generator anywhere in the repository (2026-09-15 unification)
//!
//! # Conventions (two persisted formats, two explicit names)
//!
//! | family | function | x domain | coefficient order | persisted by |
//! |--------|----------|----------|-------------------|--------------|
//! | polynomial | [`fit_polynomial`] / [`generate_polynomial`] | `x = 0, 1, …, n-1` | ascending (`c[0] + c[1]x + …`) | `alice-db` segments |
//! | polynomial | [`fit_polynomial_unit`] / [`generate_polynomial_unit`] | `x = i / (n-1) ∈ [0, 1]` | **descending** (`c[0]x^k + … + c[k]`) | `.alice` container (`libalice`) |
//! | Fourier | [`analyze_signal`] / [`generate_from_coefficients`] | bins `1..=n/2` | `(k, magnitude, phase)` | both |
//! | Fourier | `analyze_signal_fft` (`fft` feature) | same contract as `analyze_signal`, O(n log n) | | `libalice` |
//! | sinusoid | [`generate_sine_wave`] / [`generate_multi_sine`] | `2π f i / n + φ` | | both |
//! | noise | [`generate_fbm_1d`] | 1D value noise (hash lattice) | | `alice-db` `PerlinNoise` model |
//! | noise | [`PerlinNoise`] / [`generate_perlin_2d`] / [`generate_perlin_advanced`] | 2D gradient Perlin (seeded permutation) | | `.alice` container, FFI, Python |
//!
//! The two polynomial conventions are *different laws* that happen to share a
//! solver; they are named apart so a caller can never pass coefficients of one
//! format into the evaluator of the other by accident
//!
//! # Precision
//!
//! Naive DFT and Gaussian elimination over `f64`; the `fft` feature adds a
//! rustfft path with the same output contract (`tests/analytic_oracle.rs`
//! pins naive-vs-FFT parity) Under `no_std` transcendental functions go
//! through `libm` and may differ from the `std` build in the last ulp

mod fourier;
mod perlin;
mod polynomial;

#[cfg(feature = "fft")]
pub use fourier::analyze_signal_fft;
pub use fourier::{
    analyze_signal, generate_from_coefficients, generate_multi_sine, generate_sine_wave,
};
pub use perlin::{generate_fbm_1d, generate_perlin_2d, generate_perlin_advanced, PerlinNoise};
pub use polynomial::{
    fit_polynomial, fit_polynomial_unit, generate_polynomial, generate_polynomial_unit,
};
