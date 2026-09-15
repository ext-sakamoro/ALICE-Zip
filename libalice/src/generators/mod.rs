//! Procedural generators — thin re-export of the laws in the `alice-zip` core
//! crate (`../src/generators/`)
//!
//! Until 2.2.0 this module carried its own copies of the polynomial / Fourier
//! / Perlin generators; they diverged from the core crate (different
//! coefficient conventions under the same names, a Nyquist-bin factor, an
//! unused `_dimension` argument). Since 2.3.0 every law lives exactly once in
//! `alice_zip::generators` and this module only fixes the *conventions* the
//! `.alice` container, the FFI and the Python bindings persist:
//!
//! | libalice name | core function | convention |
//! |---------------|---------------|------------|
//! | [`fit_polynomial`] / [`generate_polynomial`] | `fit_polynomial_unit` / `generate_polynomial_unit` | `x = i/(n-1)`, **descending** coefficients (`.alice` container, Python) |
//! | [`analyze_signal`] | `analyze_signal_fft` | rustfft, bins `1..=n/2` |
//! | [`generate_from_coefficients`] / [`generate_sine_wave`] / [`generate_multi_sine`] | same | |
//! | [`PerlinNoise`] / [`generate_perlin_2d`] / [`generate_perlin_advanced`] | same | 2D gradient Perlin, `Result` (invalid `scale` / `octaves` are errors) |
//!
//! The C FFI (`alice_polynomial_generate`) documents `y = c0 + c1·x + …` at
//! `x = 0..n-1` and therefore binds the core `generate_polynomial` (ascending,
//! integer `x`) directly — see `ffi.rs`

pub use alice_zip::generators::{
    analyze_signal_fft as analyze_signal, fit_polynomial_unit as fit_polynomial,
    generate_from_coefficients, generate_multi_sine, generate_perlin_2d, generate_perlin_advanced,
    generate_polynomial_unit as generate_polynomial, generate_sine_wave, PerlinNoise,
};
