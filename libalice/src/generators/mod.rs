//! Procedural Generators Module
//!
//! High-performance implementations of procedural generation algorithms.
//!
//! License: MIT
//! Author: Moroya Sakamoto

pub mod fourier;
pub mod perlin;
pub mod polynomial;

// Re-export main functions
pub use fourier::{
    analyze_signal, generate_from_coefficients, generate_multi_sine, generate_sine_wave,
};
pub use perlin::{generate_perlin_2d, generate_perlin_advanced, PerlinNoise};
pub use polynomial::{fit_polynomial, generate_polynomial};
