//! Procedural Generators Module
//!
//! High-performance implementations of procedural generation algorithms.
//!
//! License: MIT
//! Author: Moroya Sakamoto

pub mod perlin;
pub mod fourier;
pub mod polynomial;

// Re-export main functions
pub use perlin::{generate_perlin_2d, generate_perlin_advanced, PerlinNoise};
pub use fourier::{
    generate_from_coefficients, generate_sine_wave, generate_multi_sine, analyze_signal
};
pub use polynomial::{generate_polynomial, fit_polynomial};
