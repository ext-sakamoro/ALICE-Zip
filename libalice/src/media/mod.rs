//! Media Generators for ALICE-Zip
//!
//! Rust port of the Python media_generators.py module.
//! Provides procedural image and audio generation.
//!
//! License: MIT
//! Author: Moroya Sakamoto

pub mod image;
pub mod audio;

pub use image::{ImageGenerator, ImagePattern, ImageParams};
pub use audio::{AudioGenerator, AudioPattern, AudioParams};
