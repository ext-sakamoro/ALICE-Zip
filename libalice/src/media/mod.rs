//! Media Generators for ALICE-Zip
//!
//! Rust port of the Python media_generators.py module.
//! Provides procedural image and audio generation.
//!
//! License: MIT
//! Author: Moroya Sakamoto

pub mod audio;
pub mod image;

pub use audio::{AudioGenerator, AudioParams, AudioPattern};
pub use image::{ImageGenerator, ImageParams, ImagePattern};
