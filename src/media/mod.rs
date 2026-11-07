//! Media Generators for ALICE-Zip
//!
//! Procedural image and audio generation.
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

pub mod audio;
pub mod image;

pub use audio::{AudioGenerator, AudioParams, AudioPattern};
pub use image::{ImageGenerator, ImageParams, ImagePattern};
