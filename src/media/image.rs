//! Image Generator
//!
//! Generates procedural images from parametric descriptions. The Python package calls this through the
//! extension (`alice_zip.native_accelerator.media_image`). The bits of each pattern are golden entries of
//! [`crate::semantics::ZIP_SEMANTICS_PINS`].
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

use crate::generators::generate_perlin_2d;

/// The octaves of [`ImagePattern::Fractal`]
const FRACTAL_OCTAVES: u32 = 5;

// ============================================================================
// Enums
// ============================================================================

/// Detected / requested image patterns (mirrors Python `ImagePattern` enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePattern {
    Solid,
    LinearGradient,
    RadialGradient,
    Checkerboard,
    Noise,
    Fractal,
}

// ============================================================================
// Params
// ============================================================================

/// Parameters for procedural image generation
///
/// Mirrors Python `ImageParams` dataclass.
#[derive(Debug, Clone)]
pub struct ImageParams {
    /// Which pattern to generate
    pub pattern: ImagePattern,
    /// Output image width in pixels
    pub width: usize,
    /// Output image height in pixels
    pub height: usize,
    /// Two colours: index 0 = start/color1, index 1 = end/color2.
    /// For Solid only index 0 is used.
    pub colors: Vec<[u8; 3]>,
    /// Cell size for Checkerboard (pixels per square) and the lattice cell of Noise and Fractal (Noise and Fractal take at least 2: on a cell of 1 every pixel is a lattice point, where the noise is 0)
    pub block_size: usize,
    /// Gradient direction in degrees (0 = left-to-right, 90 = top-to-bottom)
    pub angle_degrees: f32,
    /// Seed of the texture of Noise and Fractal
    pub seed: u64,
    /// Centre of RadialGradient, as a fraction of the last pixel index (0.5 is the middle of the image)
    pub center: [f32; 2],
}

impl Default for ImageParams {
    fn default() -> Self {
        Self {
            pattern: ImagePattern::Solid,
            width: 64,
            height: 64,
            colors: vec![[0, 0, 0], [255, 255, 255]],
            block_size: 8,
            angle_degrees: 0.0,
            seed: 42,
            center: [0.5, 0.5],
        }
    }
}

// ============================================================================
// Generator
// ============================================================================

/// Procedural image generator
pub struct ImageGenerator;

impl ImageGenerator {
    /// Create a new generator (stateless — always succeeds)
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate RGB pixel data (width * height * 3 bytes, row-major)
    #[must_use]
    pub fn generate(&self, params: &ImageParams) -> Vec<u8> {
        let start = params.colors.first().copied().unwrap_or([0, 0, 0]);
        let end = params.colors.get(1).copied().unwrap_or([255, 255, 255]);
        self.generate_channels(params, &start, &end)
    }

    /// Generate pixel data with `start.len()` channels (width * height * channels bytes, row-major), from
    /// `start` to `end`; `params.colors` is not read
    ///
    /// # Panics
    ///
    /// When `start` is empty, `end` has another length, or width * height * channels overflows
    #[must_use]
    pub fn generate_channels(&self, params: &ImageParams, start: &[u8], end: &[u8]) -> Vec<u8> {
        assert!(
            !start.is_empty() && start.len() == end.len(),
            "start and end name the same channels"
        );
        let (w, h) = (params.width, params.height);
        assert!(
            w.checked_mul(h)
                .and_then(|n| n.checked_mul(start.len()))
                .is_some(),
            "the image size overflows"
        );
        match params.pattern {
            ImagePattern::Solid => Self::mix(w * h, start, end, |_| 0.0),
            ImagePattern::LinearGradient => {
                Self::generate_linear_gradient(w, h, start, end, params.angle_degrees)
            }
            ImagePattern::RadialGradient => {
                Self::generate_radial_gradient(w, h, start, end, params.center)
            }
            ImagePattern::Checkerboard => {
                Self::generate_checkerboard(w, h, start, end, params.block_size.max(1))
            }
            ImagePattern::Noise => Self::generate_perlin(w, h, start, end, params, 1),
            ImagePattern::Fractal => {
                Self::generate_perlin(w, h, start, end, params, FRACTAL_OCTAVES)
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    /// The law every pattern ends in: for each of `n` pixels, each channel is `start` to `end` at `t(pixel)`,
    /// rounded to the nearest level
    fn mix(n: usize, start: &[u8], end: &[u8], t: impl Fn(usize) -> f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * start.len());
        for i in 0..n {
            let t = t(i);
            for (&s, &e) in start.iter().zip(end) {
                let v = f32::from(s) * (1.0 - t) + f32::from(e) * t;
                out.push(alice_det_math::round(v).clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    /// `start` and `end` mixed by the seeded Perlin texture, with `block_size` pixels to a lattice cell: one
    /// octave is noise, several are fractal Brownian motion
    fn generate_perlin(
        w: usize,
        h: usize,
        start: &[u8],
        end: &[u8],
        params: &ImageParams,
        octaves: u32,
    ) -> Vec<u8> {
        let scale = params.block_size.max(2) as f32;
        let texture = generate_perlin_2d(w, h, params.seed, scale, octaves)
            .expect("the scale is at least 2, there is an octave, and the size fits a buffer");
        Self::mix(texture.len(), start, end, |i| texture[i])
    }

    /// Linear gradient along an angle in degrees: 0 is left to right, 90 top to bottom
    fn generate_linear_gradient(
        w: usize,
        h: usize,
        start: &[u8],
        end: &[u8],
        angle_degrees: f32,
    ) -> Vec<u8> {
        let rcp_w = if w > 1 { 1.0 / (w - 1) as f32 } else { 1.0 };
        let rcp_h = if h > 1 { 1.0 / (h - 1) as f32 } else { 1.0 };
        let angle_rad = angle_degrees * (PI / 180.0);
        let nx = alice_det_math::cos(angle_rad);
        let ny = alice_det_math::sin(angle_rad);
        let divisor = w as f32 * nx.abs() + h as f32 * ny.abs();
        let rcp_divisor = if divisor > 1e-10 { 1.0 / divisor } else { 1.0 };
        let width = w.max(1);
        Self::mix(w * h, start, end, |i| {
            let (x, y) = ((i % width) as f32, (i / width) as f32);
            if angle_degrees.abs() < f32::EPSILON {
                x * rcp_w
            } else if (angle_degrees - 90.0).abs() < f32::EPSILON {
                y * rcp_h
            } else {
                ((x * nx + y * ny) * rcp_divisor).clamp(0.0, 1.0)
            }
        })
    }

    /// Radial gradient: `start` at the centre, `end` at the farthest corner
    fn generate_radial_gradient(
        w: usize,
        h: usize,
        start: &[u8],
        end: &[u8],
        center: [f32; 2],
    ) -> Vec<u8> {
        let last_x = w.saturating_sub(1) as f32;
        let last_y = h.saturating_sub(1) as f32;
        let (cx, cy) = (center[0] * last_x, center[1] * last_y);
        let dx_max = alice_det_math::max(cx, last_x - cx);
        let dy_max = alice_det_math::max(cy, last_y - cy);
        let max_dist = alice_det_math::hypot(dx_max, dy_max);
        let width = w.max(1);
        Self::mix(w * h, start, end, |i| {
            if max_dist <= 1e-10 {
                return 0.0;
            }
            let dx = (i % width) as f32 - cx;
            let dy = (i / width) as f32 - cy;
            (alice_det_math::hypot(dx, dy) / max_dist).clamp(0.0, 1.0)
        })
    }

    /// Checkerboard of cells of `block_size` pixels, the cell at the origin being `start`
    fn generate_checkerboard(
        w: usize,
        h: usize,
        start: &[u8],
        end: &[u8],
        block_size: usize,
    ) -> Vec<u8> {
        let width = w.max(1);
        Self::mix(w * h, start, end, |i| {
            let (x, y) = (i % width, i / width);
            if (x / block_size + y / block_size).is_multiple_of(2) {
                0.0
            } else {
                1.0
            }
        })
    }
}

impl Default for ImageGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(pattern: ImagePattern) -> ImageParams {
        ImageParams {
            pattern,
            width: 16,
            height: 16,
            colors: vec![[255, 0, 0], [0, 0, 255]],
            block_size: 4,
            angle_degrees: 0.0,
            seed: 0,
            ..ImageParams::default()
        }
    }

    #[test]
    fn test_solid_color() {
        let gen = ImageGenerator::new();
        let mut p = make_params(ImagePattern::Solid);
        p.colors = vec![[42, 100, 200]];
        let pixels = gen.generate(&p);

        assert_eq!(pixels.len(), 16 * 16 * 3);
        // Every pixel must equal the solid colour
        for chunk in pixels.chunks(3) {
            assert_eq!(chunk[0], 42);
            assert_eq!(chunk[1], 100);
            assert_eq!(chunk[2], 200);
        }
    }

    #[test]
    fn test_linear_gradient_size() {
        let gen = ImageGenerator::new();
        let p = make_params(ImagePattern::LinearGradient);
        let pixels = gen.generate(&p);
        assert_eq!(pixels.len(), 16 * 16 * 3);

        // Left edge (x=0) must be close to start colour [255, 0, 0]
        // Right edge (x=15) must be close to end colour [0, 0, 255]
        let left = &pixels[0..3];
        assert_eq!(left[0], 255, "left R");
        assert_eq!(left[1], 0, "left G");
        assert_eq!(left[2], 0, "left B");

        let right_offset = (15) * 3; // last pixel of first row
        let right = &pixels[right_offset..right_offset + 3];
        assert_eq!(right[0], 0, "right R");
        assert_eq!(right[1], 0, "right G");
        assert_eq!(right[2], 255, "right B");
    }

    #[test]
    fn test_checkerboard_pattern() {
        let gen = ImageGenerator::new();
        let p = ImageParams {
            pattern: ImagePattern::Checkerboard,
            width: 8,
            height: 8,
            colors: vec![[0, 0, 0], [255, 255, 255]],
            block_size: 4,
            angle_degrees: 0.0,
            seed: 0,
            ..ImageParams::default()
        };
        let pixels = gen.generate(&p);
        assert_eq!(pixels.len(), 8 * 8 * 3);

        // Top-left 4x4 block: colour index 0 = [0,0,0]
        for y in 0..4_usize {
            for x in 0..4_usize {
                let idx = (y * 8 + x) * 3;
                assert_eq!(pixels[idx], 0, "top-left block R at ({x},{y})");
            }
        }

        // Top-right 4x4 block: colour index 1 = [255,255,255]
        for y in 0..4_usize {
            for x in 4..8_usize {
                let idx = (y * 8 + x) * 3;
                assert_eq!(pixels[idx], 255, "top-right block R at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_radial_gradient() {
        let gen = ImageGenerator::new();
        let p = ImageParams {
            pattern: ImagePattern::RadialGradient,
            width: 32,
            height: 32,
            colors: vec![[255, 255, 255], [0, 0, 0]],
            block_size: 4,
            angle_degrees: 0.0,
            seed: 0,
            ..ImageParams::default()
        };
        let pixels = gen.generate(&p);
        assert_eq!(pixels.len(), 32 * 32 * 3);

        // Centre pixel should be close to start colour (white)
        let cx = 16_usize;
        let cy = 16_usize;
        let centre_idx = (cy * 32 + cx) * 3;
        // Allow some tolerance due to float rounding
        assert!(
            pixels[centre_idx] > 230,
            "centre R should be near 255, got {}",
            pixels[centre_idx]
        );

        // Corner pixel (0,0) should be close to end colour (black / dark)
        let corner_r = pixels[0];
        assert!(corner_r < 80, "corner R should be near 0, got {}", corner_r);
    }
}
