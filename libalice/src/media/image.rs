//! Image Generator
//!
//! Rust port of the Python ImageGenerator in media_generators.py.
//! Generates procedural images from parametric descriptions.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use std::f32::consts::PI;

// ============================================================================
// Enums
// ============================================================================

/// Detected / requested image patterns (mirrors Python ImagePattern enum)
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
    /// Cell size for Checkerboard (pixels per square)
    pub block_size: usize,
    /// Gradient direction in degrees (0 = left-to-right, 90 = top-to-bottom)
    pub angle_degrees: f32,
    /// RNG seed (reserved for Noise / Fractal)
    pub seed: u64,
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
    pub fn new() -> Self {
        Self
    }

    /// Generate RGB pixel data (width * height * 3 bytes, row-major)
    pub fn generate(&self, params: &ImageParams) -> Vec<u8> {
        let w = params.width;
        let h = params.height;

        // Colour helpers with safe fallbacks
        let color0 = params.colors.first().copied().unwrap_or([0, 0, 0]);
        let color1 = params.colors.get(1).copied().unwrap_or([255, 255, 255]);

        match params.pattern {
            ImagePattern::Solid => Self::generate_solid(w, h, color0),
            ImagePattern::LinearGradient => {
                Self::generate_linear_gradient(w, h, color0, color1, params.angle_degrees)
            }
            ImagePattern::RadialGradient => {
                Self::generate_radial_gradient(w, h, color0, color1)
            }
            ImagePattern::Checkerboard => {
                let block = params.block_size.max(1);
                Self::generate_checkerboard(w, h, color0, color1, block)
            }
            // Noise / Fractal fall back to a mid-grey fill (matches Python's `else` branch)
            ImagePattern::Noise | ImagePattern::Fractal => {
                Self::generate_solid(w, h, [128, 128, 128])
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    /// Fill the entire image with a single colour
    #[inline]
    fn generate_solid(w: usize, h: usize, color: [u8; 3]) -> Vec<u8> {
        let n = w * h;
        let mut out = Vec::with_capacity(n * 3);
        for _ in 0..n {
            out.push(color[0]);
            out.push(color[1]);
            out.push(color[2]);
        }
        out
    }

    /// Linear gradient along a given angle (vectorised, branchless interpolation)
    ///
    /// `angle_degrees`:
    ///   - 0   → left-to-right
    ///   - 90  → top-to-bottom
    ///   - other → arbitrary direction
    #[inline]
    fn generate_linear_gradient(
        w: usize,
        h: usize,
        start: [u8; 3],
        end: [u8; 3],
        angle_degrees: f32,
    ) -> Vec<u8> {
        let rcp_w = if w > 1 { 1.0 / (w - 1) as f32 } else { 1.0 };
        let rcp_h = if h > 1 { 1.0 / (h - 1) as f32 } else { 1.0 };

        let angle_rad = angle_degrees * (PI / 180.0);
        let nx = angle_rad.cos();
        let ny = angle_rad.sin();

        // Pre-compute divisor reciprocal to avoid per-pixel division
        let divisor = (w as f32) * nx.abs() + (h as f32) * ny.abs();
        let rcp_divisor = if divisor > 1e-10 { 1.0 / divisor } else { 1.0 };

        // Convert start/end to f32 once
        let s = [start[0] as f32, start[1] as f32, start[2] as f32];
        let e = [end[0] as f32, end[1] as f32, end[2] as f32];

        let mut out = Vec::with_capacity(w * h * 3);

        for y in 0..h {
            for x in 0..w {
                // t = interpolation factor in [0, 1]
                let t = if angle_degrees == 0.0 {
                    x as f32 * rcp_w
                } else if angle_degrees == 90.0 {
                    y as f32 * rcp_h
                } else {
                    let raw = (x as f32 * nx + y as f32 * ny) * rcp_divisor;
                    // Branchless clamp: clamp(raw, 0, 1)
                    raw.clamp(0.0, 1.0)
                };

                // Branchless lerp: start * (1 - t) + end * t
                let one_minus_t = 1.0 - t;
                for ch in 0..3 {
                    let v = s[ch] * one_minus_t + e[ch] * t;
                    out.push(v.clamp(0.0, 255.0) as u8);
                }
            }
        }
        out
    }

    /// Radial gradient from centre (start) to edge (end)
    #[inline]
    fn generate_radial_gradient(
        w: usize,
        h: usize,
        start: [u8; 3],
        end: [u8; 3],
    ) -> Vec<u8> {
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;

        // Maximum possible distance from centre (to a corner)
        let dx_max = cx.max(w as f32 - cx);
        let dy_max = cy.max(h as f32 - cy);
        let max_dist = (dx_max * dx_max + dy_max * dy_max).sqrt();
        let rcp_max = if max_dist > 1e-10 { 1.0 / max_dist } else { 1.0 };

        let s = [start[0] as f32, start[1] as f32, start[2] as f32];
        let e = [end[0] as f32, end[1] as f32, end[2] as f32];

        let mut out = Vec::with_capacity(w * h * 3);

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let t = ((dx * dx + dy * dy).sqrt() * rcp_max).clamp(0.0, 1.0);
                let one_minus_t = 1.0 - t;
                for ch in 0..3 {
                    let v = s[ch] * one_minus_t + e[ch] * t;
                    out.push(v.clamp(0.0, 255.0) as u8);
                }
            }
        }
        out
    }

    /// Checkerboard pattern — fully branchless colour selection
    ///
    /// `((x / block) + (y / block)) & 1` selects colour index (0 or 1).
    #[inline]
    fn generate_checkerboard(
        w: usize,
        h: usize,
        c1: [u8; 3],
        c2: [u8; 3],
        block_size: usize,
    ) -> Vec<u8> {
        // Colour table so we can index without branching
        let palette: [[u8; 3]; 2] = [c1, c2];

        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                // Branchless: parity = ((x/block) + (y/block)) & 1
                let parity = ((x / block_size) + (y / block_size)) & 1;
                let col = palette[parity];
                out.push(col[0]);
                out.push(col[1]);
                out.push(col[2]);
            }
        }
        out
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
        assert!(
            corner_r < 80,
            "corner R should be near 0, got {}",
            corner_r
        );
    }
}
