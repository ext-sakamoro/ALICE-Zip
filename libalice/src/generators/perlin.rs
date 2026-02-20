//! High-performance Perlin Noise Generator
//!
//! Vectorized implementation using Rayon for parallel processing.
//! This replaces the Python O(n²) loop with efficient SIMD-friendly operations.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Perlin noise gradient vectors (precomputed)
const GRAD2: [[f32; 2]; 8] = [
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [0.7071, 0.7071], [-0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, -0.7071],
];

/// Permutation table for noise generation
#[derive(Clone, Copy)]
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with given seed
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut perm = [0u8; 512];

        // Initialize permutation table
        for i in 0..256 {
            perm[i] = i as u8;
        }

        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }

        // Duplicate for overflow handling
        for i in 0..256 {
            perm[256 + i] = perm[i];
        }

        Self { perm }
    }

    /// Smooth interpolation (quintic)
    #[inline(always)]
    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Linear interpolation
    #[inline(always)]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    /// Gradient function
    #[inline(always)]
    fn grad(&self, hash: usize, x: f32, y: f32) -> f32 {
        let g = &GRAD2[hash & 7];
        g[0] * x + g[1] * y
    }

    /// Sample noise at a single point
    #[inline(always)]
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        // Grid cell coordinates
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        // Relative coordinates within cell
        let xf = x - xi as f32;
        let yf = y - yi as f32;

        // Wrap coordinates
        let xi = (xi & 255) as usize;
        let yi = (yi & 255) as usize;

        // Hash coordinates
        let aa = self.perm[self.perm[xi] as usize + yi] as usize;
        let ab = self.perm[self.perm[xi] as usize + yi + 1] as usize;
        let ba = self.perm[self.perm[xi + 1] as usize + yi] as usize;
        let bb = self.perm[self.perm[xi + 1] as usize + yi + 1] as usize;

        // Gradient contributions
        let g00 = self.grad(aa, xf, yf);
        let g10 = self.grad(ba, xf - 1.0, yf);
        let g01 = self.grad(ab, xf, yf - 1.0);
        let g11 = self.grad(bb, xf - 1.0, yf - 1.0);

        // Interpolation weights
        let u = Self::fade(xf);
        let v = Self::fade(yf);

        // Bilinear interpolation
        Self::lerp(
            Self::lerp(g00, g10, u),
            Self::lerp(g01, g11, u),
            v
        )
    }

    /// Generate fractal Brownian motion (fBm) noise
    #[inline(always)]
    pub fn fbm(&self, x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
        let mut total = 0.0f32;
        let mut amplitude = 1.0f32;
        let mut frequency = 1.0f32;
        let mut max_value = 0.0f32;

        for _ in 0..octaves {
            total += self.noise2d(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        // Reciprocal multiply: avoid division on hot path
        let rcp_max = 1.0 / max_value;
        total * rcp_max
    }
}

/// Generate a 2D noise texture (vectorized, parallel)
///
/// # Arguments
/// * `width` - Image width
/// * `height` - Image height
/// * `seed` - Random seed
/// * `scale` - Noise scale (larger = more zoomed out)
/// * `octaves` - Number of fBm octaves
///
/// # Returns
/// Flattened f32 array of size width * height, values in [0, 1]
pub fn generate_perlin_2d(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
) -> Vec<f32> {
    let noise = PerlinNoise::new(seed);
    let persistence = 0.5f32;
    let lacunarity = 2.0f32;
    // Pre-compute reciprocal to avoid per-pixel division inside the loop
    let rcp_scale = 1.0 / scale;

    // Parallel generation using Rayon
    (0..height)
        .into_par_iter()
        .flat_map(|y| {
            (0..width).map(move |x| {
                let nx = x as f32 * rcp_scale;
                let ny = y as f32 * rcp_scale;
                // Normalize from [-1, 1] to [0, 1] and clamp to handle
                // floating-point errors that may produce values slightly outside range
                let normalized = (noise.fbm(nx, ny, octaves, persistence, lacunarity) + 1.0) * 0.5;
                normalized.clamp(0.0, 1.0)
            }).collect::<Vec<_>>()
        })
        .collect()
}

/// Generate a 2D noise texture with specified parameters (advanced)
pub fn generate_perlin_advanced(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> Vec<f32> {
    let noise = PerlinNoise::new(seed);
    // Pre-compute reciprocal to avoid per-pixel division inside the loop
    let rcp_scale = 1.0 / scale;

    (0..height)
        .into_par_iter()
        .flat_map(|y| {
            (0..width).map(move |x| {
                let nx = x as f32 * rcp_scale;
                let ny = y as f32 * rcp_scale;
                // Normalize from [-1, 1] to [0, 1] and clamp to handle
                // floating-point errors that may produce values slightly outside range
                let normalized = (noise.fbm(nx, ny, octaves, persistence, lacunarity) + 1.0) * 0.5;
                normalized.clamp(0.0, 1.0)
            }).collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_deterministic() {
        let noise1 = generate_perlin_2d(64, 64, 42, 10.0, 4);
        let noise2 = generate_perlin_2d(64, 64, 42, 10.0, 4);
        assert_eq!(noise1, noise2);
    }

    #[test]
    fn test_perlin_range() {
        let noise = generate_perlin_2d(128, 128, 123, 20.0, 4);
        for &v in &noise {
            assert!(v >= 0.0 && v <= 1.0, "Value {} out of range [0, 1]", v);
        }
    }

    #[test]
    fn test_different_seeds() {
        let noise1 = generate_perlin_2d(32, 32, 1, 10.0, 4);
        let noise2 = generate_perlin_2d(32, 32, 2, 10.0, 4);
        assert_ne!(noise1, noise2);
    }
}
