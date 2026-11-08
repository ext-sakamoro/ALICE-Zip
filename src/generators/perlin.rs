//! Noise generators — 2D gradient Perlin ([`PerlinNoise`]) and 1D value-noise
//! fBm ([`generate_fbm_1d`])
//!
//! Both are deterministic in `seed`; the 2D law is what the `.alice` container,
//! the FFI and the Python bindings persist (moved here from `libalice`
//! 2026-09-15), the 1D law is the `alice-db` `PerlinNoise` segment model
//! (formerly `generate_perlin_advanced(n, _dimension, …)` with an unused
//! `_dimension` argument)
//!
//! Analytic properties pinned by `tests/analytic_oracle.rs`: gradient noise
//! is exactly `0` on every integer lattice point, the fBm normalisation keeps
//! `|fbm| ≤ 1`, and the textures are seed-deterministic

use alloc::vec::Vec;
use core::f32::consts::FRAC_1_SQRT_2;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::ZipError;
use crate::math::floor_f32;

/// The most noise evaluations one call may make: points × octaves (a 2D texture counts each
/// pixel). Policy, tied to alice-db's whole-segment read budget: 1 GiB at 24 bytes a point gives
/// it 44 739 242 points a segment, and 64 octaves each is far above what any writer uses (5 at
/// most, `media::image`'s fractal pattern). A call over it is refused with
/// [`ZipError::InvalidParameter`] before anything is computed or allocated: the octave count
/// is a loop bound, and a stored model with octaves near `u32::MAX` would otherwise spin.
/// Measured at the budget (44 739 242 points × 64 octaves, release, arm64 10 cores, default
/// features): `generate_fbm_1d` 7.2 s, `generate_perlin_advanced` 14.0 s.
pub const MAX_FBM_STEPS: u64 = 44_739_242 * 64;

/// Whether `points` evaluations of `octaves` octaves fit [`MAX_FBM_STEPS`]
fn within_budget(points: usize, octaves: u32) -> bool {
    u64::try_from(points)
        .ok()
        .and_then(|p| p.checked_mul(u64::from(octaves)))
        .is_some_and(|steps| steps <= MAX_FBM_STEPS)
}

/// Perlin noise gradient vectors (precomputed)
const GRAD2: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [FRAC_1_SQRT_2, FRAC_1_SQRT_2],
    [-FRAC_1_SQRT_2, FRAC_1_SQRT_2],
    [FRAC_1_SQRT_2, -FRAC_1_SQRT_2],
    [-FRAC_1_SQRT_2, -FRAC_1_SQRT_2],
];

/// Seeded permutation table for 2D gradient (Perlin) noise
///
/// The table is a Fisher–Yates shuffle of `0..256` drawn from
/// `ChaCha8Rng::seed_from_u64(seed)`, so the same seed gives the same texture
/// on every platform
#[derive(Clone, Copy)]
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl core::fmt::Debug for PerlinNoise {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PerlinNoise")
            .field("perm", &&self.perm[..8])
            .finish_non_exhaustive()
    }
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with given seed
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut perm = [0u8; 512];
        for (i, p) in perm.iter_mut().enumerate().take(256) {
            *p = i as u8;
        }
        // Fisher-Yates shuffle
        // the range is u64: a usize range draws by pointer width, and wasm32 shuffled another table
        for i in (1..256_u64).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i as usize, j as usize);
        }
        // Duplicate for overflow handling
        for i in 0..256 {
            perm[256 + i] = perm[i];
        }
        Self { perm }
    }

    /// Quintic fade `6t⁵ − 15t⁴ + 10t³`
    #[inline]
    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    #[inline]
    fn grad(hash: usize, x: f32, y: f32) -> f32 {
        let g = &GRAD2[hash & 7];
        g[0] * x + g[1] * y
    }

    /// Sample gradient noise at a single point, range `[-1, 1]`, exactly `0`
    /// on integer lattice points
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        let xi = floor_f32(x) as i32;
        let yi = floor_f32(y) as i32;
        let xf = x - xi as f32;
        let yf = y - yi as f32;
        let xi = (xi & 255) as usize;
        let yi = (yi & 255) as usize;

        let aa = self.perm[self.perm[xi] as usize + yi] as usize;
        let ab = self.perm[self.perm[xi] as usize + yi + 1] as usize;
        let ba = self.perm[self.perm[xi + 1] as usize + yi] as usize;
        let bb = self.perm[self.perm[xi + 1] as usize + yi + 1] as usize;

        let g00 = Self::grad(aa, xf, yf);
        let g10 = Self::grad(ba, xf - 1.0, yf);
        let g01 = Self::grad(ab, xf, yf - 1.0);
        let g11 = Self::grad(bb, xf - 1.0, yf - 1.0);

        let u = Self::fade(xf);
        let v = Self::fade(yf);
        Self::lerp(Self::lerp(g00, g10, u), Self::lerp(g01, g11, u), v)
    }

    /// Fractal Brownian motion: `Σ_o persistence^o · noise(lacunarity^o · p)`
    /// normalised by `Σ_o persistence^o`, so `|fbm| ≤ 1` for `persistence ≥ 0`
    /// `octaves == 0` returns `0.0`
    ///
    /// `0.0` where [`Self::try_fbm`] refuses: more than [`MAX_FBM_STEPS`] octaves, or an octave
    /// whose frequency left the finite range.
    #[inline]
    #[must_use]
    #[deprecated(note = "returns 0.0 for what it cannot sum; use `try_fbm`, which says so")]
    pub fn fbm(&self, x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
        self.try_fbm(x, y, octaves, persistence, lacunarity)
            .unwrap_or(0.0)
    }

    /// [`Self::fbm`], refusing what it cannot sum: the sum stops at the first octave whose
    /// amplitude is `0` (every later one adds nothing), and an octave whose frequency or
    /// position is not finite is an error, where the noise would otherwise be `NaN`
    ///
    /// # Errors
    ///
    /// [`ZipError::InvalidParameter`] for more than [`MAX_FBM_STEPS`] octaves, or a frequency
    /// that left the finite range before the amplitude reached `0`
    pub fn try_fbm(
        &self,
        x: f32,
        y: f32,
        octaves: u32,
        persistence: f32,
        lacunarity: f32,
    ) -> Result<f32, ZipError> {
        if !within_budget(1, octaves) {
            return Err(ZipError::InvalidParameter);
        }
        let mut total = 0.0_f32;
        let mut amplitude = 1.0_f32;
        let mut frequency = 1.0_f32;
        let mut max_value = 0.0_f32;
        for _ in 0..octaves {
            if amplitude == 0.0 {
                break;
            }
            let (px, py) = (x * frequency, y * frequency);
            if !(px.is_finite() && py.is_finite()) {
                return Err(ZipError::InvalidParameter);
            }
            total += self.noise2d(px, py) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        Ok(if max_value > 0.0 {
            total / max_value
        } else {
            0.0
        })
    }
}

/// Generate a `width × height` 2D fBm Perlin texture (row-major, values in
/// `[0, 1]`) with `persistence = 0.5`, `lacunarity = 2.0`
///
/// `scale` is the lattice period in pixels (larger = smoother)
///
/// # Errors
///
/// [`ZipError::InvalidParameter`] when `scale` is not finite and `> 0`,
/// `octaves == 0`, or `width * height` overflows `usize`
pub fn generate_perlin_2d(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
) -> Result<Vec<f32>, ZipError> {
    generate_perlin_advanced(width, height, seed, scale, octaves, 0.5, 2.0)
}

/// Generate a `width × height` 2D fBm Perlin texture (row-major, values in
/// `[0, 1]`) with explicit `persistence` / `lacunarity`
///
/// # Errors
///
/// [`ZipError::InvalidParameter`] when `scale` is not finite and `> 0`,
/// `octaves == 0`, or `width * height` overflows `usize`
#[allow(clippy::cast_precision_loss)]
pub fn generate_perlin_advanced(
    width: usize,
    height: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> Result<Vec<f32>, ZipError> {
    if !(scale.is_finite() && scale > 0.0) || octaves == 0 {
        return Err(ZipError::InvalidParameter);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZipError::InvalidParameter)?;
    if !within_budget(pixels, octaves) {
        return Err(ZipError::InvalidParameter);
    }
    let noise = PerlinNoise::new(seed);
    let rcp_scale = 1.0 / scale;
    let row = move |y: usize| -> Result<Vec<f32>, ZipError> {
        let ny = y as f32 * rcp_scale;
        (0..width)
            .map(|x| {
                let nx = x as f32 * rcp_scale;
                let v = (noise.try_fbm(nx, ny, octaves, persistence, lacunarity)? + 1.0) * 0.5;
                Ok(v.clamp(0.0, 1.0))
            })
            .collect()
    };
    #[cfg(feature = "parallel")]
    let rows: Vec<Vec<f32>> = (0..height)
        .into_par_iter()
        .map(row)
        .collect::<Result<_, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<Vec<f32>> = (0..height).map(row).collect::<Result<_, _>>()?;
    Ok(rows.into_iter().flatten().collect())
}

/// 1D fractional-Brownian-motion value noise (hashed integer lattice, cubic
/// Hermite interpolation) — deterministic given `seed`, output in `[-1, 1]`
///
/// Sample `i` is evaluated at `x = i / n` (the whole series spans one unit),
/// `scale` is the base spatial frequency (larger = finer detail)
///
/// This is the `alice-db` `PerlinNoise` segment model law (it is value noise,
/// not gradient Perlin — the name is historical); use
/// [`generate_perlin_advanced`] for textures
///
/// # Errors
///
/// [`ZipError::InvalidParameter`] when `scale` is not finite and `> 0` or
/// `octaves == 0`
#[allow(clippy::cast_precision_loss)]
pub fn generate_fbm_1d(
    n: usize,
    seed: u64,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> Result<Vec<f32>, ZipError> {
    if !(scale.is_finite() && scale > 0.0) || octaves == 0 || !within_budget(n, octaves) {
        return Err(ZipError::InvalidParameter);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            let mut sum = 0.0_f32;
            let mut amplitude = 1.0_f32;
            let mut frequency = scale;
            let mut max_amp = 0.0_f32;
            let x = i as f32 * inv_n;
            for _ in 0..octaves {
                // a zero amplitude adds nothing from here on; a position that is not finite
                // would make the noise NaN
                if amplitude == 0.0 {
                    break;
                }
                let at = x * frequency;
                if !at.is_finite() {
                    return Err(ZipError::InvalidParameter);
                }
                sum += amplitude * value_noise_1d(at, seed);
                max_amp += amplitude;
                amplitude *= persistence;
                frequency *= lacunarity;
            }
            Ok(if max_amp > 0.0 { sum / max_amp } else { sum })
        })
        .collect()
}

// ---------- internals ----------

/// 1D value noise with cubic Hermite (smoothstep) interpolation between
/// hashed integer lattice points
#[allow(clippy::cast_possible_truncation)]
fn value_noise_1d(x: f32, seed: u64) -> f32 {
    let x0 = floor_f32(x);
    let x1 = x0 + 1.0;
    let t = x - x0;
    let s = t * t * (3.0 - 2.0 * t);
    let g0 = hash_to_unit(x0 as i32, seed);
    let g1 = hash_to_unit(x1 as i32, seed);
    g0 * (1.0 - s) + g1 * s
}

#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn hash_to_unit(x: i32, seed: u64) -> f32 {
    // Small mixed-word hash (Wang-style) mapped to `[-1.0, 1.0]`
    let mut h = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= seed.wrapping_mul(0x85eb_ca6b_c2b2_ae35);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc2b2_ae35_9e37_79b9);
    h ^= h >> 33;
    let f = (h & 0x00ff_ffff) as f32 / 8_388_608.0; // 2^23
    f - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_deterministic_and_seed_sensitive() {
        let a = generate_perlin_2d(64, 64, 42, 10.0, 4).unwrap();
        let b = generate_perlin_2d(64, 64, 42, 10.0, 4).unwrap();
        let c = generate_perlin_2d(64, 64, 43, 10.0, 4).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64 * 64);
    }

    #[test]
    fn perlin_range() {
        let noise = generate_perlin_2d(128, 128, 123, 20.0, 4).unwrap();
        for &v in &noise {
            assert!((0.0..=1.0).contains(&v), "Value {v} out of range [0, 1]");
        }
    }

    #[test]
    fn perlin_rejects_invalid_parameters() {
        assert_eq!(
            generate_perlin_2d(4, 4, 1, 0.0, 4),
            Err(ZipError::InvalidParameter)
        );
        assert_eq!(
            generate_perlin_2d(4, 4, 1, f32::NAN, 4),
            Err(ZipError::InvalidParameter)
        );
        assert_eq!(
            generate_perlin_2d(4, 4, 1, 10.0, 0),
            Err(ZipError::InvalidParameter)
        );
        assert_eq!(
            generate_perlin_2d(usize::MAX, 2, 1, 10.0, 1),
            Err(ZipError::InvalidParameter)
        );
        assert!(generate_perlin_2d(0, 5, 1, 10.0, 1).unwrap().is_empty());
    }

    #[test]
    fn noise_is_zero_on_lattice() {
        let p = PerlinNoise::new(7);
        for x in -3..4 {
            for y in -3..4 {
                #[allow(clippy::cast_precision_loss)]
                let v = p.noise2d(x as f32, y as f32);
                assert_eq!(v, 0.0, "lattice ({x}, {y})");
            }
        }
        assert_eq!(p.try_fbm(1.0, 2.0, 0, 0.5, 2.0), Ok(0.0));
    }

    #[test]
    fn fbm_1d_contract() {
        let a = generate_fbm_1d(256, 9, 4.0, 3, 0.5, 2.0).unwrap();
        let b = generate_fbm_1d(256, 9, 4.0, 3, 0.5, 2.0).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 256);
        for &v in &a {
            assert!((-1.0..=1.0).contains(&v));
        }
        assert!(generate_fbm_1d(0, 1, 1.0, 1, 0.5, 2.0).unwrap().is_empty());
        assert_eq!(
            generate_fbm_1d(8, 1, 0.0, 1, 0.5, 2.0),
            Err(ZipError::InvalidParameter)
        );
        assert_eq!(
            generate_fbm_1d(8, 1, 1.0, 0, 0.5, 2.0),
            Err(ZipError::InvalidParameter)
        );
    }
}
