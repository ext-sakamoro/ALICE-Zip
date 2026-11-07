//! What each image pattern draws, against what it must be by construction: Perlin noise is 0 on its lattice, a
//! gradient reaches its two colours, a checkerboard alternates, and a seed changes a noise image.

use alice_zip::media::{ImageGenerator, ImageParams, ImagePattern};

const A: [u8; 3] = [10, 200, 40];
const B: [u8; 3] = [250, 0, 90];

fn image(pattern: ImagePattern, w: usize, h: usize, block: usize, seed: u64) -> Vec<u8> {
    ImageGenerator::new().generate(&ImageParams {
        pattern,
        width: w,
        height: h,
        colors: vec![A, B],
        block_size: block,
        angle_degrees: 0.0,
        seed,
        ..ImageParams::default()
    })
}

fn pixel(px: &[u8], w: usize, x: usize, y: usize) -> [u8; 3] {
    let i = (y * w + x) * 3;
    [px[i], px[i + 1], px[i + 2]]
}

/// where a pixel lies from `A` (0) to `B` (1), by its first channel
fn position(px: &[u8]) -> Vec<f64> {
    let (a, b) = (f64::from(A[0]), f64::from(B[0]));
    px.chunks_exact(3)
        .map(|p| (f64::from(p[0]) - a) / (b - a))
        .collect()
}

/// the mean step between pixels `lag` apart along a row
fn step(px: &[u8], w: usize, lag: usize) -> f64 {
    let t = position(px);
    let (mut sum, mut n) = (0.0, 0u32);
    for row in t.chunks_exact(w) {
        for x in lag..w {
            sum += (row[x] - row[x - lag]).abs();
            n += 1;
        }
    }
    assert!(n > 0, "nothing compared");
    sum / f64::from(n)
}

#[test]
fn noise_is_the_midpoint_on_its_lattice() {
    // Perlin noise is exactly 0 at a lattice point, which is the middle of the two colours
    let (w, h, block) = (96, 64, 8);
    let px = image(ImagePattern::Noise, w, h, block, 7);
    let mid: Vec<u8> = (0..3)
        .map(|c| (f32::from(A[c]) * 0.5 + f32::from(B[c]) * 0.5) as u8)
        .collect();
    let mut n = 0;
    for y in (0..h).step_by(block) {
        for x in (0..w).step_by(block) {
            assert_eq!(pixel(&px, w, x, y).to_vec(), mid, "({x}, {y})");
            n += 1;
        }
    }
    assert_eq!(n, 96);
}

#[test]
fn the_seed_decides_the_image() {
    for pattern in [ImagePattern::Noise, ImagePattern::Fractal] {
        let a = image(pattern, 128, 128, 8, 1);
        assert_eq!(
            a,
            image(pattern, 128, 128, 8, 1),
            "{pattern:?}: the same seed gives the same bytes"
        );
        let b = image(pattern, 128, 128, 8, 2);
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(
            differ * 2 > a.len(),
            "{pattern:?}: another seed changed {differ} of {} bytes",
            a.len()
        );
    }
}

/// (mean, standard deviation, lowest, highest) of the position over 32 seeds
fn distribution(pattern: ImagePattern, block: usize) -> (f64, f64, f64, f64) {
    let mut t = Vec::new();
    for seed in 0..32 {
        t.extend(position(&image(pattern, 128, 128, block, seed)));
    }
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let var = t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let lo = t.iter().copied().fold(f64::INFINITY, alice_det_math::min64);
    let hi = t
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, alice_det_math::max64);
    (mean, var.sqrt(), lo, hi)
}

#[test]
fn noise_is_centred_and_spread_over_many_seeds() {
    for pattern in [ImagePattern::Noise, ImagePattern::Fractal] {
        let (mean, deviation, lo, hi) = distribution(pattern, 8);
        assert!((mean - 0.5).abs() < 0.02, "{pattern:?}: mean {mean}");
        assert!(deviation > 0.05, "{pattern:?}: deviation {deviation}");
        assert!(
            lo >= 0.0 && hi <= 1.0,
            "{pattern:?}: leaves the two colours: {lo}..{hi}"
        );
        assert!(lo < 0.4 && hi > 0.6, "{pattern:?}: reaches only {lo}..{hi}");
    }
}

/// Octaves of independent noise with amplitudes `a` and the sum normalised: the deviation is that of one octave
/// times `sqrt(sum a^2) / sum a`. Five octaves at persistence 0.5 give 0.5963.
#[test]
fn fractal_deviation_is_the_closed_form_of_its_octaves() {
    let amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625_f64];
    let squares: f64 = amplitudes.iter().map(|a| a * a).sum();
    let want = squares.sqrt() / amplitudes.iter().sum::<f64>();
    let (_, noise, _, _) = distribution(ImagePattern::Noise, 16);
    let (_, fractal, _, _) = distribution(ImagePattern::Fractal, 16);
    let got = fractal / noise;
    assert!((got - want).abs() < 0.03, "{got} against {want}");
}

#[test]
fn every_channel_stays_between_the_two_colours() {
    for pattern in [ImagePattern::Noise, ImagePattern::Fractal] {
        let px = image(pattern, 64, 64, 8, 3);
        for p in px.chunks_exact(3) {
            for c in 0..3 {
                let (lo, hi) = (A[c].min(B[c]), A[c].max(B[c]));
                assert!(p[c] >= lo && p[c] <= hi, "{pattern:?}: {p:?}");
            }
        }
    }
}

#[test]
fn fractal_has_finer_detail_than_noise_at_the_same_scale() {
    // a smooth field's step grows with the distance, so neighbours differ little against pixels a cell apart;
    // the finer octaves take that away
    let ratio = |pattern| {
        let (mut near, mut far) = (0.0, 0.0);
        for seed in 0..8 {
            let px = image(pattern, 128, 128, 16, seed);
            near += step(&px, 128, 1);
            far += step(&px, 128, 8);
        }
        assert!(far > 0.0, "{pattern:?}: flat");
        near / far
    };
    let (noise, fractal) = (ratio(ImagePattern::Noise), ratio(ImagePattern::Fractal));
    assert!(
        fractal > noise * 1.5,
        "fractal {fractal} against noise {noise}"
    );
}

#[test]
fn the_size_is_three_bytes_a_pixel_and_empty_when_there_are_no_pixels() {
    for pattern in [ImagePattern::Noise, ImagePattern::Fractal] {
        assert_eq!(image(pattern, 5, 3, 2, 1).len(), 45);
        assert!(image(pattern, 0, 3, 2, 1).is_empty());
        assert!(image(pattern, 5, 0, 2, 1).is_empty());
    }
}

#[test]
fn a_linear_gradient_reaches_both_colours_at_its_ends() {
    let (w, h) = (33, 3);
    let px = image(ImagePattern::LinearGradient, w, h, 8, 0);
    for y in 0..h {
        assert_eq!(pixel(&px, w, 0, y), A);
        assert_eq!(pixel(&px, w, w - 1, y), B);
    }
}

#[test]
fn a_radial_gradient_is_the_first_colour_at_the_centre_and_the_second_at_every_corner() {
    let (w, h) = (33, 33);
    let px = image(ImagePattern::RadialGradient, w, h, 8, 0);
    assert_eq!(pixel(&px, w, 16, 16), A);
    for (x, y) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
        assert_eq!(pixel(&px, w, x, y), B, "({x}, {y})");
    }
}

#[test]
fn a_radial_gradient_is_mirror_symmetric() {
    let (w, h) = (32, 20);
    let px = image(ImagePattern::RadialGradient, w, h, 8, 0);
    for y in 0..h {
        for x in 0..w {
            assert_eq!(
                pixel(&px, w, x, y),
                pixel(&px, w, w - 1 - x, y),
                "({x}, {y})"
            );
            assert_eq!(
                pixel(&px, w, x, y),
                pixel(&px, w, x, h - 1 - y),
                "({x}, {y})"
            );
        }
    }
}

#[test]
fn a_radial_gradient_takes_its_centre() {
    let params = ImageParams {
        pattern: ImagePattern::RadialGradient,
        width: 9,
        height: 9,
        colors: vec![A, B],
        center: [0.0, 0.0],
        ..ImageParams::default()
    };
    let px = ImageGenerator::new().generate(&params);
    assert_eq!(pixel(&px, 9, 0, 0), A);
    assert_eq!(pixel(&px, 9, 8, 8), B);
}

#[test]
fn channels_follow_the_two_colours() {
    let params = ImageParams {
        pattern: ImagePattern::Noise,
        width: 16,
        height: 16,
        ..ImageParams::default()
    };
    let gen = ImageGenerator::new();
    let one = gen.generate_channels(&params, &[0], &[255]);
    let four = gen.generate_channels(&params, &[0, 0, 0, 0], &[255, 255, 255, 255]);
    assert_eq!(one.len(), 256);
    assert_eq!(four.len(), 1024);
    for (i, v) in one.iter().enumerate() {
        assert_eq!(&four[i * 4..i * 4 + 4], &[*v; 4]);
    }
}

#[test]
fn a_level_is_rounded_to_the_nearest() {
    // the middle of 0 and 255 is 127.5, on a lattice point of the noise
    let params = ImageParams {
        pattern: ImagePattern::Noise,
        width: 9,
        height: 1,
        block_size: 8,
        ..ImageParams::default()
    };
    let px = ImageGenerator::new().generate_channels(&params, &[0], &[255]);
    assert_eq!(px[0], 128);
    assert_eq!(px[8], 128);
}

#[test]
fn a_noise_cell_of_one_is_not_a_flat_image() {
    // on a cell of 1 every pixel is a lattice point, where Perlin noise is 0: the smallest cell is 2
    for pattern in [ImagePattern::Noise, ImagePattern::Fractal] {
        let px = image(pattern, 16, 16, 1, 1);
        assert_eq!(px, image(pattern, 16, 16, 2, 1), "{pattern:?}");
        assert!(
            px.chunks_exact(3).any(|p| p != &px[..3]),
            "{pattern:?} is flat"
        );
    }
}

#[test]
fn a_checkerboard_alternates_by_block() {
    let (w, h, block) = (12, 12, 3);
    let px = image(ImagePattern::Checkerboard, w, h, block, 0);
    for y in 0..h {
        for x in 0..w {
            let want = if (x / block + y / block) % 2 == 0 {
                A
            } else {
                B
            };
            assert_eq!(pixel(&px, w, x, y), want, "({x}, {y})");
        }
    }
}

#[test]
fn solid_is_the_first_colour_everywhere() {
    let px = image(ImagePattern::Solid, 4, 4, 8, 0);
    assert_eq!(px.len(), 48);
    assert!(px.chunks_exact(3).all(|p| p == A));
}
