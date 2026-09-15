//! Analytic-oracle tests (CLAUDE.md § 解析解突合テスト規律)
//!
//! Every numerical law in the crate is checked against a closed-form answer or
//! an independent reference, through the **default** configuration, and with
//! its precision / size parameters swept — golden values are never pinned.
//!
//! What this file caught while being written (2026-09-15, all fixed in 0.4.0):
//! `shannon_entropy` of a uniform byte distribution was 6.74 instead of 8.0
//! (20-term series, the unit test pinned the wrong value), the Nyquist bin was
//! reconstructed at twice its amplitude, `lz77_decode` / `Dictionary::add`
//! panicked on malformed input, and `analyze_signal` normalised its energy
//! threshold by the truncated spectrum instead of the total.

#![cfg(feature = "std")]

use alice_zip::generators::{
    analyze_signal, fit_polynomial, fit_polynomial_unit, generate_fbm_1d,
    generate_from_coefficients, generate_multi_sine, generate_perlin_2d, generate_perlin_advanced,
    generate_polynomial, generate_polynomial_unit, generate_sine_wave, PerlinNoise,
};
use alice_zip::prelude::*;
use std::f32::consts::PI;

/// Small deterministic corpus shared by the round-trip laws
fn corpus() -> Vec<Vec<u8>> {
    let mut lcg = 0x2545_F491_4F6C_DD1Du64;
    let mut pseudo = Vec::with_capacity(4096);
    for _ in 0..4096 {
        lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        pseudo.push((lcg >> 33) as u8);
    }
    vec![
        Vec::new(),
        vec![0],
        vec![7; 3000],
        b"abcabcabcabcabcabcabcabcabc".to_vec(),
        (0..=255u8).cycle().take(2048).collect(),
        b"the quick brown fox jumps over the lazy dog ".repeat(40),
        pseudo,
        (0..3000u32).map(|i| ((i * i) % 7) as u8).collect(),
    ]
}

// ---------------------------------------------------------------- entropy

#[test]
fn entropy_equiprobable_k_symbols_is_log2_k() {
    for k in [1u32, 2, 3, 5, 8, 16, 37, 100, 256] {
        let data: Vec<u8> = (0..k * 11).map(|i| (i % k) as u8).collect();
        let expected = f64::from(k).log2();
        let e = shannon_entropy(&data);
        assert!((e - expected).abs() < 1e-9, "k={k}: {e} vs {expected}");
    }
}

#[test]
fn entropy_binary_source_matches_closed_form() {
    // H(p) = -p log2 p - (1-p) log2 (1-p), sampled at exact ratios
    for (ones, total) in [(1usize, 4usize), (1, 10), (3, 8), (7, 16), (255, 256)] {
        let mut data = vec![0u8; total - ones];
        data.extend(std::iter::repeat_n(1u8, ones));
        let p = ones as f64 / total as f64;
        let expected = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2());
        let e = shannon_entropy(&data);
        assert!((e - expected).abs() < 1e-9, "p={p}: {e} vs {expected}");
    }
}

#[test]
fn entropy_bounds_and_size_law() {
    for data in corpus() {
        let e = shannon_entropy(&data);
        assert!((0.0..=8.0).contains(&e), "{e}");
        let bits = e * data.len() as f64;
        assert_eq!(theoretical_min_size(&data), (bits / 8.0).ceil() as usize);
        assert!(theoretical_min_size(&data) <= data.len());
    }
}

// ---------------------------------------------------------------- lz77

#[test]
fn lz77_roundtrip_is_invariant_under_window_and_lookahead() {
    // window / lookahead only change the token stream, never the decoded bytes
    let params = [1usize, 2, 3, 4, 15, 16, 255, 256, 4096, 65_535, 1 << 20];
    for data in corpus() {
        for &w in &params {
            for &l in &[0usize, 1, 2, 17, 258, 65_535, 1 << 20] {
                let tokens = lz77_encode(&data, w, l);
                let decoded = lz77_decode(&tokens).unwrap();
                assert_eq!(decoded, data, "w={w} l={l}");
                assert!(tokens.len() <= data.len().max(1));
            }
        }
    }
}

#[test]
fn lz77_clamps_to_u16_and_lookahead_zero_is_all_literals() {
    let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
    // Anything above MAX_WINDOW behaves exactly like MAX_WINDOW (no truncation)
    let a = lz77_encode(&data, MAX_WINDOW, MAX_WINDOW);
    let b = lz77_encode(&data, 1 << 20, 1 << 20);
    assert_eq!(a, b);
    assert!(a.iter().all(|t| t.length as usize <= MAX_WINDOW));

    let tokens = lz77_encode(b"aaaaaaaa", 8, 0);
    assert_eq!(tokens.len(), 8);
    assert!(tokens.iter().all(|t| t.length == 0 && t.offset == 0));
}

#[test]
fn lz77_decode_rejects_malformed_tokens() {
    let bad = |offset, length| LzToken {
        offset,
        length,
        literal: b'x',
    };
    assert_eq!(lz77_decode(&[bad(1, 1)]), Err(ZipError::InvalidData));
    assert_eq!(
        lz77_decode(&[bad(0, 0), bad(5, 1)]),
        Err(ZipError::InvalidData)
    );
    assert_eq!(
        lz77_decode(&[bad(0, 0), bad(0, 3)]),
        Err(ZipError::InvalidData)
    );
    // Overlapping copy (run-length) is valid
    let tokens = [bad(0, 0), bad(1, 5)];
    assert_eq!(lz77_decode(&tokens).unwrap(), b"xxxxxxx");
}

// ---------------------------------------------------------------- dictionary

#[test]
fn dictionary_fifo_eviction_law() {
    for cap in [1usize, 2, 3, 7, 64] {
        for k in [0usize, 1, cap, cap + 1, 3 * cap + 2] {
            let mut dict = Dictionary::new(cap);
            let phrases: Vec<Vec<u8>> = (0..k).map(|i| format!("p{i}").into_bytes()).collect();
            for p in &phrases {
                dict.add(p).unwrap();
            }
            let live = k.min(cap);
            assert_eq!(dict.len(), live, "cap={cap} k={k}");
            assert_eq!(dict.is_empty(), live == 0);
            for i in 0..live {
                let expected = &phrases[k - live + i];
                assert_eq!(dict.lookup(i as u32), Some(expected.as_slice()));
                // Re-adding a live phrase returns its index without insertion
                assert_eq!(dict.add(expected).unwrap(), i as u32);
                assert_eq!(dict.len(), live);
            }
            assert_eq!(dict.lookup(live as u32), None);
        }
    }
}

#[test]
fn dictionary_capacity_zero_fails_instead_of_panicking() {
    let mut dict = Dictionary::new(0);
    assert_eq!(dict.add(b"x"), Err(ZipError::DictionaryFull));
    assert!(dict.is_empty());
}

// ---------------------------------------------------------------- bpe

#[test]
fn bpe_replace_length_law() {
    // Output length = input length − number of non-overlapping pair matches,
    // and the replacement round-trips when the replacement byte is unused
    for data in corpus() {
        let Some(pair) = find_most_frequent_pair(&data) else {
            continue;
        };
        let replacement = (0..=255u8).find(|b| !data.contains(b)).unwrap_or(0xFF);
        let out = bpe_replace(&data, pair, replacement);
        let mut matches = 0;
        let mut i = 0;
        while i + 1 < data.len() {
            if data[i] == pair.0 && data[i + 1] == pair.1 {
                matches += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
        assert_eq!(out.len(), data.len() - matches);
        // The pair count is over adjacent positions (overlaps included), so the
        // most frequent pair occurs ≥ 2 times overlapping but may be replaced once
        let overlapping = data
            .windows(2)
            .filter(|w| w[0] == pair.0 && w[1] == pair.1)
            .count();
        assert!(
            overlapping >= 2 && matches >= 1,
            "{pair:?}: {overlapping} / {matches}"
        );
        if !data.contains(&replacement) {
            let restored: Vec<u8> = out
                .iter()
                .flat_map(|&b| {
                    if b == replacement {
                        vec![pair.0, pair.1]
                    } else {
                        vec![b]
                    }
                })
                .collect();
            assert_eq!(restored, data);
        }
    }
    assert_eq!(find_most_frequent_pair(b"abcd"), None);
    assert_eq!(find_most_frequent_pair(b"abab"), Some((b'a', b'b')));
}

// ---------------------------------------------------------------- polynomial

#[test]
fn fit_polynomial_recovers_exactly_representable_polynomials() {
    // Integer coefficients, small n: every sample is an integer < 2^24, so the
    // f32 input carries no quantisation noise and recovery is exact (f64)
    for (coeffs, n) in [
        (vec![3.0, 2.0], 20usize),
        (vec![1.0, -2.0, 3.0], 24),
        (vec![1.0, 2.0, 3.0, 4.0], 20),
        (vec![-7.0, 0.0, 0.5, 0.0, 0.25], 16),
    ] {
        let d = coeffs.len() - 1;
        let values = generate_polynomial(n, &coeffs);
        let (fit, degree, err) = fit_polynomial(&values, 6, 1e-12).expect("fit");
        assert_eq!(degree, d, "{fit:?}");
        assert!(err < 1e-12, "{err}");
        for (a, b) in fit.iter().zip(&coeffs) {
            assert!((a - b).abs() < 1e-9, "{fit:?} vs {coeffs:?}");
        }
    }
}

#[test]
fn fit_polynomial_recovers_f32_sampled_polynomials() {
    // General case: the samples are f32 (≈ 6e-8 relative quantisation), which
    // the Vandermonde conditioning amplifies — the law is still recovered to
    // 1e-3 in the unit-scaled coefficients a_j = c_j / h^j, and the degree
    // search never over- or under-shoots
    for d in 1..=5usize {
        for n in [d + 1, 2 * d + 1, 20, 64] {
            let h = 1.0 / (n - 1) as f64;
            let unit_coeffs: Vec<f64> = (0..=d).map(|j| (j + 1) as f64 * 0.25).collect();
            let coeffs: Vec<f64> = unit_coeffs
                .iter()
                .enumerate()
                .map(|(j, a)| a * h.powi(j as i32))
                .collect();
            let values = generate_polynomial(n, &coeffs);
            let (fit, degree, err) = fit_polynomial(&values, 6, 1e-9).expect("fit");
            assert!(err < 1e-9);
            if n == d + 1 {
                assert!(degree <= d); // interpolation through every point
            } else {
                assert_eq!(degree, d, "n={n} d={d}: {fit:?}");
                for (j, (a, b)) in fit.iter().zip(&unit_coeffs).enumerate() {
                    let a_unit = a / h.powi(j as i32);
                    assert!(
                        (a_unit - b).abs() < 1e-3,
                        "n={n} d={d} j={j}: {a_unit} vs {b}"
                    );
                }
            }
            let regen = generate_polynomial(n, &fit);
            for (a, b) in regen.iter().zip(&values) {
                assert!((a - b).abs() < 1e-5 * b.abs().max(1.0));
            }
        }
    }
}

#[test]
fn fit_polynomial_unit_recovers_exact_polynomials() {
    // Unit-interval descending convention (.alice container); f32 samples
    for d in 1..=6usize {
        let coeffs: Vec<f64> = (0..=d).map(|j| (d - j) as f64 - 1.5).collect();
        for n in [16usize, 50, 100] {
            let values = generate_polynomial_unit(n, &coeffs);
            let (fit, degree, err) = fit_polynomial_unit(&values, 8, 1e-9).expect("fit");
            assert_eq!(degree, d, "n={n} d={d}: {fit:?}");
            assert!(err < 1e-9);
            for (a, b) in fit.iter().zip(&coeffs) {
                assert!((a - b).abs() < 1e-3, "{fit:?} vs {coeffs:?}");
            }
            let regen = generate_polynomial_unit(n, &fit);
            for (a, b) in regen.iter().zip(&values) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
    // Exactly representable samples → exact coefficients
    let (fit, degree, _) =
        fit_polynomial_unit(&generate_polynomial_unit(17, &[16.0, -8.0, 4.0]), 5, 1e-12).unwrap();
    assert_eq!(degree, 2);
    for (a, b) in fit.iter().zip(&[16.0, -8.0, 4.0]) {
        assert!((a - b).abs() < 1e-9, "{fit:?}");
    }
    // Degree search is minimal: a line is not reported as a quadratic
    let line: Vec<f32> = (0..30).map(|i| 2.0 - 0.5 * i as f32 / 29.0).collect();
    assert_eq!(fit_polynomial_unit(&line, 5, 1e-9).unwrap().1, 1);
    assert_eq!(fit_polynomial(&line, 5, 1e-9).unwrap().1, 1);
}

#[test]
fn polynomial_generators_agree_on_conventions() {
    // Same polynomial expressed in both conventions gives the same samples
    // when x is rescaled: p(x) with x = i/(n-1) ⇔ q(i) = p(i/(n-1))
    let n = 21usize;
    let desc = [2.0, -3.0, 0.5]; // 2x² − 3x + 0.5 on [0, 1]
    let unit = generate_polynomial_unit(n, &desc);
    let h = 1.0 / (n - 1) as f64;
    let asc = [0.5, -3.0 * h, 2.0 * h * h];
    let int = generate_polynomial(n, &asc);
    for (a, b) in unit.iter().zip(&int) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

// ---------------------------------------------------------------- fourier

#[test]
fn single_bin_reconstruction_is_exact_including_nyquist() {
    // A pure cosine at bin k has |X_k| = A n / 2 (k < n/2) or A n (k = n/2);
    // reconstruction must return the original samples for every k
    for n in [8usize, 16, 31, 64] {
        for k in 1..=n / 2 {
            let amp = 1.5f32;
            let phase = 0.4f32;
            let samples: Vec<f32> = (0..n)
                .map(|i| amp * (2.0 * PI * k as f32 * i as f32 / n as f32 + phase).cos() + 0.25)
                .collect();
            let (coefs, dc) = analyze_signal(&samples, 1, 1.0);
            assert_eq!(coefs.len(), 1, "n={n} k={k}: {coefs:?}");
            let (kk, mag, ph) = coefs[0];
            assert_eq!(kk, k);
            // At Nyquist cos(π i + φ) = cos φ · (−1)^i: the phase collapses into
            // the (real) amplitude, |X| = A n |cos φ|; elsewhere |X| = A n / 2
            let expected_mag = if 2 * k == n {
                amp * n as f32 * phase.cos().abs()
            } else {
                amp * n as f32 / 2.0
            };
            assert!(
                (mag - expected_mag).abs() < 1e-3 * expected_mag,
                "n={n} k={k}: {mag} vs {expected_mag}"
            );
            if 2 * k != n {
                assert!((ph - phase).abs() < 1e-3, "n={n} k={k}: phase {ph}");
            }
            let recon = generate_from_coefficients(n, &coefs, dc);
            for (a, b) in recon.iter().zip(&samples) {
                assert!((a - b).abs() < 1e-4, "n={n} k={k}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn full_spectrum_reconstructs_any_signal() {
    // Fourier completeness: keeping all n/2 bins reproduces the input
    for n in [4usize, 9, 32, 65, 128] {
        let signal: Vec<f32> = (0..n)
            .map(|i| ((i * i) % 13) as f32 - 6.0 + (i as f32).sin())
            .collect();
        let (coefs, dc) = analyze_signal(&signal, n, 1.0);
        let recon = generate_from_coefficients(n, &coefs, dc);
        for (a, b) in recon.iter().zip(&signal) {
            assert!((a - b).abs() < 2e-3, "n={n}: {a} vs {b}");
        }
    }
}

#[test]
fn multi_sine_is_sum_of_single_sines() {
    let comps = [(1.0f32, 2.0f32, 0.0f32), (3.0, 0.5, 1.0), (7.5, 1.25, -2.0)];
    for n in [1usize, 7, 64, 333] {
        let multi = generate_multi_sine(n, &comps, 0.75);
        let singles: Vec<Vec<f32>> = comps
            .iter()
            .map(|&(f, a, p)| generate_sine_wave(n, f, a, p, 0.0))
            .collect();
        for i in 0..n {
            let sum: f32 = singles.iter().map(|s| s[i]).sum::<f32>() + 0.75;
            assert!((multi[i] - sum).abs() < 1e-5, "n={n} i={i}");
            // closed form
            let expected: f32 = comps
                .iter()
                .map(|&(f, a, p)| a * (2.0 * PI * f * i as f32 / n as f32 + p).sin())
                .sum::<f32>()
                + 0.75;
            assert!((multi[i] - expected).abs() < 1e-4);
        }
    }
}

#[cfg(feature = "fft")]
#[test]
fn fft_path_matches_naive_dft_on_random_signals() {
    use alice_zip::generators::analyze_signal_fft;
    let mut lcg = 12345u64;
    for n in [2usize, 3, 8, 17, 64, 100, 256, 1000] {
        let signal: Vec<f32> = (0..n)
            .map(|_| {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((lcg >> 40) as f32 / 16_777_216.0) * 10.0 - 5.0
            })
            .collect();
        let (a, dc_a) = analyze_signal(&signal, n, 1.0);
        let (b, dc_b) = analyze_signal_fft(&signal, n, 1.0);
        assert!((dc_a - dc_b).abs() < 1e-4);
        assert_eq!(a.len(), b.len(), "n={n}");
        // Compare per bin (sort by k), not by rank: ties may reorder
        let mut a_by_k = a.clone();
        let mut b_by_k = b.clone();
        a_by_k.sort_by_key(|c| c.0);
        b_by_k.sort_by_key(|c| c.0);
        for ((ka, ma, pa), (kb, mb, pb)) in a_by_k.iter().zip(&b_by_k) {
            assert_eq!(ka, kb);
            assert!(
                (ma - mb).abs() < 1e-2 + 1e-3 * ma,
                "n={n} k={ka}: {ma} vs {mb}"
            );
            if *ma > 1e-2 {
                let dp = (pa - pb).abs();
                assert!(
                    dp < 1e-2 || (dp - 2.0 * PI).abs() < 1e-2,
                    "n={n} k={ka}: {pa} vs {pb}"
                );
            }
        }
        // f32 FFT round-off grows with n (the naive path accumulates in f64)
        let tol = 1e-3 + 2e-6 * n as f32;
        let ra = generate_from_coefficients(n, &a, dc_a);
        let rb = generate_from_coefficients(n, &b, dc_b);
        for ((x, y), s) in ra.iter().zip(&rb).zip(&signal) {
            assert!((x - y).abs() < tol, "n={n}: {x} vs {y}");
            assert!((x - s).abs() < tol, "n={n}: {x} vs {s}");
        }
    }
}

// ---------------------------------------------------------------- noise

#[test]
fn gradient_noise_is_zero_on_lattice_and_bounded() {
    for seed in [0u64, 1, 42, u64::MAX] {
        let p = PerlinNoise::new(seed);
        for x in -5..6 {
            for y in -5..6 {
                assert_eq!(p.noise2d(x as f32, y as f32), 0.0);
            }
        }
        for i in 0..2000 {
            let x = (i as f32) * 0.0731 - 50.0;
            let y = (i as f32) * 0.0173 + 3.0;
            let v = p.noise2d(x, y);
            // 2D gradient noise with unit gradients is bounded by √2 / 2
            assert!(v.abs() <= 0.7072, "({x}, {y}) = {v}");
            for octaves in 1..=6 {
                assert!(p.fbm(x, y, octaves, 0.5, 2.0).abs() <= 1.0 + 1e-6);
            }
        }
    }
}

#[test]
fn perlin_textures_are_deterministic_and_seed_sensitive() {
    let a = generate_perlin_advanced(37, 23, 7, 8.0, 3, 0.5, 2.0).unwrap();
    let b = generate_perlin_advanced(37, 23, 7, 8.0, 3, 0.5, 2.0).unwrap();
    let c = generate_perlin_advanced(37, 23, 8, 8.0, 3, 0.5, 2.0).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.len(), 37 * 23);
    assert!(a.iter().all(|v| (0.0..=1.0).contains(v)));
    // 2D default parameters == advanced with (0.5, 2.0)
    assert_eq!(
        generate_perlin_2d(16, 9, 3, 5.0, 4).unwrap(),
        generate_perlin_advanced(16, 9, 3, 5.0, 4, 0.5, 2.0).unwrap()
    );
    // Row-major: row y of the texture equals a 1-row texture offset in y? No —
    // instead check the single-octave texture equals the direct noise samples
    let p = PerlinNoise::new(3);
    let tex = generate_perlin_advanced(16, 9, 3, 5.0, 1, 0.5, 2.0).unwrap();
    for y in 0..9 {
        for x in 0..16 {
            let direct = (p.noise2d(x as f32 / 5.0, y as f32 / 5.0) + 1.0) * 0.5;
            assert!((tex[y * 16 + x] - direct.clamp(0.0, 1.0)).abs() < 1e-6);
        }
    }
}

/// Independent reference: the `libalice` 2.2.0 Perlin implementation (mul_add
/// form, rayon-free) — the `.alice` container persists textures generated by
/// it, so the unified law must reproduce them to floating-point round-off
mod libalice_2_2_reference {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const GRAD2: [[f32; 2]; 8] = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [S, S],
        [-S, S],
        [S, -S],
        [-S, -S],
    ];
    pub struct Ref {
        perm: [u8; 512],
    }
    impl Ref {
        pub fn new(seed: u64) -> Self {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut perm = [0u8; 512];
            for (i, p) in perm.iter_mut().enumerate().take(256) {
                *p = i as u8;
            }
            for i in (1..256).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }
            for i in 0..256 {
                perm[256 + i] = perm[i];
            }
            Self { perm }
        }
        fn fade(t: f32) -> f32 {
            t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
        }
        fn lerp(a: f32, b: f32, t: f32) -> f32 {
            t.mul_add(b - a, a)
        }
        fn grad(hash: usize, x: f32, y: f32) -> f32 {
            let g = &GRAD2[hash & 7];
            g[0].mul_add(x, g[1] * y)
        }
        pub fn noise2d(&self, x: f32, y: f32) -> f32 {
            let xi = x.floor() as i32;
            let yi = y.floor() as i32;
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
            total * (1.0 / max_value)
        }
    }
}

#[test]
fn perlin_matches_libalice_2_2_reference() {
    for seed in [0u64, 42, 1234, u64::MAX] {
        let ours = PerlinNoise::new(seed);
        let reference = libalice_2_2_reference::Ref::new(seed);
        for i in 0..4000 {
            let x = (i as f32) * 0.0311 - 40.0;
            let y = (i as f32) * 0.0177 + 1.0;
            let a = ours.noise2d(x, y);
            let b = reference.noise2d(x, y);
            assert!((a - b).abs() < 1e-6, "seed={seed} ({x}, {y}): {a} vs {b}");
            let fa = ours.fbm(x, y, 4, 0.5, 2.0);
            let fb = reference.fbm(x, y, 4, 0.5, 2.0);
            assert!(
                (fa - fb).abs() < 1e-6,
                "seed={seed} fbm ({x}, {y}): {fa} vs {fb}"
            );
        }
        // Whole texture as libalice generated it (row-major, [0, 1])
        let tex = generate_perlin_advanced(64, 48, seed, 10.0, 4, 0.5, 2.0).unwrap();
        for y in 0..48 {
            for x in 0..64 {
                let v = (reference.fbm(x as f32 / 10.0, y as f32 / 10.0, 4, 0.5, 2.0) + 1.0) * 0.5;
                assert!((tex[y * 64 + x] - v.clamp(0.0, 1.0)).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn fbm_1d_is_bounded_deterministic_and_octave_normalised() {
    for seed in [0u64, 9, 77] {
        for scale in [0.5f32, 1.0, 4.0, 64.0] {
            for octaves in 1..=5u32 {
                let a = generate_fbm_1d(200, seed, scale, octaves, 0.5, 2.0).unwrap();
                let b = generate_fbm_1d(200, seed, scale, octaves, 0.5, 2.0).unwrap();
                assert_eq!(a, b);
                assert!(a.iter().all(|v| v.abs() <= 1.0 + 1e-6), "{a:?}");
            }
        }
    }
    // One octave: sample 0 sits on lattice point x = 0 → hash value itself,
    // and the series is a smoothstep between lattice hashes (monotone between
    // adjacent lattice points, so |Δ| per sample stays below the lattice gap)
    let s = generate_fbm_1d(400, 5, 2.0, 1, 0.5, 2.0).unwrap();
    for w in s.windows(2) {
        assert!((w[1] - w[0]).abs() <= 2.0 * 1.5 / 200.0 + 1e-6);
    }
}

// ---------------------------------------------------------------- zlib

#[test]
fn zlib_roundtrip_is_invariant_under_level() {
    use alice_zip::compression::{zlib_compress, zlib_decompress};
    for data in corpus() {
        for level in 0..=12u32 {
            let c = zlib_compress(&data, level).unwrap();
            assert_eq!(zlib_decompress(&c).unwrap(), data, "level={level}");
        }
    }
    assert!(zlib_decompress(b"not zlib at all").is_err());
    assert!(zlib_decompress(&[]).is_err());
}
