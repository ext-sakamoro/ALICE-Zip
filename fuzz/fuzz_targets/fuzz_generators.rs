//! Generators never panic on arbitrary finite input and honour their
//! documented bounds (entropy / noise range / fit error < threshold)
#![no_main]
use alice_zip::generators::{
    analyze_signal, fit_polynomial, fit_polynomial_unit, generate_fbm_1d,
    generate_from_coefficients, generate_multi_sine, generate_perlin_advanced,
    generate_polynomial, generate_polynomial_unit, generate_sine_wave,
};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
    max_degree: u8,
    max_coefficients: u8,
    energy_threshold: f32,
    seed: u64,
    scale: f32,
    octaves: u8,
    persistence: f32,
    lacunarity: f32,
    width: u8,
    height: u8,
}

fuzz_target!(|input: Input| {
    let samples: Vec<f32> = input
        .samples
        .iter()
        .copied()
        .filter(|v| v.is_finite() && v.abs() < 1e6)
        .take(512)
        .collect();
    let n = samples.len();

    // Polynomial: both conventions; a returned fit must satisfy its own threshold
    let thr = 1e-3;
    if let Some((coeffs, degree, err)) = fit_polynomial(&samples, usize::from(input.max_degree % 10), thr) {
        assert!(err < thr || degree == 0, "{err}");
        assert_eq!(coeffs.len(), degree + 1);
        assert_eq!(generate_polynomial(n, &coeffs).len(), n);
    }
    if let Some((coeffs, degree, err)) = fit_polynomial_unit(&samples, usize::from(input.max_degree % 10), thr) {
        assert!(err < thr || degree == 0, "{err}");
        assert_eq!(coeffs.len(), degree + 1);
        assert_eq!(generate_polynomial_unit(n, &coeffs).len(), n);
    }

    // Fourier: bins in range, at most max_coefficients, reconstruction has n samples
    let maxc = usize::from(input.max_coefficients);
    let (coefs, dc) = analyze_signal(&samples, maxc, input.energy_threshold);
    assert!(coefs.len() <= maxc.min(n / 2));
    for &(k, mag, phase) in &coefs {
        assert!(k >= 1 && k <= n / 2);
        assert!(mag.is_finite() && mag >= 0.0);
        assert!(phase.is_finite());
    }
    assert_eq!(generate_from_coefficients(n, &coefs, dc).len(), n);
    assert_eq!(generate_sine_wave(n, input.scale, input.persistence, input.lacunarity, dc).len(), n);
    assert_eq!(generate_multi_sine(n, &coefs.iter().map(|&(k, m, p)| (k as f32, m, p)).collect::<Vec<_>>(), dc).len(), n);

    // Noise: errors only for the documented invalid parameters, output bounded
    let octaves = u32::from(input.octaves % 9);
    let valid = input.scale.is_finite() && input.scale > 0.0 && octaves > 0;
    let fbm = generate_fbm_1d(n, input.seed, input.scale, octaves, input.persistence, input.lacunarity);
    assert_eq!(fbm.is_ok(), valid);
    if let Ok(v) = &fbm {
        assert_eq!(v.len(), n);
        if input.persistence >= 0.0 && input.persistence.is_finite() && input.lacunarity.is_finite() {
            assert!(v.iter().all(|x| x.is_nan() || x.abs() <= 1.0 + 1e-5), "{v:?}");
        }
    }
    let (w, h) = (usize::from(input.width % 32), usize::from(input.height % 32));
    let tex = generate_perlin_advanced(w, h, input.seed, input.scale, octaves, input.persistence, input.lacunarity);
    assert_eq!(tex.is_ok(), valid);
    if let Ok(t) = tex {
        assert_eq!(t.len(), w * h);
        assert!(t.iter().all(|x| x.is_nan() || (0.0..=1.0).contains(x)));
    }
});
