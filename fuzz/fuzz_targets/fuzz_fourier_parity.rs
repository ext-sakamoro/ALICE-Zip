//! Parity fuzz (two evaluation paths of one law): the naive f64-accumulated DFT
//! and the rustfft path must select the same bins and reconstruct the same
//! signal; with every bin kept the reconstruction must equal the input
#![no_main]
use alice_zip::generators::{analyze_signal, analyze_signal_fft, generate_from_coefficients};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: Vec<f32>| {
    let signal: Vec<f32> = input
        .into_iter()
        .filter(|v| v.is_finite() && v.abs() < 1e3)
        .take(1024)
        .collect();
    let n = signal.len();
    if n < 2 {
        return;
    }
    let (a, dc_a) = analyze_signal(&signal, n, 1.0);
    let (b, dc_b) = analyze_signal_fft(&signal, n, 1.0);
    assert!((dc_a - dc_b).abs() <= 1e-4 * (1.0 + dc_a.abs()), "dc {dc_a} vs {dc_b}");

    // Same bin set (numerical-zero bins may be dropped by one path only when
    // the other reports them as ~0 too)
    let mut ka: Vec<usize> = a.iter().map(|c| c.0).collect();
    let mut kb: Vec<usize> = b.iter().map(|c| c.0).collect();
    ka.sort_unstable();
    kb.sort_unstable();
    let scale = signal.iter().fold(0.0f32, |m, v| m.max(v.abs())) * n as f32;
    let tol = 1e-4 * scale + 1e-3;
    for k in ka.iter().chain(kb.iter()) {
        let ma = a.iter().find(|c| c.0 == *k).map_or(0.0, |c| c.1);
        let mb = b.iter().find(|c| c.0 == *k).map_or(0.0, |c| c.1);
        assert!((ma - mb).abs() <= tol, "bin {k}: {ma} vs {mb} (tol {tol})");
    }

    // Reconstruction parity and completeness
    let ra = generate_from_coefficients(n, &a, dc_a);
    let rb = generate_from_coefficients(n, &b, dc_b);
    let amp = signal.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = 2e-4 * amp * (1.0 + (n as f32).log2()) + 1e-3;
    for ((x, y), s) in ra.iter().zip(&rb).zip(&signal) {
        assert!((x - y).abs() <= tol, "recon {x} vs {y} (tol {tol})");
        assert!((x - s).abs() <= tol, "recon {x} vs signal {s} (tol {tol})");
    }
});
