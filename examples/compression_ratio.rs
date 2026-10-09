//! Measure the compression ratios the README quotes, instead of asserting them.
//!
//! ```text
//! cargo run --release --example compression_ratio --features lzma
//! ```
//!
//! A law-based pipeline has **two** numbers per input, and quoting only the
//! first is how "1400x" ended up next to the word "lossless" in this README:
//!
//! 1. **parameters only** — enough to regenerate an *approximation* of the
//!    signal. Tiny, and lossy: the fitted law differs from the sampled signal
//!    in the low bits of every sample.
//! 2. **parameters + residual** — enough to get the original samples back.
//!    Bigger, and the only one that may be called lossless.
//!
//! So every row prints both, with the error of (1), whether (2) came back bit
//! for bit, and what plain zlib does on the same bytes as the floor to beat.
//!
//! The residual here is the **bit pattern xor** against the model
//! (`compression::compress_residual_xor`), which is reversible by
//! construction. The older subtraction form is printed alongside it: it is
//! smaller on some signals and larger on others, but it is not exact — on the
//! sine it loses the 99 samples nearest the zero crossings.

use alice_zip::compression::{
    compress_residual_lossless, compress_residual_quantized, compress_residual_xor,
    decompress_residual_quantized, decompress_residual_xor, zlib_compress,
};
use alice_zip::generators::{
    analyze_signal, fit_polynomial_unit, generate_from_coefficients, generate_polynomial_unit,
    generate_sine_wave,
};

/// Deflate level for every container and for the baseline (same on both sides)
const LEVEL: u32 = 9;
/// Samples per case, matching the README's "100K samples"
const N: usize = 100_000;

/// Serialised Fourier law: `n` (u32) + dc (f32) + `(bin: u32, mag: f32, phase: f32)` each
fn fourier_param_bytes(coeff_count: usize) -> usize {
    4 + 4 + coeff_count * 12
}

/// Serialised polynomial law: `n` (u32) + one f64 per coefficient
fn polynomial_param_bytes(coeff_count: usize) -> usize {
    4 + coeff_count * 8
}

fn raw_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Row {
    name: &'static str,
    raw: usize,
    params: usize,
    /// max |original - model|: the error you accept if you keep only the law
    law_only_err: f32,
    xor_total: usize,
    xor_exact: bool,
    sub_total: usize,
    sub_exact: bool,
    quantized_total: usize,
    quantized_err: f32,
    zlib_baseline: usize,
}

#[allow(clippy::cast_precision_loss)]
fn ratio(raw: usize, got: usize) -> f64 {
    raw as f64 / got as f64
}

fn measure(name: &'static str, original: &[f32], model: &[f32], params: usize) -> Row {
    // (1) the law on its own
    let law_only_err = original
        .iter()
        .zip(model.iter())
        .map(|(o, m)| (o - m).abs())
        .fold(0.0f32, f32::max);

    // (2a) parameters + xor residual — exact by construction
    let xor = compress_residual_xor(original, model, LEVEL).expect("xor container");
    let back = decompress_residual_xor(&xor, model).expect("xor round trip");
    let xor_exact = back.len() == original.len()
        && original
            .iter()
            .zip(back.iter())
            .all(|(o, b)| o.to_bits() == b.to_bits());

    // (2b) parameters + subtraction residual — the older form, not exact
    let residual: Vec<f32> = original
        .iter()
        .zip(model.iter())
        .map(|(o, m)| o - m)
        .collect();
    let sub = compress_residual_lossless(&residual, LEVEL).expect("lossless container");
    let sub_exact = original
        .iter()
        .zip(model.iter())
        .zip(residual.iter())
        .all(|((o, m), r)| (m + r).to_bits() == o.to_bits());

    // (3) quantised residual — lossy on purpose
    let quantized = compress_residual_quantized(&residual, 16, LEVEL).expect("quantised container");
    let qback = decompress_residual_quantized(&quantized).expect("quantised round trip");
    let quantized_err = original
        .iter()
        .zip(model.iter())
        .zip(qback.iter())
        .map(|((o, m), r)| (m + r - o).abs())
        .fold(0.0f32, f32::max);

    Row {
        name,
        raw: original.len() * 4,
        params,
        law_only_err,
        xor_total: params + xor.len(),
        xor_exact,
        sub_total: params + sub.len(),
        sub_exact,
        quantized_total: params + quantized.len(),
        quantized_err,
        zlib_baseline: zlib_compress(&raw_bytes(original), LEVEL)
            .expect("zlib baseline")
            .len(),
    }
}

fn sine_case() -> Row {
    // 50 cycles over the window, amplitude 1.0 — the README's "freq=50Hz, amp=1.0"
    let original = generate_sine_wave(N, 50.0, 1.0, 0.0, 0.0);
    let (coeffs, dc) = analyze_signal(&original, 8, 0.999);
    let model = generate_from_coefficients(N, &coeffs, dc);
    measure(
        "Sine wave",
        &original,
        &model,
        fourier_param_bytes(coeffs.len()),
    )
}

fn polynomial_case() -> Row {
    let original = generate_polynomial_unit(N, &[0.25, -1.5, 2.0, -0.75]);
    let (fitted, _, _) = fit_polynomial_unit(&original, 3, 1e-12).expect("degree-3 fit exists");
    let model = generate_polynomial_unit(N, &fitted);
    measure(
        "Polynomial (degree 3)",
        &original,
        &model,
        polynomial_param_bytes(fitted.len()),
    )
}

fn gradient_case() -> Row {
    // descending coefficients: 1.0 * x + 0.0
    let original = generate_polynomial_unit(N, &[1.0, 0.0]);
    let (fitted, _, _) = fit_polynomial_unit(&original, 1, 1e-12).expect("degree-1 fit exists");
    let model = generate_polynomial_unit(N, &fitted);
    measure(
        "Linear gradient",
        &original,
        &model,
        polynomial_param_bytes(fitted.len()),
    )
}

fn noise_case() -> Row {
    // Deterministic pseudo-random samples: no law fits, so the pipeline is
    // reduced to compressing the whole signal — this is the fallback path.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let original: Vec<f32> = (0..N)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (state >> 40) as f32 / 16_777_216.0;
            u * 2.0 - 1.0
        })
        .collect();
    let model = vec![0.0f32; N];
    measure("Random data", &original, &model, 4)
}

fn main() {
    let rows = [
        sine_case(),
        polynomial_case(),
        gradient_case(),
        noise_case(),
    ];

    println!(
        "deflate level {LEVEL}, n = {N} f32 samples ({} bytes)\n",
        N * 4
    );

    println!("| Data type | Raw | Law + residual (xor) | Ratio | Bit-exact | zlib alone |");
    println!("|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} B | {} B | **{:.0}x** | {} | {} B ({:.1}x) |",
            r.name,
            r.raw,
            r.xor_total,
            ratio(r.raw, r.xor_total),
            if r.xor_exact { "yes" } else { "**NO**" },
            r.zlib_baseline,
            ratio(r.raw, r.zlib_baseline),
        );
    }

    println!("\n| Data type | Law parameters only | Ratio | Max abs error |");
    println!("|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} B | {:.0}x | {:.3e} |",
            r.name,
            r.params,
            ratio(r.raw, r.params),
            r.law_only_err,
        );
    }

    println!("\nOther containers, same inputs:\n");
    println!(
        "| Data type | Subtraction residual | Bit-exact | Quantised (16-bit) | Max abs error |"
    );
    println!("|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} B ({:.0}x) | {} | {} B ({:.0}x) | {:.3e} |",
            r.name,
            r.sub_total,
            ratio(r.raw, r.sub_total),
            if r.sub_exact { "yes" } else { "no" },
            r.quantized_total,
            ratio(r.raw, r.quantized_total),
            r.quantized_err,
        );
    }
}
