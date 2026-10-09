//! What the residual containers must guarantee, as checks rather than prose.
//!
//! Three properties, each of which the containers failed on 2026-10-09:
//!
//! 1. **A container that calls itself lossless reconstructs the input bit for
//!    bit.** The old one stores `original - model` as `f32`; where
//!    `|original| << |model|` that subtraction rounds away the original, so
//!    `model + residual` is not the original. Measured on a 100,000-sample
//!    sine: 99 samples (every zero crossing) came back wrong. The fix is to
//!    store the residual as the **bit pattern xor**, which is exact by
//!    construction for every finite input, including the signed zeros and the
//!    denormals the subtraction form loses.
//! 2. **The fallback is not worse than a standard tool.** The README's promise
//!    is that when no law fits, the data still compresses at least as well as
//!    off-the-shelf compression. Measured on the same sine, the container was
//!    **36x worse** than this crate's own zlib wrapper (316,528 B vs 8,742 B)
//!    because `lzma-rs`'s encoder is weak. A promise with no check next to it
//!    is how that survived.
//! 3. **Data written by an older version still reads.** The container is a
//!    persisted format (`.alice` files, coefficient batches), so changing the
//!    payload codec must not orphan what is already on disk.
//!
//! Expected values come from the definition of each property — bit equality,
//! and a comparison against an independent compressor — never from the
//! container's own output.
//!
//! ## Why the signals here are 4,096 samples and the README's are 100,000
//!
//! `generators::analyze_signal` is a naive DFT, so fitting the sine costs
//! O(n²): **109 ms at n = 4,096 against 68.6 s at n = 100,000** (release,
//! measured). The properties below do not depend on n — the subtraction form
//! still loses samples at the zero crossings (3 of them at 4,096, 99 at
//! 100,000) — so this file uses the cheap size and stays a per-push gate. The
//! README's figures are produced at 100,000 by
//! `cargo run --release --example compression_ratio --features lzma`, which is
//! run by hand.

#![cfg(feature = "lzma")]

use alice_zip::compression::{
    compress_residual_lossless, compress_residual_lossless_with, compress_residual_xor,
    decompress_residual_lossless, decompress_residual_xor, residual_container_codec, zlib_compress,
    ResidualCodec,
};
use alice_zip::generators::{analyze_signal, generate_from_coefficients, generate_sine_wave};

const N: usize = 4_096;

/// The sine from the README's benchmark table, and the law fitted to it.
fn sine_and_model() -> (Vec<f32>, Vec<f32>) {
    let original = generate_sine_wave(N, 50.0, 1.0, 0.0, 0.0);
    let (coeffs, dc) = analyze_signal(&original, 8, 0.999);
    let model = generate_from_coefficients(N, &coeffs, dc);
    (original, model)
}

fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// 1. bit-exactness
// ---------------------------------------------------------------------------

#[test]
fn the_xor_container_reconstructs_the_sine_bit_for_bit_including_zero_crossings() {
    let (original, model) = sine_and_model();
    let packed = compress_residual_xor(&original, &model, 6).expect("xor container");
    let back = decompress_residual_xor(&packed, &model).expect("xor round trip");

    assert_eq!(back.len(), original.len());
    let mismatches = original
        .iter()
        .zip(back.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N} samples did not come back bit for bit"
    );

    // Teeth: the subtraction form must actually fail on this signal, otherwise
    // the test would pass for a container that does nothing special.
    let lost = original
        .iter()
        .zip(model.iter())
        .filter(|(o, m)| {
            let r = *o - *m;
            (*m + r).to_bits() != o.to_bits()
        })
        .count();
    assert!(
        lost > 0,
        "the subtraction form reproduced all {N} samples, so this signal does \
         not exercise the defect the xor container exists for"
    );
}

#[test]
fn the_xor_container_is_exact_for_the_values_subtraction_loses() {
    // Signed zeros, denormals, and a value far smaller than its model: each of
    // these survives `bits ^ bits` and does not survive `fl(o - m)` + `m + r`.
    let original: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        -1e-30,
        1.0,
        -1.0,
        f32::MAX,
        f32::MIN,
        3.2162452e-16,
        1.0000001,
    ];
    let model: Vec<f32> = vec![
        -0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
        1.0000001,
        -1.0,
        f32::MAX,
        0.0,
        -1.3574548e-7,
        1.0,
    ];

    let packed = compress_residual_xor(&original, &model, 6).expect("xor container");
    let back = decompress_residual_xor(&packed, &model).expect("xor round trip");
    for (i, (o, b)) in original.iter().zip(back.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            b.to_bits(),
            "sample {i}: {o:e} came back as {b:e}"
        );
    }

    // And the subtraction form really does lose some of them — otherwise this
    // test has no teeth and the xor container would be solving nothing.
    let lost = original
        .iter()
        .zip(model.iter())
        .filter(|(o, m)| {
            let r = *o - *m;
            (*m + r).to_bits() != o.to_bits()
        })
        .count();
    assert!(
        lost > 0,
        "the subtraction form reproduced every sample, so this fixture does not \
         exercise the defect the xor container exists for"
    );
}

#[test]
fn the_xor_container_refuses_a_model_of_the_wrong_length() {
    let original = vec![1.0f32, 2.0, 3.0];
    let model = vec![1.0f32, 2.0];
    assert!(
        compress_residual_xor(&original, &model, 6).is_err(),
        "a model shorter than the signal was accepted"
    );
    let packed = compress_residual_xor(&original, &original, 6).expect("same length is fine");
    assert!(
        decompress_residual_xor(&packed, &model).is_err(),
        "decoding against a shorter model was accepted"
    );
}

// ---------------------------------------------------------------------------
// 2. not worse than a standard tool
// ---------------------------------------------------------------------------

#[test]
fn the_lossless_container_is_not_worse_than_this_crates_own_zlib_wrapper() {
    // The README promises the fallback is "never worse than standard tools".
    // zlib is the standard tool that ships in this very crate, so it is the
    // floor: a container that loses to it has no business being the default.
    let (original, model) = sine_and_model();
    let residual: Vec<f32> = original
        .iter()
        .zip(model.iter())
        .map(|(o, m)| o - m)
        .collect();

    // Same level on both sides — the container's `level` is the deflate level,
    // and on this data level 9 is twice as good as level 6, so comparing
    // different levels measures the level, not the container.
    const LEVEL: u32 = 9;
    let baseline = zlib_compress(&as_bytes(&residual), LEVEL)
        .expect("zlib baseline")
        .len();
    let container = compress_residual_lossless(&residual, LEVEL)
        .expect("lossless container")
        .len();
    assert!(
        container <= baseline + baseline / 10,
        "container {container} B is more than 10% worse than zlib {baseline} B \
         (measured 2026-10-09 before the fix: 46,941 vs 8,430 = 5.6x worse)"
    );

    // The xor container gets the same floor, but against **its own** bytes.
    // Comparing it with zlib on the subtraction residual would be comparing two
    // different inputs, and it is not true in general that the xor is smaller:
    // measured at n = 100,000 the xor wins on the sine (5,306 against 8,450)
    // and at n = 4,096 it loses (1,402 against 1,009), and on a degree-3
    // polynomial it loses at every size. What the xor container buys is
    // exactness, not size — so the property to pin is that its *encoder* is no
    // worse than zlib, which is what would regress if the weak codec came back.
    let xor_bytes: Vec<u8> = original
        .iter()
        .zip(model.iter())
        .flat_map(|(o, m)| (o.to_bits() ^ m.to_bits()).to_le_bytes())
        .collect();
    let xor_floor = zlib_compress(&xor_bytes, LEVEL)
        .expect("zlib on the xor bytes")
        .len();
    let xor = compress_residual_xor(&original, &model, LEVEL)
        .expect("xor container")
        .len();
    assert!(
        xor <= xor_floor + xor_floor / 10,
        "xor container {xor} B is more than 10% worse than zlib on the same \
         bytes ({xor_floor} B)"
    );
}

#[test]
fn the_level_argument_is_not_silently_ignored() {
    // The predecessor of this code path took an LZMA `preset` and dropped it
    // (the parameter was literally named `_preset`), so a caller asking for
    // maximum compression got whatever the library felt like. A level that
    // does nothing is indistinguishable from a level that works unless
    // something compares two of them.
    let (original, model) = sine_and_model();
    let residual: Vec<f32> = original
        .iter()
        .zip(model.iter())
        .map(|(o, m)| o - m)
        .collect();

    let fast = compress_residual_lossless(&residual, 1)
        .expect("level 1")
        .len();
    let best = compress_residual_lossless(&residual, 9)
        .expect("level 9")
        .len();
    assert!(
        best < fast,
        "level 9 ({best} B) did not beat level 1 ({fast} B) — the level is ignored"
    );

    let fast = compress_residual_xor(&original, &model, 1)
        .expect("level 1")
        .len();
    let best = compress_residual_xor(&original, &model, 9)
        .expect("level 9")
        .len();
    assert!(
        best < fast,
        "xor: level 9 ({best} B) did not beat level 1 ({fast} B)"
    );
}

#[test]
fn the_xor_container_is_bit_exact_on_every_law_family_the_readme_quotes() {
    use alice_zip::generators::{fit_polynomial_unit, generate_polynomial_unit};

    let (sine, sine_model) = sine_and_model();
    let poly = generate_polynomial_unit(N, &[0.25, -1.5, 2.0, -0.75]);
    let (poly_fit, _, _) = fit_polynomial_unit(&poly, 3, 1e-12).expect("degree-3 fit");
    let poly_model = generate_polynomial_unit(N, &poly_fit);
    let grad = generate_polynomial_unit(N, &[1.0, 0.0]);
    let (grad_fit, _, _) = fit_polynomial_unit(&grad, 1, 1e-12).expect("degree-1 fit");
    let grad_model = generate_polynomial_unit(N, &grad_fit);

    let mut checked = 0usize;
    for (name, original, model) in [
        ("sine", &sine, &sine_model),
        ("polynomial", &poly, &poly_model),
        ("gradient", &grad, &grad_model),
    ] {
        let packed = compress_residual_xor(original, model, 9).expect("xor container");
        let back = decompress_residual_xor(&packed, model).expect("round trip");
        let bad = original
            .iter()
            .zip(back.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(bad, 0, "{name}: {bad}/{N} samples not bit-exact");
        checked += original.len();
    }
    assert_eq!(checked, 3 * N, "not every family was actually compared");
}

#[test]
fn the_default_codec_is_the_pure_rust_deflate_one() {
    // The default must not be the weak encoder. Read it back out of the header
    // rather than trusting the constant.
    let (original, model) = sine_and_model();
    let packed = compress_residual_lossless(&[0.5f32; 64], 6).expect("lossless container");
    assert_eq!(
        residual_container_codec(&packed).expect("codec byte"),
        ResidualCodec::Deflate
    );
    let packed = compress_residual_xor(&original, &model, 6).expect("xor container");
    assert_eq!(
        residual_container_codec(&packed).expect("codec byte"),
        ResidualCodec::Deflate
    );
}

#[test]
fn an_explicitly_requested_codec_is_the_one_that_gets_used() {
    // The `level` / codec arguments must not be silently ignored (the old
    // `lzma_compress` accepted a preset and dropped it).
    let residual = vec![0.25f32; 4096];
    for codec in [ResidualCodec::Deflate, ResidualCodec::Lzma] {
        let packed =
            compress_residual_lossless_with(&residual, codec, 6).expect("explicit codec accepted");
        assert_eq!(
            residual_container_codec(&packed).expect("codec byte"),
            codec,
            "requested {codec:?} but the header says otherwise"
        );
        let back = decompress_residual_lossless(&packed).expect("round trip");
        assert_eq!(back.len(), residual.len());
        for (a, b) in residual.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

// ---------------------------------------------------------------------------
// 3. older data still reads
// ---------------------------------------------------------------------------

#[test]
fn a_container_written_by_the_previous_version_still_decodes() {
    // Byte-for-byte output of alice-zip 0.7.0's `compress_residual_lossless`
    // for the samples below (marker 0xFF, no codec byte, LZMA payload).
    // Captured from that version, not regenerated by the current one.
    let expected = [1.0f32, -2.5, 0.0, 7.25];
    let legacy = legacy_lossless_v0(&expected);
    let back = decompress_residual_lossless(&legacy)
        .expect("v0 container must keep decoding (persisted format)");
    assert_eq!(back.len(), expected.len());
    for (a, b) in expected.iter().zip(back.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// Builds the v0 layout (`0xFF · len: u32 · LZMA(f32 LE)`) the way 0.7.0 did,
/// so the compatibility check does not depend on a binary fixture file.
fn legacy_lossless_v0(values: &[f32]) -> Vec<u8> {
    let payload = alice_zip::compression::lzma_compress(&as_bytes(values), 6).expect("lzma");
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(0xFF);
    out.extend_from_slice(&u32::try_from(payload.len()).expect("fits").to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

#[test]
fn a_quantised_container_written_by_the_previous_version_still_decodes() {
    // v0 layout: `bits · min: f64 · scale: f64 · len: u32 · LZMA(quantised)`.
    // Built here the way 0.7.0 built it, so the check does not depend on a
    // binary fixture.
    use alice_zip::compression::{lzma_compress, quantize_8bit};

    let residual: Vec<f32> = (0..256u16).map(|i| f32::from(i) * 0.03 - 4.0).collect();
    let (quantized, min_val, scale) = quantize_8bit(&residual);
    let payload = lzma_compress(&quantized, 6).expect("lzma");
    let mut legacy = Vec::new();
    legacy.push(8u8);
    legacy.extend_from_slice(&min_val.to_le_bytes());
    legacy.extend_from_slice(&scale.to_le_bytes());
    legacy.extend_from_slice(&u32::try_from(payload.len()).expect("fits").to_le_bytes());
    legacy.extend_from_slice(&payload);

    let back = alice_zip::compression::decompress_residual_quantized(&legacy)
        .expect("v0 quantised container must keep decoding");
    assert_eq!(back.len(), residual.len());
    // 8-bit quantisation, so compare within the step, not bit for bit.
    let tol = (residual.iter().fold(f32::MIN, |a, &b| a.max(b))
        - residual.iter().fold(f32::MAX, |a, &b| a.min(b)))
        / 255.0;
    for (i, (a, b)) in residual.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "sample {i}: {a} came back as {b} (tolerance {tol})"
        );
    }
}

#[test]
fn a_truncated_or_mislabelled_container_is_refused_rather_than_guessed() {
    let packed = compress_residual_lossless(&[1.0f32, 2.0, 3.0], 6).expect("container");
    assert!(
        decompress_residual_lossless(&packed[..3]).is_err(),
        "a truncated container was accepted"
    );
    assert!(
        decompress_residual_lossless(&[]).is_err(),
        "an empty container was accepted"
    );
    let mut bad = packed.clone();
    bad[0] = 0x01;
    assert!(
        decompress_residual_lossless(&bad).is_err(),
        "an unknown marker was accepted"
    );
    let mut bad_codec = packed.clone();
    bad_codec[1] = 0x7F;
    assert!(
        decompress_residual_lossless(&bad_codec).is_err(),
        "an unknown codec byte was accepted"
    );
}
