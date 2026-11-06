//! The bits of what the media generators compute, as SHA-256 over the little-endian output.
//!
//! The generators call alice-det-math, so these hold on every target and move only when a function they call
//! moves. The phases cross 2^13, where `sin` and `cos` of alice-det-math changed once. The output is not
//! stamped with an identifier and no file records one: `alice_zip::law::SEMANTICS_ID` does not name it, so
//! these digests are what fixes it.

use alice_core::media::{
    AudioGenerator, AudioParams, AudioPattern, ImageGenerator, ImageParams, ImagePattern,
};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn digest_f32(samples: &[f32]) -> String {
    assert!(!samples.is_empty(), "nothing compared");
    let mut h = Sha256::new();
    for s in samples {
        h.update(s.to_bits().to_le_bytes());
    }
    hex(&h.finalize())
}

fn digest_u8(bytes: &[u8]) -> String {
    assert!(!bytes.is_empty(), "nothing compared");
    hex(&Sha256::digest(bytes))
}

fn audio(pattern: AudioPattern, components: Vec<(f32, f32, f32)>) -> Vec<f32> {
    AudioGenerator::new().generate(&AudioParams {
        pattern,
        sample_rate: 44100,
        duration_secs: 4.0,
        frequency: 440.0,
        amplitude: 0.75,
        components,
        seed: 7,
        ..AudioParams::default()
    })
}

fn image(pattern: ImagePattern) -> Vec<u8> {
    ImageGenerator::new().generate(&ImageParams {
        pattern,
        width: 96,
        height: 64,
        colors: vec![[12, 200, 77], [250, 3, 140]],
        block_size: 7,
        angle_degrees: 33.0,
        seed: 7,
    })
}

/// 4 s of 440 Hz reaches an angle past 2^13
#[test]
fn sine_past_two_to_the_13_is_the_recorded_bits() {
    let s = audio(AudioPattern::Sine, Vec::new());
    assert_eq!(
        digest_f32(&s),
        "e338189ce7e717595862bc1c540d1cb6add7305b04bd03455d685e5ba0bb6c1b"
    );
}

/// a phase past 2^13 from the first sample
#[test]
fn multi_sine_with_a_large_phase_is_the_recorded_bits() {
    let s = audio(
        AudioPattern::MultiSine,
        vec![(220.0, 0.5, 9000.0), (660.5, 0.25, 0.3)],
    );
    assert_eq!(
        digest_f32(&s),
        "17afe55f06bcbe14ca482d2f541551e39b902228866debd6af9ed7767f949a53"
    );
}

#[test]
fn white_noise_is_the_recorded_bits() {
    let s = audio(AudioPattern::WhiteNoise, Vec::new());
    assert_eq!(
        digest_f32(&s),
        "91a5591effee688b709b9535929c253e69607ada5f859d52f87d3cc2f4e9674a"
    );
}

#[test]
fn linear_gradient_is_the_recorded_bits() {
    assert_eq!(
        digest_u8(&image(ImagePattern::LinearGradient)),
        "7928cdd4a4d52be8ee025723d679d26bf74149cc52073e0d47604390fd6b7c69"
    );
}

#[test]
fn radial_gradient_is_the_recorded_bits() {
    assert_eq!(
        digest_u8(&image(ImagePattern::RadialGradient)),
        "7f8bbe6b4444b5b89b4925e390e8ba8a9c180443d422c3a08c631af2e1e9ad70"
    );
}

#[test]
fn checkerboard_is_the_recorded_bits() {
    assert_eq!(
        digest_u8(&image(ImagePattern::Checkerboard)),
        "e108e09f3359d840657dbaa2d6430f6fff40b4949b19b6822912ee1606230708"
    );
}
