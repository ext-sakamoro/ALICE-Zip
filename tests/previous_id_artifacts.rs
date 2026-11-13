//! Files written by the build before the function identifiers still read.
//!
//! `tests/data/previous-id/` holds what alice-zip at 7f80729 wrote (the build whose identifier
//! folded alice-det-math's whole identifier, `06c11067…`). The files are that build's output, not
//! made here: the header says so, every reader takes them as a verified earlier identifier, and
//! everything after the identifier is byte for byte what this build writes.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_zip::compression::{
    compress_residual_lossless, compress_residual_quantized, compress_residual_xor,
    decompress_residual_lossless_strict, decompress_residual_quantized_strict,
    decompress_residual_xor_strict, residual_container_status, ResidualStatus,
};
use alice_zip::container::ContainerView;
use alice_zip::law::{Provenance, SignalLaw, PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};

const OLD: &str = "06c11067caed2f27b3323c25b023b1237fb5f5936bf86624b2dfb33759183461";
const QUANTIZED: &[u8] = include_bytes!("data/previous-id/quantized.bin");
const LOSSLESS: &[u8] = include_bytes!("data/previous-id/lossless.bin");
const XOR: &[u8] = include_bytes!("data/previous-id/xor.bin");
const CONTAINER: &[u8] = include_bytes!("data/previous-id/container.bin");

fn old_id() -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&OLD[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

fn signal() -> (Vec<f32>, Vec<f32>) {
    let original: Vec<f32> = (0..512u32)
        .map(|i| ((i * 37) % 101) as f32 / 7.0 - 5.0)
        .collect();
    let model: Vec<f32> = original.iter().map(|v| v * 0.999).collect();
    (original, model)
}

#[test]
fn the_old_identifier_is_a_previous_one_and_not_this_one() {
    assert!(PREVIOUS_SEMANTICS_IDS.contains(&old_id()));
    assert_ne!(SEMANTICS_ID, old_id());
}

#[test]
fn old_residual_containers_read_as_a_verified_earlier_identifier() {
    let (original, model) = signal();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    for (name, data) in [
        ("quantized", QUANTIZED),
        ("lossless", LOSSLESS),
        ("xor", XOR),
    ] {
        assert_eq!(
            data[2..34],
            old_id(),
            "{name}: the file carries the old identifier"
        );
        assert_eq!(
            residual_container_status(data).unwrap(),
            ResidualStatus::VerifiedPrevious(old_id()),
            "{name}"
        );
    }
    let lossless = decompress_residual_lossless_strict(LOSSLESS, usize::MAX).unwrap();
    assert_eq!(bits(&lossless), bits(&original));
    let xor = decompress_residual_xor_strict(XOR, &model).unwrap();
    assert_eq!(bits(&xor), bits(&original));
    let quantized = decompress_residual_quantized_strict(QUANTIZED, usize::MAX).unwrap();
    assert_eq!(quantized.len(), original.len());
}

#[test]
fn past_the_identifier_this_build_writes_the_same_bytes() {
    let (original, model) = signal();
    for (name, old, new) in [
        (
            "quantized",
            QUANTIZED,
            compress_residual_quantized(&original, 16, 6).unwrap(),
        ),
        (
            "lossless",
            LOSSLESS,
            compress_residual_lossless(&original, 6).unwrap(),
        ),
        (
            "xor",
            XOR,
            compress_residual_xor(&original, &model, 6).unwrap(),
        ),
    ] {
        assert_eq!(new[2..34], SEMANTICS_ID, "{name}");
        assert_eq!(old[..2], new[..2], "{name}: marker and codec");
        assert_eq!(old[34..], new[34..], "{name}: the bits did not move");
    }
}

#[test]
fn an_old_container_reads_its_law_identifiers_as_this_arithmetic() {
    let points: Vec<(f64, f64)> = (0..16)
        .map(|i| (f64::from(i), 2.0 * f64::from(i) + 1.0))
        .collect();
    let law = SignalLaw::fit_polynomial(&points, 1, Provenance::new("test", "linear")).unwrap();
    let view = ContainerView::open(CONTAINER).unwrap();
    assert_eq!(view.semantics_id(), old_id());
    let rows = view.law_ids_for_this_arithmetic().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].law_id, law.law_id(&old_id()));
    assert_eq!(rows[0].semantics_id, old_id());
    // a changed identifier is refused, at the header's integrity check or as another arithmetic
    let mut other = CONTAINER.to_vec();
    let at = other.windows(32).position(|w| w == old_id()).unwrap();
    other[at] ^= 1;
    let refused =
        ContainerView::open(&other).map_or(true, |v| v.law_ids_for_this_arithmetic().is_err());
    assert!(refused);
}
