//! Residual containers between the released 0.10 and the current version
//!
//! The current writer records the semantics identifier of the model; the
//! released reader refuses those containers. What the released writer wrote
//! still decodes in the current reader, to the same samples, as legacy, and
//! the strict decoders refuse it.

use current::compression as now;
use released::compression as old;

fn signal() -> (Vec<f32>, Vec<f32>) {
    let original: Vec<f32> = (0..256u32)
        .map(|i| ((i * 37) % 101) as f32 / 7.0 - 5.0)
        .collect();
    let model: Vec<f32> = original.iter().map(|v| v * 0.999).collect();
    (original, model)
}

fn is_legacy_error(e: &std::io::Error) -> bool {
    e.get_ref()
        .and_then(|inner| inner.downcast_ref::<now::ResidualIdError>())
        == Some(&now::ResidualIdError::Legacy)
}

#[test]
fn the_released_reader_refuses_every_current_residual_container() {
    let (original, model) = signal();
    let quantized = now::compress_residual_quantized(&original, 16, 6).unwrap();
    let lossless = now::compress_residual_lossless(&original, 6).unwrap();
    let xor = now::compress_residual_xor(&original, &model, 6).unwrap();
    assert!(old::decompress_residual_quantized(&quantized).is_err());
    assert!(old::decompress_residual_lossless(&lossless).is_err());
    assert!(old::decompress_residual_xor(&xor, &model).is_err());
}

#[test]
fn released_residual_containers_decode_as_legacy_and_strict_refuses_them() {
    let (original, model) = signal();
    let quantized = old::compress_residual_quantized(&original, 16, 6).unwrap();
    let lossless = old::compress_residual_lossless(&original, 6).unwrap();
    let xor = old::compress_residual_xor(&original, &model, 6).unwrap();
    for c in [&quantized, &lossless, &xor] {
        assert_eq!(
            now::residual_container_status(c).unwrap(),
            now::ResidualStatus::Legacy
        );
    }
    assert_eq!(
        now::decompress_residual_quantized(&quantized).unwrap(),
        old::decompress_residual_quantized(&quantized).unwrap()
    );
    assert_eq!(
        now::decompress_residual_lossless(&lossless).unwrap(),
        original
    );
    assert_eq!(
        now::decompress_residual_xor(&xor, &model).unwrap(),
        original
    );
    assert!(is_legacy_error(
        &now::decompress_residual_quantized_strict(&quantized).unwrap_err()
    ));
    assert!(is_legacy_error(
        &now::decompress_residual_lossless_strict(&lossless).unwrap_err()
    ));
    assert!(is_legacy_error(
        &now::decompress_residual_xor_strict(&xor, &model).unwrap_err()
    ));
}
