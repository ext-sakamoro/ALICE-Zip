//! The semantics identifier of the residual containers
//!
//! Every container records the identifier the model was evaluated under, at
//! bytes `2..34`; the decoders compare the stored identifier with
//! `law::check_semantics`, not with their own build. Expected values come from
//! the documented layout and from the identifiers in `law`, never from the
//! decoder's output. The containers written before the identifier are checked
//! against the released 0.10 writer in `compat/v0_10_reader`.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_zip::compression::{
    compress_residual_lossless, compress_residual_quantized, compress_residual_xor,
    decompress_residual_lossless, decompress_residual_lossless_strict,
    decompress_residual_quantized, decompress_residual_quantized_strict, decompress_residual_xor,
    decompress_residual_xor_strict, residual_container_status, ResidualIdError, ResidualStatus,
};
use alice_zip::law::{PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};

fn signal() -> (Vec<f32>, Vec<f32>) {
    let original: Vec<f32> = (0..512u32)
        .map(|i| ((i * 37) % 101) as f32 / 7.0 - 5.0)
        .collect();
    let model: Vec<f32> = original.iter().map(|v| v * 0.999).collect();
    (original, model)
}

/// The three containers, each with its marker
fn containers() -> Vec<(&'static str, u8, Vec<u8>)> {
    let (original, model) = signal();
    vec![
        (
            "quantized",
            0xFB,
            compress_residual_quantized(&original, 16, 6).unwrap(),
        ),
        (
            "lossless",
            0xFA,
            compress_residual_lossless(&original, 6).unwrap(),
        ),
        (
            "xor",
            0xF9,
            compress_residual_xor(&original, &model, 6).unwrap(),
        ),
    ]
}

/// Decodes with the plain and the strict decoder of `name`
fn decode(name: &str, data: &[u8]) -> [std::io::Result<Vec<f32>>; 2] {
    let (_, model) = signal();
    match name {
        "quantized" => [
            decompress_residual_quantized(data),
            decompress_residual_quantized_strict(data),
        ],
        "lossless" => [
            decompress_residual_lossless(data),
            decompress_residual_lossless_strict(data),
        ],
        "xor" => [
            decompress_residual_xor(data, &model),
            decompress_residual_xor_strict(data, &model),
        ],
        other => panic!("unknown container {other}"),
    }
}

fn id_error(e: &std::io::Error) -> Option<ResidualIdError> {
    e.get_ref()
        .and_then(|inner| inner.downcast_ref::<ResidualIdError>())
        .copied()
}

#[test]
fn every_container_records_this_builds_identifier_and_reads_back() {
    let (original, _) = signal();
    let all = containers();
    assert_eq!(all.len(), 3);
    for (name, marker, c) in all {
        assert_eq!(c[0], marker, "{name}");
        assert_eq!(c[2..34], SEMANTICS_ID, "{name}");
        assert_eq!(
            residual_container_status(&c).unwrap(),
            ResidualStatus::Verified,
            "{name}"
        );
        for got in decode(name, &c) {
            let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(got.len(), original.len(), "{name}");
        }
    }
}

#[test]
fn another_identifier_is_refused_naming_both_before_decoding() {
    let mut other = SEMANTICS_ID;
    other[0] ^= 0x01;
    for (name, _, mut c) in containers() {
        c[2..34].copy_from_slice(&other);
        // the payload is made undecodable too: the identifier must be what refuses
        let last = c.len() - 1;
        c[last] ^= 0xFF;
        let want = ResidualIdError::Mismatch {
            stored: other,
            running: SEMANTICS_ID,
        };
        let status = residual_container_status(&c).unwrap_err();
        assert_eq!(id_error(&status), Some(want), "{name}");
        for got in decode(name, &c) {
            let e = got.unwrap_err();
            assert_eq!(id_error(&e), Some(want), "{name}: {e}");
            let text = e.to_string();
            let hex = |b: &[u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
            assert!(text.contains(&hex(&other)), "{name}: {text}");
            assert!(text.contains(&hex(&SEMANTICS_ID)), "{name}: {text}");
        }
    }
}

#[test]
fn a_verified_earlier_identifier_decodes_as_verified_previous() {
    assert!(!PREVIOUS_SEMANTICS_IDS.is_empty());
    let (original, _) = signal();
    for &previous in PREVIOUS_SEMANTICS_IDS {
        for (name, _, mut c) in containers() {
            c[2..34].copy_from_slice(&previous);
            assert_eq!(
                residual_container_status(&c).unwrap(),
                ResidualStatus::VerifiedPrevious(previous),
                "{name}"
            );
            for got in decode(name, &c) {
                assert_eq!(got.unwrap().len(), original.len(), "{name}");
            }
        }
    }
}

#[test]
fn a_container_written_before_the_identifier_is_legacy_and_strict_refuses_it() {
    // the layout before the identifier: the same container without bytes 2..34,
    // under the earlier marker (0xFC / 0xFD / 0xFE)
    let (original, _) = signal();
    for (name, marker, c) in containers() {
        let earlier = match marker {
            0xFB => 0xFC,
            0xFA => 0xFD,
            _ => 0xFE,
        };
        let mut legacy = vec![earlier, c[1]];
        legacy.extend_from_slice(&c[34..]);
        assert_eq!(
            residual_container_status(&legacy).unwrap(),
            ResidualStatus::Legacy,
            "{name}"
        );
        let [plain, strict] = decode(name, &legacy);
        assert_eq!(plain.unwrap().len(), original.len(), "{name}");
        assert_eq!(
            id_error(&strict.unwrap_err()),
            Some(ResidualIdError::Legacy),
            "{name}"
        );
    }
}

#[test]
fn a_header_too_short_for_the_identifier_is_refused() {
    for (name, _, c) in containers() {
        for n in [0, 1, 2, 33] {
            assert!(residual_container_status(&c[..n]).is_err(), "{name} {n}");
            for got in decode(name, &c[..n]) {
                assert!(got.is_err(), "{name} {n}");
            }
        }
    }
}
