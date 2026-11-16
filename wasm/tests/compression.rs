//! zlib and LZMA of the core in the browser target: the same bytes as on the host, round trips, and the bounded
//! decompress refusing more than its limit.
//!
//! Each test runs on the host (`cargo test`) and in Node (`cargo test --target wasm32-unknown-unknown`, with
//! wasm-bindgen-test's runner). Oracles: the recorded encoder identifiers (`DEFLATE_BYTES_ID`, `LZMA_BYTES_ID`, the
//! bytes the host build gives for the probe inputs) and the inputs themselves for the round trips.

use alice_zip::compression::{
    check_codec_bytes, codec_probe_inputs, lzma_compress, lzma_decompress, zlib_compress,
    zlib_decompress,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

/// This build's encoders give the recorded bytes, for both codecs (nothing compared fails)
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn the_encoders_give_the_recorded_bytes() {
    let compared = check_codec_bytes().expect("recorded codec bytes");
    assert_eq!(compared, ["deflate", "lzma"], "both codecs compared");
}

/// Every probe input comes back through zlib at every level, with the exact length as the limit
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn zlib_round_trips_every_probe_input() {
    let inputs = codec_probe_inputs();
    let mut compared = 0;
    for (name, data) in &inputs {
        for level in [1, 6, 9] {
            let packed = zlib_compress(data, level).expect("compress");
            let back = zlib_decompress(&packed, data.len()).expect("decompress within the limit");
            assert_eq!(&back, data, "{name} level {level}");
            compared += 1;
        }
    }
    assert!(compared >= 3, "compared {compared}");
}

/// Every probe input comes back through LZMA, with the exact length as the limit
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lzma_round_trips_every_probe_input() {
    let inputs = codec_probe_inputs();
    let mut compared = 0;
    for (name, data) in &inputs {
        let packed = lzma_compress(data, 6).expect("compress");
        let back = lzma_decompress(&packed, data.len()).expect("decompress within the limit");
        assert_eq!(&back, data, "{name}");
        compared += 1;
    }
    assert!(compared >= 1, "compared {compared}");
}

/// A limit one byte short of the content is refused, not cut short (zlib and LZMA)
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn the_bounded_decompress_refuses_a_limit_one_byte_short() {
    let inputs = codec_probe_inputs();
    let (name, data) = inputs
        .iter()
        .find(|(_, d)| !d.is_empty())
        .expect("a non-empty probe input");
    let z = zlib_compress(data, 6).expect("compress");
    assert!(zlib_decompress(&z, data.len() - 1).is_err(), "zlib {name}");
    let l = lzma_compress(data, 6).expect("compress");
    assert!(lzma_decompress(&l, data.len() - 1).is_err(), "lzma {name}");
}
