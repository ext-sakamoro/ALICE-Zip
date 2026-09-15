//! zlib: round trip at every level, arbitrary bytes never panic the decoder
#![no_main]
use alice_zip::compression::{zlib_compress, zlib_decompress};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    if input.is_empty() || input.len() > 16384 {
        return;
    }
    let level = u32::from(input[0]);
    let data = &input[1..];
    let c = zlib_compress(data, level).expect("compress");
    assert_eq!(zlib_decompress(&c).expect("decompress"), data);
    // Arbitrary input to the decoder: Ok or Err, never a panic
    let _ = zlib_decompress(data);
});
