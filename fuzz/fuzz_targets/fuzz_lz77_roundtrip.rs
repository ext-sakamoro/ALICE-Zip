//! lz77_decode(lz77_encode(data, w, l)) == Ok(data) for every data / w / l
//! (w, l drawn from the input so the parameter space is fuzzed as well)
#![no_main]
use alice_zip::lz77::{lz77_decode, lz77_encode};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    if input.len() < 4 || input.len() > 4096 {
        return;
    }
    let window = usize::from(u16::from_le_bytes([input[0], input[1]])) + 1;
    let lookahead = usize::from(u16::from_le_bytes([input[2], input[3]]));
    let data = &input[4..];
    let tokens = lz77_encode(data, window, lookahead);
    assert!(tokens.len() <= data.len().max(1));
    let decoded = lz77_decode(&tokens).expect("encoder output must decode");
    assert_eq!(decoded, data);
});
