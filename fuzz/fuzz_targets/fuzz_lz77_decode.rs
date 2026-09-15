//! lz77_decode never panics on arbitrary token streams (Ok or InvalidData)
#![no_main]
use alice_zip::lz77::{lz77_decode, LzToken};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Tok {
    offset: u16,
    length: u16,
    literal: u8,
}

fuzz_target!(|tokens: Vec<Tok>| {
    if tokens.len() > 512 {
        return;
    }
    let tokens: Vec<LzToken> = tokens
        .iter()
        .map(|t| LzToken {
            offset: t.offset,
            length: t.length,
            literal: t.literal,
        })
        .collect();
    if let Ok(out) = lz77_decode(&tokens) {
        // Every token contributes exactly length + 1 bytes
        let expected: usize = tokens.iter().map(|t| t.length as usize + 1).sum();
        assert_eq!(out.len(), expected);
    }
});
