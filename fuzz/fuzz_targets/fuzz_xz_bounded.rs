//! xz and LZMA alone streams from arbitrary bytes: the container and LZMA2 chunk headers are parsed before
//! decoding, so the decoders never panic and never return more than the caller's limit
#![no_main]
use alice_zip::compression::{lzma_decompress, xz_decompress};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    if input.len() < 2 || input.len() > 65536 {
        return;
    }
    // the first two bytes choose the limit (up to 64 KiB), the rest is the stream
    let limit = usize::from(u16::from_le_bytes([input[0], input[1]]));
    let data = &input[2..];
    if let Ok(out) = xz_decompress(data, limit) {
        assert!(
            out.len() <= limit,
            "xz: {} bytes past the limit {limit}",
            out.len()
        );
    }
    if let Ok(out) = lzma_decompress(data, limit) {
        assert!(
            out.len() <= limit,
            "lzma: {} bytes past the limit {limit}",
            out.len()
        );
    }
});
