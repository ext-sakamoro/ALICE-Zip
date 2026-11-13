//! The decompression bound holds for memory, not only for the returned length
//!
//! oracle (from the rules, not read from the code under test): a stream that expands past the caller's limit
//! is refused while the live allocation stays within twice the limit plus a small constant (the decoder's
//! window and the output it has produced), whatever the stream's headers declare: an LZMA "alone" stream
//! declaring a 1 GiB dictionary, a zlib stream and an xz stream of 64 MiB of zeros. A stream whose chunk header lies about its
//! size (the other headers made to agree) is refused when the decoded data diverges, also within the bound.
//! A decoder that buffers differently (for example a dependency that reserves the declared dictionary) makes
//! this go red even though it still returns `TooLarge`.
//!
//! Fixtures: tests/data/bomb/make_bombs.py (written by liblzma); the lying stream is derived here from one. The allocator records the peak of live
//! bytes; this file holds a single test so nothing else allocates while it measures.

#![cfg(feature = "lzma")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_zip::compression::{
    lzma_decompress, xz_decompress, zlib_compress, zlib_decompress, DecompressError,
};

struct Peak;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller's layout to the system allocator.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static A: Peak = Peak;

const ALONE_DICT_1G: &[u8] = include_bytes!("data/bomb/alone_zeros_64m_dict1g.lzma");
const XZ_ZEROS: &[u8] = include_bytes!("data/bomb/xz_zeros_64m.xz");
const XZ_VARIED_NONE: &[u8] = include_bytes!("data/bomb/xz_varied_none.xz");

const LIMIT: usize = 1 << 20;
/// The decoder's own state (range coder, probability tables, small buffers)
const SLACK: usize = 256 << 10;

fn too_large(e: &io::Error) -> Option<usize> {
    match e
        .get_ref()
        .and_then(|i| i.downcast_ref::<DecompressError>())
    {
        Some(DecompressError::TooLarge { limit }) => Some(*limit),
        _ => None,
    }
}

/// CRC-32 (IEEE, reflected), the checksum of the xz index
fn crc32(b: &[u8]) -> u32 {
    let mut c = !0_u32;
    for &x in b {
        c ^= u32::from(x);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// `stream` (one block, no check, no sizes in the block header) with its first LZMA chunk's unpacked size, the
/// index record and the index CRC lowered by one byte: the headers agree with each other, not with the data
fn lying(stream: &[u8]) -> Vec<u8> {
    let mut s = stream.to_vec();
    let at = 12 + (usize::from(s[12]) + 1) * 4;
    assert!(
        s[12 + 1] & 0xC0 == 0 && s[at] >= 0x80,
        "a block without sizes whose first chunk is LZMA"
    );
    let stored =
        (u32::from(s[at] & 0x1F) << 16) | u32::from(u16::from_be_bytes([s[at + 1], s[at + 2]]));
    let lowered = stored - 1;
    s[at] = (s[at] & 0xE0) | (lowered >> 16) as u8;
    s[at + 1..at + 3].copy_from_slice(&((lowered & 0xFFFF) as u16).to_be_bytes());
    // index: indicator, record count, (unpadded size, unpacked size) as varints, padding, CRC-32
    let backward = u32::from_le_bytes(s[s.len() - 8..s.len() - 4].try_into().unwrap()) as usize;
    let index_at = s.len() - 12 - (backward + 1) * 4;
    let mut i = index_at + 2; // indicator and a one-byte count
    while s[i] & 0x80 != 0 {
        i += 1;
    }
    i += 1; // past the unpadded size
    let (mut v, mut shift, start) = (0_u64, 0, i);
    loop {
        v |= u64::from(s[i] & 0x7F) << shift;
        shift += 7;
        i += 1;
        if s[i - 1] & 0x80 == 0 {
            break;
        }
    }
    let mut w = v - 1;
    for byte in &mut s[start..i] {
        let more = *byte & 0x80;
        *byte = (w & 0x7F) as u8 | more;
        w >>= 7;
    }
    assert_eq!(w, 0, "the index record keeps its length");
    let end = index_at + (backward + 1) * 4 - 4;
    let crc = crc32(&s[index_at..end]);
    s[end..end + 4].copy_from_slice(&crc.to_le_bytes());
    s
}

/// Peak live bytes while `f` runs, over what was live before
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    let out = f();
    (out, PEAK.load(Relaxed) - before)
}

#[test]
fn every_bomb_is_refused_within_twice_the_limit() {
    let bound = 2 * LIMIT + SLACK;

    let (r, peak) = peak_of(|| lzma_decompress(ALONE_DICT_1G, LIMIT));
    assert_eq!(too_large(&r.unwrap_err()), Some(LIMIT));
    assert!(
        peak <= bound,
        "alone, 1 GiB dictionary: peak {peak} bytes > {bound}"
    );

    // zlib: 64 MiB of zeros, compressed here (before measuring) by the crate's own writer
    let zlib_bomb = zlib_compress(&vec![0u8; 64 << 20], 9).unwrap();
    let (r, peak) = peak_of(|| zlib_decompress(&zlib_bomb, LIMIT));
    assert_eq!(too_large(&r.unwrap_err()), Some(LIMIT));
    assert!(peak <= bound, "zlib of 64 MiB: peak {peak} bytes > {bound}");

    let (r, peak) = peak_of(|| xz_decompress(XZ_ZEROS, LIMIT));
    assert_eq!(too_large(&r.unwrap_err()), Some(LIMIT));
    assert!(peak <= bound, "xz of 64 MiB: peak {peak} bytes > {bound}");

    // the chunk headers agree with the index and there is no check, so only the decoded data shows the lie
    let liar = lying(XZ_VARIED_NONE);
    assert!(xz_decompress(XZ_VARIED_NONE, LIMIT).is_ok());
    let (r, peak) = peak_of(|| xz_decompress(&liar, LIMIT));
    let e = r.expect_err("a stream whose chunk header lies was read");
    assert_eq!(too_large(&e), None, "{e}");
    assert!(
        !e.to_string().contains("single-block"),
        "refused by the parse, not the decoder: {e}"
    );
    assert!(peak <= bound, "lying chunk: peak {peak} bytes > {bound}");

    // the measurement itself has teeth: the full 64 MiB is far above the bound
    let (full, peak) = peak_of(|| xz_decompress(XZ_ZEROS, 64 << 20).unwrap().len());
    assert_eq!(full, 64 << 20);
    assert!(
        peak > bound,
        "the allocator did not see the expansion ({peak} bytes)"
    );
}
