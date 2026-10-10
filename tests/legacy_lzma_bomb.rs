//! A payload that expands far beyond `original_size` is refused without the
//! expansion ever being held in memory.
//!
//! `legacy_lzma_bomb.alice` (`tests/data/container/make_legacy_lzma.py`) has
//! the header of the 4096-byte uint8 fixture and an xz stream of 61 KB that
//! expands to 400 MiB. The allocator below records the peak number of live
//! bytes; this file holds a single test so that nothing else allocates while
//! it runs.

#![cfg(feature = "lzma")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_zip::container::{decompress_legacy_alice_zip, ContainerError};

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

const BOMB: &[u8] = include_bytes!("data/container/legacy_lzma_bomb.alice");

/// Generous next to the 4096-byte result, small next to the 400 MiB
/// expansion.
const BOUND: usize = 1 << 20;

#[test]
fn an_expansion_past_original_size_is_refused_within_a_small_bound() {
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    assert_eq!(
        decompress_legacy_alice_zip(BOMB, u64::MAX),
        Err(ContainerError::LegacyPayload)
    );
    let peak = PEAK.load(Relaxed) - before;
    assert!(peak < BOUND, "peak {peak} bytes while refusing the payload");
}
