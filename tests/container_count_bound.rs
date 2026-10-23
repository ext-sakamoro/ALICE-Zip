//! A section count that cannot fit in the file is refused before anything is
//! allocated for the table, by the view and by the streamed reader.
//!
//! The header of a small container is changed to declare a million sections
//! (a table of tens of MiB) in a file of a few hundred bytes. The allocator
//! below records the peak number of live bytes; a reader that allocates the
//! table before checking the count against the file's length goes past the
//! bound. This file holds a single test so that nothing else allocates while
//! it runs.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use alice_zip::container::stream::{StreamError, StreamReader};
use alice_zip::container::{Container, ContainerError, ContainerView, Tag, ENTRY_LEN};

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

/// Sections declared by the changed header
const COUNT: u64 = 1_000_000;
/// Far below the table the count declares, far above the few hundred bytes of the file
const BOUND: usize = 1 << 20;

/// Runs `f` and returns the peak of live bytes it added
fn peak_of(f: impl FnOnce()) -> usize {
    let before = LIVE.load(Relaxed);
    PEAK.store(before, Relaxed);
    f();
    PEAK.load(Relaxed) - before
}

#[test]
fn a_section_count_larger_than_the_file_is_refused_before_allocating_the_table() {
    let mut c = Container::new([0; 32]);
    c.push(Tag::RAW, false, b"payload".to_vec());
    let mut bytes = c.to_bytes();
    // the section count is the u64 at offset 48 of the header
    bytes[48..56].copy_from_slice(&COUNT.to_le_bytes());
    assert!(
        COUNT as usize * ENTRY_LEN > 40 * BOUND,
        "the declared table is not large"
    );

    let view = peak_of(|| {
        assert_eq!(
            ContainerView::open(&bytes).err(),
            Some(ContainerError::Layout)
        );
    });
    assert!(
        view < BOUND,
        "the view allocated {view} bytes for the refused count"
    );

    let stream = peak_of(|| {
        let r = StreamReader::open_with_known(Cursor::new(&bytes[..]), &[]);
        assert!(
            matches!(r, Err(StreamError::Container(ContainerError::Layout))),
            "{:?}",
            r.err()
        );
    });
    assert!(
        stream < BOUND,
        "the streamed reader allocated {stream} bytes for the refused count"
    );
}
