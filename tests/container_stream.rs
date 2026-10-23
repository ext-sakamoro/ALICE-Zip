//! Oracles for writing and reading a container as a stream
//! (`alice_zip::container::stream`)
//!
//! The streamed writer must produce the same bytes as `Container::to_bytes` for
//! the same sections, and the streamed reader must accept and refuse exactly
//! what `ContainerView` accepts and refuses: both are compared on the reference
//! fixtures (written by `tests/data/container/container_ref.py`) and on every
//! single-byte change of two of them.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use alice_zip::container::stream::{StreamError, StreamReader, StreamWriter};
use alice_zip::container::{Container, ContainerView, Tag};
use sha2::{Digest, Sha256};

const ONE: &[u8] = include_bytes!("data/container/one.bin");
const EMPTY: &[u8] = include_bytes!("data/container/empty.bin");
const DUP_TAG: &[u8] = include_bytes!("data/container/dup_tag.bin");
const ZERO_LEN: &[u8] = include_bytes!("data/container/zero_len.bin");
const UNKNOWN_NONCRITICAL: &[u8] = include_bytes!("data/container/unknown_noncritical.bin");
const BUNDLE: &[u8] = include_bytes!("data/container/bundle.bin");
const LEGACY_V1: &[u8] = include_bytes!("data/container/legacy_v1.alice");
const V1_ONE: &[u8] = include_bytes!("data/container/v1_one.bin");
const V1_LIDS: &[u8] = include_bytes!("data/container/v1_lids.bin");

const PRODUCT: Tag = Tag(*b"ZZZZ");

/// The fixtures written by the current format (they start with the format identifier)
fn current() -> Vec<(&'static str, &'static [u8])> {
    vec![
        ("one", ONE),
        ("empty", EMPTY),
        ("dup_tag", DUP_TAG),
        ("zero_len", ZERO_LEN),
        ("unknown_noncritical", UNKNOWN_NONCRITICAL),
        ("bundle", BUNDLE),
    ]
}

/// Writes the sections of `c` (after its format identifier) with the streamed writer
fn stream_write(c: &Container) -> Vec<u8> {
    assert_eq!(c.sections[0].tag, Tag::FMID);
    let declared: Vec<(Tag, bool)> = c.sections[1..]
        .iter()
        .map(|s| (s.tag, s.critical))
        .collect();
    let mut w = StreamWriter::new(Cursor::new(Vec::new()), c.semantics_id, &declared).unwrap();
    for s in &c.sections[1..] {
        // through the Write side, in uneven pieces
        let mut sec = w.begin_section().unwrap();
        for piece in s.payload.chunks(7) {
            sec.write_all(piece).unwrap();
        }
        let sha = sec.finish().unwrap();
        assert_eq!(sha, <[u8; 32]>::from(Sha256::digest(&s.payload)));
    }
    w.finish().unwrap().into_inner()
}

#[test]
fn a_streamed_write_is_byte_identical_to_to_bytes() {
    let mut compared = 0;
    for (name, bytes) in current() {
        let c = Container::from_bytes_with_known(bytes, &[PRODUCT]).unwrap();
        let streamed = stream_write(&c);
        assert_eq!(streamed, c.to_bytes(), "{name}: differs from to_bytes");
        assert_eq!(
            streamed, bytes,
            "{name}: differs from the reference fixture"
        );
        compared += 1;
    }
    // a large payload too (more than any internal buffer)
    let mut c = Container::new([0x11; 32]);
    c.push(
        Tag::RAW,
        false,
        (0..1_048_576u32).map(|i| (i % 251) as u8).collect(),
    );
    c.push(PRODUCT, true, b"tail".to_vec());
    assert_eq!(stream_write(&c), c.to_bytes(), "large payload");
    assert_eq!(compared, 6);
}

/// The verdict of a reader on some bytes: the identifier and sections, or the error text
type Verdict = Result<(Vec<(Tag, bool, Vec<u8>)>, [u8; 32]), String>;

fn view_verdict(bytes: &[u8], known: &[Tag]) -> Verdict {
    let v = ContainerView::open_with_known(bytes, known).map_err(|e| format!("{e:?}"))?;
    let mut sections = Vec::new();
    for i in 0..v.len() {
        sections.push((
            v.tag(i),
            v.critical(i),
            v.section(i).map_err(|e| format!("{e:?}"))?.to_vec(),
        ));
    }
    Ok((sections, v.id()))
}

fn stream_verdict(bytes: &[u8], known: &[Tag]) -> Verdict {
    let mut r =
        StreamReader::open_with_known(Cursor::new(bytes.to_vec()), known).map_err(|e| match e {
            StreamError::Container(e) => format!("{e:?}"),
            other => format!("{other:?}"),
        })?;
    let mut sections = Vec::new();
    for i in 0..r.len() {
        let (tag, critical) = (r.tag(i), r.critical(i));
        let mut payload = Vec::new();
        let read = r.section(i).and_then(|mut s| {
            s.read_to_end(&mut payload).map_err(StreamError::Io)?;
            s.finish()
        });
        read.map_err(|e| match e {
            StreamError::Container(e) => format!("{e:?}"),
            StreamError::Io(e) => e
                .into_inner()
                .and_then(|inner| {
                    inner
                        .downcast::<alice_zip::container::ContainerError>()
                        .ok()
                })
                .map_or_else(|| "io".to_string(), |e| format!("{e:?}")),
            other => format!("{other:?}"),
        })?;
        sections.push((tag, critical, payload));
    }
    Ok((sections, r.id()))
}

#[test]
fn the_streamed_reader_agrees_with_the_view_on_every_fixture() {
    let mut inputs: Vec<(&str, Vec<u8>)> = current()
        .into_iter()
        .map(|(n, b)| (n, b.to_vec()))
        .collect();
    inputs.extend([
        ("legacy_v1 (not a container)", LEGACY_V1.to_vec()),
        ("v1_one", V1_ONE.to_vec()),
        ("v1_lids", V1_LIDS.to_vec()),
        ("too short", ONE[..40].to_vec()),
        ("empty input", Vec::new()),
    ]);
    for (name, bytes) in &inputs {
        for known in [&[][..], &[PRODUCT][..]] {
            assert_eq!(
                stream_verdict(bytes, known),
                view_verdict(bytes, known),
                "{name} {known:?}"
            );
        }
    }
}

#[test]
fn the_streamed_reader_agrees_with_the_view_on_every_changed_byte() {
    // every single-byte change of two fixtures: header, table, payloads and trailer
    let mut compared = 0;
    for base in [ONE, UNKNOWN_NONCRITICAL] {
        for at in 0..base.len() {
            let mut v = base.to_vec();
            v[at] ^= 0x01;
            assert_eq!(stream_verdict(&v, &[]), view_verdict(&v, &[]), "byte {at}");
            // the same change with the trailer rewritten: the checks after the trailer
            // (unknown critical sections, the identifier's hash and form) decide
            let body = v.len() - 32;
            let t: [u8; 32] = Sha256::digest(&v[..body]).into();
            v[body..].copy_from_slice(&t);
            assert_eq!(
                stream_verdict(&v, &[]),
                view_verdict(&v, &[]),
                "byte {at}, trailer rewritten"
            );
            compared += 1;
        }
    }
    assert!(compared > 300, "{compared}");
}

#[test]
fn a_changed_payload_with_a_rewritten_trailer_is_caught_when_the_section_is_read() {
    // the trailer matches, the section's own SHA-256 in the table does not
    let mut v = ONE.to_vec();
    let last_payload_byte = v.len() - 32 - 1;
    v[last_payload_byte] ^= 0x01;
    let body = v.len() - 32;
    let t: [u8; 32] = Sha256::digest(&v[..body]).into();
    v[body..].copy_from_slice(&t);
    let mut r = StreamReader::open_with_known(Cursor::new(v.clone()), &[]).unwrap();
    let last = r.len() - 1;
    let mut s = r.section(last).unwrap();
    let mut sink = Vec::new();
    let read = s.read_to_end(&mut sink);
    assert!(
        read.is_err(),
        "the changed payload was read without an error"
    );
    assert!(view_verdict(&v, &[]).is_err());
}

#[test]
fn writing_more_or_fewer_sections_than_declared_is_refused() {
    let declared = [(Tag::RAW, false), (PRODUCT, true)];
    // fewer
    let mut w = StreamWriter::new(Cursor::new(Vec::new()), [0; 32], &declared).unwrap();
    w.write_section(b"one").unwrap();
    assert!(matches!(
        w.finish(),
        Err(StreamError::SectionCount {
            declared: 3,
            written: 2
        })
    ));
    // more
    let mut w = StreamWriter::new(Cursor::new(Vec::new()), [0; 32], &declared).unwrap();
    w.write_section(b"one").unwrap();
    w.write_section(b"two").unwrap();
    assert!(matches!(
        w.begin_section(),
        Err(StreamError::SectionCount {
            declared: 3,
            written: 3
        })
    ));
}

#[test]
fn an_unfinished_writer_or_section_leaves_nothing_that_opens() {
    let declared = [(Tag::RAW, false)];
    let mut w = StreamWriter::new(Cursor::new(Vec::new()), [0; 32], &declared).unwrap();
    w.write_section(b"payload").unwrap();
    let partial = w.into_unfinished().into_inner();
    assert!(ContainerView::open(&partial).is_err());
    assert!(StreamReader::open_with_known(Cursor::new(partial), &[]).is_err());

    // a section writer dropped without finish poisons the container
    let mut w = StreamWriter::new(Cursor::new(Vec::new()), [0; 32], &declared).unwrap();
    {
        let mut s = w.begin_section().unwrap();
        s.write_all(b"half").unwrap();
    }
    assert!(matches!(w.finish(), Err(StreamError::Unfinished)));
}

/// A shared buffer the test can change while the writer owns it
#[derive(Clone)]
struct Shared(Rc<RefCell<Cursor<Vec<u8>>>>);
impl Read for Shared {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().read(b)
    }
}
impl Write for Shared {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Seek for Shared {
    fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
        self.0.borrow_mut().seek(p)
    }
}

#[test]
fn a_payload_changed_on_disk_before_finish_is_not_trailered() {
    let shared = Shared(Rc::new(RefCell::new(Cursor::new(Vec::new()))));
    let mut w = StreamWriter::new(shared.clone(), [0; 32], &[(Tag::RAW, false)]).unwrap();
    w.write_section(b"payload written").unwrap();
    // change a payload byte behind the writer's back
    {
        let mut c = shared.0.borrow_mut();
        let v = c.get_mut();
        let at = v.len() - 1;
        v[at] ^= 0x01;
    }
    let r = w.finish();
    assert!(
        matches!(
            r,
            Err(StreamError::Container(
                alice_zip::container::ContainerError::SectionHash { index: 1 }
            ))
        ),
        "{:?}",
        r.err()
    );
}

/// A section whose hash does not match stays failed: a read after the failing
/// read and `finish` return the same error (never the end of the section or `Ok`)
#[test]
fn a_failed_section_check_is_returned_again_by_every_later_read_and_by_finish() {
    let mut v = ONE.to_vec();
    let last_payload_byte = v.len() - 32 - 1;
    v[last_payload_byte] ^= 0x01;
    let body = v.len() - 32;
    let t: [u8; 32] = Sha256::digest(&v[..body]).into();
    v[body..].copy_from_slice(&t);
    let mut r = StreamReader::open_with_known(Cursor::new(v), &[]).unwrap();
    let last = r.len() - 1;
    let mut s = r.section(last).unwrap();
    let mut sink = Vec::new();
    assert!(
        s.read_to_end(&mut sink).is_err(),
        "the changed payload read without an error"
    );
    let mut more = [0u8; 16];
    let again = s.read(&mut more);
    assert!(
        again.is_err(),
        "a read after the failure returned {again:?}"
    );
    let again = s.read(&mut more);
    assert!(
        again.is_err(),
        "a second read after the failure returned {again:?}"
    );
    let fin = s.finish();
    assert!(
        matches!(
            fin,
            Err(StreamError::Container(
                alice_zip::container::ContainerError::SectionHash { .. }
            ))
        ),
        "finish after the failure returned {fin:?}"
    );
}

/// A target that already holds bytes past the end of the container: `finish`
/// refuses (the stale tail would stay and the file would not open) instead of
/// reporting success
#[test]
fn a_target_with_bytes_past_the_container_is_refused_by_finish() {
    let old = vec![0xEE_u8; 10_000];
    let mut w = StreamWriter::new(Cursor::new(old), [0; 32], &[(Tag::RAW, false)]).unwrap();
    w.write_section(b"short").unwrap();
    let r = w.finish();
    assert!(
        matches!(r, Err(StreamError::TrailingBytes { .. })),
        "{:?}",
        r.map(|c| c.into_inner().len())
    );
    // a target exactly as long as the container (written over in place) is fine
    let mut c = Container::new([0; 32]);
    c.push(Tag::RAW, false, b"short".to_vec());
    let len = c.to_bytes().len();
    let mut w = StreamWriter::new(
        Cursor::new(vec![0xEE_u8; len]),
        [0; 32],
        &[(Tag::RAW, false)],
    )
    .unwrap();
    w.write_section(b"short").unwrap();
    assert_eq!(w.finish().unwrap().into_inner(), c.to_bytes());
}
