//! Oracles for reading a container whose format adds its own required sections
//!
//! A format built on the container may define tags of its own and mark them
//! critical: a reader that does not know such a tag must refuse the file
//! rather than skip a section the format requires. The reader declares the
//! tags it knows besides the built-in ones; a critical section whose tag is
//! neither built in nor declared is refused, a non-critical one is kept.
//!
//! The inputs are the reference fixture `unknown_noncritical.bin` (written by
//! `tests/data/container/container_ref.py`, which does not call this crate)
//! with the critical bit of its `ZZZZ` entry set or not.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_zip::container::{
    Container, ContainerError, ContainerView, FormatStatus, Tag, ENTRY_LEN, HEADER_LEN,
};
use sha2::{Digest, Sha256};

const UNKNOWN_NONCRITICAL: &[u8] = include_bytes!("data/container/unknown_noncritical.bin");
const ONE: &[u8] = include_bytes!("data/container/one.bin");
const DUP_TAG: &[u8] = include_bytes!("data/container/dup_tag.bin");
const ZERO_LEN: &[u8] = include_bytes!("data/container/zero_len.bin");
const BUNDLE: &[u8] = include_bytes!("data/container/bundle.bin");

const PRODUCT: Tag = Tag(*b"ZZZZ");
const OTHER: Tag = Tag(*b"YYYY");

/// Index of the `ZZZZ` entry in the fixture (after `FMID` and `RAW_`).
const PRODUCT_INDEX: usize = 2;

fn retrail(mut v: Vec<u8>) -> Vec<u8> {
    let body = v.len() - 32;
    let t: [u8; 32] = Sha256::digest(&v[..body]).into();
    v[body..].copy_from_slice(&t);
    v
}

/// The fixture with the `ZZZZ` section marked critical: a file of a format
/// that requires its own section.
fn product_file() -> Vec<u8> {
    let at = HEADER_LEN + PRODUCT_INDEX * ENTRY_LEN + 4;
    let mut v = UNKNOWN_NONCRITICAL.to_vec();
    assert_eq!(v[at] & 1, 0, "the fixture's ZZZZ entry is not critical");
    v[at] |= 1;
    retrail(v)
}

#[test]
fn a_reader_that_does_not_declare_the_tag_refuses_the_required_section() {
    let v = product_file();
    let refused = Some(ContainerError::UnknownCritical { tag: PRODUCT });
    assert_eq!(ContainerView::open(&v).err(), refused);
    assert_eq!(ContainerView::open_with_known(&v, &[]).err(), refused);
    assert_eq!(ContainerView::open_with_known(&v, &[OTHER]).err(), refused);
    assert_eq!(
        Container::from_bytes_with_known(&v, &[OTHER]).err(),
        refused
    );
    assert_eq!(
        ContainerView::open_strict_with_known(&v, &[OTHER]).err(),
        refused
    );
    assert_eq!(
        Container::from_bytes_strict_with_known(&v, &[OTHER]).err(),
        refused
    );
}

#[test]
fn a_reader_that_declares_the_tag_reads_the_required_section() {
    let v = product_file();
    let view = ContainerView::open_with_known(&v, &[OTHER, PRODUCT]).unwrap();
    assert_eq!(view.tag(PRODUCT_INDEX), PRODUCT);
    assert!(view.critical(PRODUCT_INDEX));
    assert_eq!(view.format_status(), FormatStatus::Verified);
    let payload = view.section(PRODUCT_INDEX).unwrap();
    let want: [u8; 32] = Sha256::digest(payload).into();
    assert_eq!(view.sha256(PRODUCT_INDEX), want);

    let strict = ContainerView::open_strict_with_known(&v, &[PRODUCT]).unwrap();
    assert_eq!(strict.id(), view.id());

    let c = Container::from_bytes_with_known(&v, &[PRODUCT]).unwrap();
    assert_eq!(c.to_bytes(), v, "reading then writing reproduces the bytes");
    assert_eq!(c.id(), view.id());
    let s = Container::from_bytes_strict_with_known(&v, &[PRODUCT]).unwrap();
    assert_eq!(s, c);
}

#[test]
fn declaring_tags_does_not_change_how_a_non_critical_section_is_read() {
    let plain = ContainerView::open(UNKNOWN_NONCRITICAL).unwrap();
    for known in [&[][..], &[PRODUCT][..], &[OTHER][..]] {
        let view = ContainerView::open_with_known(UNKNOWN_NONCRITICAL, known).unwrap();
        assert_eq!(view.id(), plain.id());
        assert_eq!(view.len(), plain.len());
        assert!(!view.critical(PRODUCT_INDEX));
        let c = Container::from_bytes_with_known(UNKNOWN_NONCRITICAL, known).unwrap();
        assert_eq!(c.to_bytes(), UNKNOWN_NONCRITICAL);
    }
}

/// A container without the [`Tag::FMID`] section (as written by alice-zip
/// 0.10 and earlier) that holds a critical section of the product's own tag.
fn legacy_product_file() -> Vec<u8> {
    let mut c = Container::from_bytes(UNKNOWN_NONCRITICAL).unwrap();
    assert_eq!(c.sections[0].tag, Tag::FMID);
    c.sections.remove(0);
    c.sections[PRODUCT_INDEX - 1].critical = true;
    c.to_bytes()
}

#[test]
fn declaring_tags_does_not_accept_a_file_without_the_format_identifier() {
    // the strict readers still refuse a container written before the
    // identifier existed, whatever tags the reader declares
    let v = legacy_product_file();
    let view = ContainerView::open_with_known(&v, &[PRODUCT]).unwrap();
    assert_eq!(view.format_status(), FormatStatus::Legacy);
    assert_eq!(
        ContainerView::open_strict_with_known(&v, &[PRODUCT]).err(),
        Some(ContainerError::LegacyFormat)
    );
    assert_eq!(
        Container::from_bytes_strict_with_known(&v, &[PRODUCT]).err(),
        Some(ContainerError::LegacyFormat)
    );
}

#[test]
fn declaring_the_format_identifier_tag_does_not_relax_its_rules() {
    // a second FMID section is malformed however the reader declares tags
    let mut c = Container::from_bytes(UNKNOWN_NONCRITICAL).unwrap();
    let fmid = c.sections[0].payload.clone();
    c.push(Tag::FMID, true, fmid);
    let v = c.to_bytes();
    let plain = ContainerView::open(&v).err();
    assert!(
        matches!(plain, Some(ContainerError::Malformed { tag }) if tag == Tag::FMID),
        "{plain:?}"
    );
    assert_eq!(
        ContainerView::open_with_known(&v, &[Tag::FMID]).err(),
        plain
    );
}

#[test]
fn repeated_and_built_in_tags_in_the_declaration_change_nothing() {
    let v = product_file();
    let known = [PRODUCT, PRODUCT, Tag::RAW, Tag::FMID];
    let a = ContainerView::open_with_known(&v, &known).unwrap();
    let b = ContainerView::open_with_known(&v, &[PRODUCT]).unwrap();
    assert_eq!(a.id(), b.id());
}

/// The tags the format defines, written out from the layout in the module
/// documentation (not read from the crate).
const DEFINED: [&[u8; 4]; 13] = [
    b"RAW_", b"RESD", b"LAW_", b"VRDT", b"SHAP", b"PHYS", b"ASET", b"SIMS", b"PROV", b"ORCL",
    b"SREF", b"LIDS", b"FMID",
];

/// Reference for the refusal: read the section table from the bytes and
/// return the tag of the first entry that is critical and neither defined nor
/// declared.
fn reference_refusal(bytes: &[u8], known: &[Tag]) -> Option<Tag> {
    let count = usize::try_from(u64::from_le_bytes(bytes[48..56].try_into().unwrap())).unwrap();
    (0..count).find_map(|i| {
        let at = HEADER_LEN + i * ENTRY_LEN;
        let tag = Tag(bytes[at..at + 4].try_into().unwrap());
        let critical = u16::from_le_bytes(bytes[at + 4..at + 6].try_into().unwrap()) & 1 == 1;
        let defined = DEFINED.iter().any(|d| **d == tag.0);
        (critical && !defined && !known.contains(&tag)).then_some(tag)
    })
}

/// Every reference fixture, and each of them with one more entry marked critical.
fn corpus() -> Vec<Vec<u8>> {
    let bases: Vec<Vec<u8>> = [ONE, DUP_TAG, ZERO_LEN, BUNDLE, UNKNOWN_NONCRITICAL]
        .iter()
        .map(|b| b.to_vec())
        .collect();
    let mut all = bases.clone();
    for base in &bases {
        let count = usize::try_from(u64::from_le_bytes(base[48..56].try_into().unwrap())).unwrap();
        all.extend((0..count).map(|i| {
            let mut v = base.clone();
            v[HEADER_LEN + i * ENTRY_LEN + 4] |= 1;
            retrail(v)
        }));
    }
    all
}

#[test]
fn the_refusal_matches_a_reference_over_every_entry_and_declaration() {
    let declarations: [&[Tag]; 5] = [&[], &[PRODUCT], &[OTHER], &[OTHER, PRODUCT], &[Tag::RAW]];
    let mut refused = 0;
    let mut read = 0;
    for v in &corpus() {
        for known in declarations {
            let got = ContainerView::open_with_known(v, known).err();
            match reference_refusal(v, known) {
                Some(tag) => {
                    refused += 1;
                    assert_eq!(got, Some(ContainerError::UnknownCritical { tag }));
                    assert_eq!(
                        Container::from_bytes_with_known(v, known).err(),
                        Some(ContainerError::UnknownCritical { tag })
                    );
                }
                None => {
                    read += 1;
                    assert!(
                        !matches!(got, Some(ContainerError::UnknownCritical { .. })),
                        "refused although every critical tag is known: {got:?}"
                    );
                }
            }
        }
    }
    // both outcomes occur, so neither branch is vacuous
    assert!(refused > 0 && read > 0, "refused {refused}, read {read}");
}
