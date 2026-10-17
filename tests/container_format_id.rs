//! Oracles for the container format identifier (`Tag::FMID`)
//!
//! The identifier is recomputed here from the format's definition with
//! `sha2`, and compared with the value the independent reference writer
//! `tests/data/container/container_ref.py` records in `golden.json`.
//! `tests/test_container_format_id.py` reads the fixtures with its own parser
//! and checks the same section; `compat/v0_10_reader` checks that the
//! released 0.10 reader refuses these files.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_zip::container::{
    read_legacy_alice_zip, Container, ContainerError, ContainerView, FormatStatus, Section, Tag,
    ENTRY_LEN, FORMAT_DOMAIN, FORMAT_ID, HEADER_LEN, ID_DOMAIN, MAGIC, MAJOR, TRAILER_LEN,
};
use sha2::{Digest, Sha256};

const SEM: [u8; 32] = [0x11; 32];
const GOLDEN: &str = include_str!("data/container/golden.json");

/// Every fixture the reference writer writes with the identifier.
const WITH_FMID: [(&str, &[u8]); 6] = [
    ("empty", include_bytes!("data/container/empty.bin")),
    ("one", include_bytes!("data/container/one.bin")),
    ("dup_tag", include_bytes!("data/container/dup_tag.bin")),
    ("zero_len", include_bytes!("data/container/zero_len.bin")),
    (
        "unknown_noncritical",
        include_bytes!("data/container/unknown_noncritical.bin"),
    ),
    ("bundle", include_bytes!("data/container/bundle.bin")),
];
/// Containers as 0.10 and earlier wrote them.
const V1: [(&str, &[u8]); 2] = [
    ("v1_one", include_bytes!("data/container/v1_one.bin")),
    ("v1_lids", include_bytes!("data/container/v1_lids.bin")),
];
const ONE: &[u8] = include_bytes!("data/container/one.bin");
const LEGACY_V2: &[u8] = include_bytes!("data/container/legacy_v2.alice");

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

fn retrail(mut v: Vec<u8>) -> Vec<u8> {
    let n = v.len() - TRAILER_LEN;
    let t = sha(&v[..n]);
    v[n..].copy_from_slice(&t);
    v
}

/// A container with exactly `sections`, no FMID added.
fn raw(sections: Vec<(Tag, bool, Vec<u8>)>) -> Container {
    Container {
        minor: 0,
        semantics_id: SEM,
        sections: sections
            .into_iter()
            .map(|(tag, critical, payload)| Section {
                tag,
                critical,
                payload,
            })
            .collect(),
    }
}

fn fmid() -> (Tag, bool, Vec<u8>) {
    (Tag::FMID, true, FORMAT_ID.to_vec())
}

#[test]
fn the_identifier_is_the_hash_of_the_format_definition() {
    // oracle: the definition in the module documentation, recomputed here
    let mut tags = Vec::new();
    for t in Tag::KNOWN {
        tags.extend_from_slice(&t.0);
    }
    assert_eq!(Tag::KNOWN.len(), 13);
    assert_eq!(Tag::KNOWN[12], Tag::FMID);
    let items: [&[u8]; 8] = [
        FORMAT_DOMAIN,
        &MAGIC,
        &MAJOR.to_le_bytes(),
        &(HEADER_LEN as u64).to_le_bytes(),
        &(ENTRY_LEN as u64).to_le_bytes(),
        &(TRAILER_LEN as u64).to_le_bytes(),
        ID_DOMAIN,
        &tags,
    ];
    let mut h = Sha256::new();
    for x in items {
        h.update((x.len() as u64).to_be_bytes());
        h.update(x);
    }
    let want: [u8; 32] = h.finalize().into();
    assert_eq!(FORMAT_ID, want);
}

#[test]
fn the_reference_writer_records_the_same_identifier() {
    let line = format!("\"format_id\": \"{}\"", hex(&FORMAT_ID));
    assert!(GOLDEN.contains(&line), "golden.json lacks {line}");
}

#[test]
fn a_new_container_starts_with_the_identifier() {
    let mut c = Container::new(SEM);
    assert_eq!(
        c.sections,
        vec![Section {
            tag: Tag::FMID,
            critical: true,
            payload: FORMAT_ID.to_vec()
        }]
    );
    c.push(Tag::RAW, true, b"x".to_vec());
    assert_eq!(c.format_status(), Ok(FormatStatus::Verified));
    let bytes = c.to_bytes();
    assert_eq!(&bytes[HEADER_LEN..HEADER_LEN + 4], b"FMID");
    let view = ContainerView::open(&bytes).unwrap();
    assert_eq!(view.format_status(), FormatStatus::Verified);
    assert_eq!(view.section(0).unwrap(), FORMAT_ID);
}

#[test]
fn every_reference_fixture_carries_the_identifier_first() {
    for (name, f) in WITH_FMID {
        let view = ContainerView::open(f).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(view.format_status(), FormatStatus::Verified, "{name}");
        assert_eq!(view.tag(0), Tag::FMID, "{name}");
        assert!(view.critical(0), "{name}");
        assert_eq!(view.section(0).unwrap(), FORMAT_ID, "{name}");
        let c = Container::from_bytes_strict(f).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(c.format_status(), Ok(FormatStatus::Verified), "{name}");
        assert!(ContainerView::open_strict(f).is_ok(), "{name}");
    }
}

#[test]
fn a_container_without_the_identifier_reads_as_legacy_and_a_strict_read_refuses_it() {
    for (name, f) in V1 {
        let view = ContainerView::open(f).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(view.format_status(), FormatStatus::Legacy, "{name}");
        assert!((0..view.len()).all(|i| view.tag(i) != Tag::FMID), "{name}");
        let c = Container::from_bytes(f).unwrap();
        assert_eq!(c.format_status(), Ok(FormatStatus::Legacy), "{name}");
        assert_eq!(c.to_bytes(), f, "{name}: written back unchanged");
        assert_eq!(
            Container::from_bytes_strict(f),
            Err(ContainerError::LegacyFormat),
            "{name}"
        );
        assert_eq!(
            ContainerView::open_strict(f).err(),
            Some(ContainerError::LegacyFormat),
            "{name}"
        );
    }
    // the law identifier table of the old file still names its section 0
    let view = ContainerView::open(V1[1].1).unwrap();
    let rows = view.law_ids().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].section, 0);
    assert_eq!(view.tag(0), Tag::LAW);
}

#[test]
fn an_earlier_alice_zip_file_converts_to_a_legacy_container() {
    let c = read_legacy_alice_zip(LEGACY_V2).unwrap();
    assert_eq!(c.format_status(), Ok(FormatStatus::Legacy));
    assert_eq!(c.sections[0].tag, Tag::PROV);
}

#[test]
fn another_identifier_is_an_unknown_format() {
    // flip one bit of the FMID payload and repair its table hash and the trailer
    let mut v = ONE.to_vec();
    let payload = HEADER_LEN + 2 * ENTRY_LEN;
    v[payload] ^= 0x80;
    let digest = sha(&v[payload..payload + 32]);
    v[HEADER_LEN + 24..HEADER_LEN + 56].copy_from_slice(&digest);
    let v = retrail(v);
    let mut found = FORMAT_ID;
    found[0] ^= 0x80;
    let want = ContainerError::UnknownFormat { found };
    assert_eq!(ContainerView::open(&v).err(), Some(want));
    assert_eq!(Container::from_bytes(&v), Err(want));
    assert_eq!(Container::from_bytes_strict(&v), Err(want));
    let c = raw(vec![(Tag::FMID, true, found.to_vec())]);
    assert_eq!(c.format_status(), Err(want));
}

#[test]
fn an_identifier_payload_that_differs_from_its_table_hash_is_refused() {
    let mut v = ONE.to_vec();
    v[HEADER_LEN + 2 * ENTRY_LEN] ^= 1;
    let v = retrail(v);
    assert_eq!(
        ContainerView::open(&v).err(),
        Some(ContainerError::SectionHash { index: 0 })
    );
}

#[test]
fn an_identifier_section_that_is_not_the_single_critical_32_byte_first_section_is_malformed() {
    let malformed = ContainerError::Malformed { tag: Tag::FMID };
    let raw_section = (Tag::RAW, true, b"r".to_vec());
    let cases: Vec<(&str, Container)> = vec![
        ("not first", raw(vec![raw_section.clone(), fmid()])),
        (
            "not critical",
            raw(vec![(Tag::FMID, false, FORMAT_ID.to_vec())]),
        ),
        (
            "31 bytes",
            raw(vec![(Tag::FMID, true, FORMAT_ID[..31].to_vec())]),
        ),
        ("33 bytes", {
            let mut p = FORMAT_ID.to_vec();
            p.push(0);
            raw(vec![(Tag::FMID, true, p)])
        }),
        ("empty", raw(vec![(Tag::FMID, true, Vec::new())])),
        ("twice", raw(vec![fmid(), raw_section.clone(), fmid()])),
        ("twice in a row", raw(vec![fmid(), fmid()])),
    ];
    for (name, c) in cases {
        assert_eq!(c.format_status(), Err(malformed), "{name}");
        let bytes = c.to_bytes();
        assert_eq!(ContainerView::open(&bytes).err(), Some(malformed), "{name}");
        assert_eq!(Container::from_bytes(&bytes), Err(malformed), "{name}");
    }
}

#[test]
fn errors_display_without_panicking() {
    for e in [
        ContainerError::UnknownFormat { found: [0; 32] },
        ContainerError::LegacyFormat,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
