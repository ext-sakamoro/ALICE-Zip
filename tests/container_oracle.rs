//! Oracles for `container`
//!
//! Expected bytes and identifiers come from the independent reference writer
//! `tests/data/container/container_ref.py` (Python `struct` + `hashlib`, it
//! does not call this crate). The fixtures and the constants below are its
//! output; regenerate both together.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_zip::container::{
    detect, read_any, read_legacy_alice_zip, Container, ContainerError, ContainerView, Format,
    Section, Tag, ENTRY_LEN, HEADER_LEN, TRAILER_LEN,
};
use alice_zip::law::{Provenance, SignalLaw, SEMANTICS_ID};
use sha2::{Digest, Sha256};

const SEM: [u8; 32] = [0x11; 32];

const EMPTY: &[u8] = include_bytes!("data/container/empty.bin");
const ONE: &[u8] = include_bytes!("data/container/one.bin");
const DUP_TAG: &[u8] = include_bytes!("data/container/dup_tag.bin");
const ZERO_LEN: &[u8] = include_bytes!("data/container/zero_len.bin");
const UNKNOWN_NONCRITICAL: &[u8] = include_bytes!("data/container/unknown_noncritical.bin");
const BUNDLE: &[u8] = include_bytes!("data/container/bundle.bin");
const LEGACY_V1: &[u8] = include_bytes!("data/container/legacy_v1.alice");
const LEGACY_V2: &[u8] = include_bytes!("data/container/legacy_v2.alice");
const LEGACY_MAJOR2: &[u8] = include_bytes!("data/container/legacy_major2.alice");
const LEGACY_UNKNOWN_PAYLOAD: &[u8] = include_bytes!("data/container/legacy_unknown_payload.alice");

// golden.json (reference writer)
const ID_EMPTY: &str = "f9742fba5c7fc270267322cca82ad98ab7856fd489e1776e8f63e44e4134c8f5";
const ID_ONE: &str = "779be9720071d3f9d150e90fef514d939687f418edad01e06e8c68a4b0838b38";
const ID_DUP_TAG: &str = "b375a0cd50313ac42638c474d0df58a86ef4ea1c88d1e8fed249a3d9c62424d0";
const ID_ZERO_LEN: &str = "fb5ebc0dbc1f1913c9103f694886ed0c733ed1c1791921b20456c2cde265809e";
const ID_UNKNOWN_NONCRITICAL: &str =
    "2500f98fba0b7763abd8a329605fecaf353e54d6dc490ea95afe3a1b382f6693";
const ID_BUNDLE: &str = "875e9529be98c8a2566b08c38a551b76704c38edc7b791d5f5908cfd4dbb6162";
const ID_BIG: &str = "c3f550d152c7e161ffe4330216bfedb3a6f5fb8365e3d2d7a24e1b8995efa7ec";
const FILE_SHA_BIG: &str = "a50c46a05e553bc198a4bb82f434d9ca6719744b623ba87c520f1ed78a72656d";
const LEN_BIG: usize = 1_048_720;

const BUILDER: &[u8] = b"alice-sdf/compiled-field/v1";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

/// Rewrites the trailer so that a deliberate change elsewhere is not caught
/// by it (to reach the checks behind the trailer).
fn retrail(mut v: Vec<u8>) -> Vec<u8> {
    let n = v.len() - TRAILER_LEN;
    let t = sha(&v[..n]);
    v[n..].copy_from_slice(&t);
    v
}

fn big_payload() -> Vec<u8> {
    (0..1usize << 20)
        .map(|i| ((i * 31 + 7) % 256) as u8)
        .collect()
}

fn bundle_parts() -> (Vec<u8>, Vec<u8>) {
    (b"shape-bytes-0123456789".to_vec(), b"world-state".to_vec())
}

fn sref_payload(rows: &[([u8; 32], [u8; 4], &[u8])]) -> Vec<u8> {
    let mut out = (rows.len() as u64).to_le_bytes().to_vec();
    for (target, kind, builder) in rows {
        out.extend_from_slice(target);
        out.extend_from_slice(kind);
        out.extend_from_slice(&(builder.len() as u16).to_le_bytes());
        out.extend_from_slice(builder);
    }
    out
}

fn lids_payload(rows: &[(u64, [u8; 32], [u8; 32])]) -> Vec<u8> {
    let mut out = (rows.len() as u64).to_le_bytes().to_vec();
    for (index, law_id, sem) in rows {
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(law_id);
        out.extend_from_slice(sem);
    }
    out
}

fn build(sections: &[(Tag, bool, &[u8])]) -> Container {
    let mut c = Container::new(SEM);
    for (tag, critical, payload) in sections {
        c.push(*tag, *critical, payload.to_vec());
    }
    c
}

fn bundle() -> Container {
    let (shape, world) = bundle_parts();
    let s = sha(&shape);
    build(&[
        (Tag::SHAP, true, &shape),
        (Tag::PHYS, true, &world),
        (
            Tag::SREF,
            true,
            &sref_payload(&[(s, *b"SHAP", BUILDER), (s, *b"SHAP", BUILDER)]),
        ),
        (Tag::LIDS, true, &lids_payload(&[(1, [0xAB; 32], SEM)])),
    ])
}

fn scenes() -> Vec<(&'static str, Container, &'static [u8], &'static str)> {
    vec![
        ("empty", build(&[]), EMPTY, ID_EMPTY),
        ("one", build(&[(Tag::RAW, true, b"hello")]), ONE, ID_ONE),
        (
            "dup_tag",
            build(&[
                (Tag::RAW, true, b"a"),
                (Tag::RAW, true, b"b"),
                (Tag::PROV, false, b"note"),
            ]),
            DUP_TAG,
            ID_DUP_TAG,
        ),
        (
            "zero_len",
            build(&[(Tag::RESD, true, b"")]),
            ZERO_LEN,
            ID_ZERO_LEN,
        ),
        (
            "unknown_noncritical",
            build(&[(Tag::RAW, true, b"x"), (Tag(*b"ZZZZ"), false, b"keep me")]),
            UNKNOWN_NONCRITICAL,
            ID_UNKNOWN_NONCRITICAL,
        ),
        ("bundle", bundle(), BUNDLE, ID_BUNDLE),
    ]
}

// ---------------------------------------------------------------- (1) round trip

#[test]
fn writer_matches_the_reference_bytes_and_identifier() {
    for (name, c, fixture, id) in scenes() {
        assert_eq!(
            c.to_bytes(),
            fixture,
            "{name}: bytes differ from the reference writer"
        );
        assert_eq!(hex(&c.id()), id, "{name}: identifier");
    }
}

#[test]
fn reading_then_writing_reproduces_the_bytes() {
    for (name, c, fixture, id) in scenes() {
        let read = Container::from_bytes(fixture).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(read, c, "{name}: sections");
        assert_eq!(read.to_bytes(), fixture, "{name}: write after read");
        let view = ContainerView::open(fixture).unwrap();
        assert_eq!(hex(&view.id()), id, "{name}: view identifier");
        assert_eq!(view.len(), c.sections.len());
        for (i, s) in c.sections.iter().enumerate() {
            assert_eq!(view.tag(i), s.tag);
            assert_eq!(view.critical(i), s.critical);
            assert_eq!(view.section(i).unwrap(), s.payload.as_slice());
            assert_eq!(view.sha256(i), sha(&s.payload));
        }
    }
}

#[test]
fn a_one_mebibyte_payload_has_the_reference_identifier() {
    let c = build(&[(Tag::RAW, true, &big_payload())]);
    let bytes = c.to_bytes();
    assert_eq!(bytes.len(), LEN_BIG);
    assert_eq!(hex(&sha(&bytes)), FILE_SHA_BIG);
    assert_eq!(hex(&c.id()), ID_BIG);
    assert_eq!(Container::from_bytes(&bytes).unwrap(), c);
}

#[test]
fn push_returns_the_payload_sha256_and_keeps_order() {
    let mut c = Container::new(SEM);
    assert_eq!(c.push(Tag::RAW, true, b"b".to_vec()), sha(b"b"));
    assert_eq!(c.push(Tag::RAW, true, b"a".to_vec()), sha(b"a"));
    assert_eq!(c.sections[0].payload, b"b");
    assert_eq!(c.minor, alice_zip::container::MINOR);
}

#[test]
fn the_trailer_is_not_part_of_the_identifier() {
    // id over header + table only: equal for two containers whose bytes
    // differ only by the trailer is impossible to build, so check the
    // definition directly against the reference formula
    let c = bundle();
    let bytes = c.to_bytes();
    let n = HEADER_LEN + ENTRY_LEN * c.sections.len();
    let domain = b"alice/container/v1";
    let mut h = Sha256::new();
    h.update((domain.len() as u64).to_be_bytes());
    h.update(domain);
    h.update((HEADER_LEN as u64).to_be_bytes());
    h.update(&bytes[..HEADER_LEN]);
    h.update(((n - HEADER_LEN) as u64).to_be_bytes());
    h.update(&bytes[HEADER_LEN..n]);
    let expect: [u8; 32] = h.finalize().into();
    assert_eq!(c.id(), expect);
}

// ---------------------------------------------------------------- (2) tampering

/// Which check a flipped bit at `pos` of BUNDLE must be refused by.
fn expected_for(pos: usize, bit: u8, sections: usize) -> &'static str {
    let table_end = HEADER_LEN + ENTRY_LEN * sections;
    match pos {
        0..=7 => "BadMagic",
        8..=9 => "UnsupportedMajor",
        10..=11 => "Trailer",
        12..=15 => "Reserved",
        16..=47 => "Trailer",
        48..=55 => "Layout",
        p if p < table_end => match (p - HEADER_LEN) % ENTRY_LEN {
            0..=3 => "Trailer",         // tag
            4 if bit == 0 => "Trailer", // critical bit
            4..=5 => "Reserved",        // other flag bits
            6..=7 => "Reserved",        // entry reserved
            8..=23 => "Layout",         // offset, length
            _ => "Trailer",             // sha256 column
        },
        _ => "Trailer", // payloads and the trailer itself
    }
}

fn kind(e: &ContainerError) -> &'static str {
    match e {
        ContainerError::TooShort => "TooShort",
        ContainerError::BadMagic => "BadMagic",
        ContainerError::UnsupportedMajor { .. } => "UnsupportedMajor",
        ContainerError::Reserved => "Reserved",
        ContainerError::Layout => "Layout",
        ContainerError::Trailer => "Trailer",
        ContainerError::SectionHash { .. } => "SectionHash",
        ContainerError::UnknownCritical { .. } => "UnknownCritical",
        ContainerError::Malformed { .. } => "Malformed",
        ContainerError::DanglingSection { .. } => "DanglingSection",
        ContainerError::UnknownBuilder { .. } => "UnknownBuilder",
        ContainerError::SemanticsMismatch { .. } => "SemanticsMismatch",
        ContainerError::LawIdMismatch { .. } => "LawIdMismatch",
        ContainerError::LegacyVersion { .. } => "LegacyVersion",
        ContainerError::LegacyField { .. } => "LegacyField",
        _ => "other",
    }
}

#[test]
fn every_single_bit_flip_is_refused_by_the_expected_check() {
    let sections = 4;
    let mut refused = 0usize;
    let mut wrong = Vec::new();
    for pos in 0..BUNDLE.len() {
        for bit in 0..8u8 {
            let mut v = BUNDLE.to_vec();
            v[pos] ^= 1 << bit;
            match Container::from_bytes(&v) {
                Ok(_) => wrong.push(format!("{pos}.{bit}: accepted")),
                Err(e) => {
                    refused += 1;
                    let want = expected_for(pos, bit, sections);
                    if kind(&e) != want {
                        wrong.push(format!("{pos}.{bit}: {e:?}, want {want}"));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} positions:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert_eq!(refused, BUNDLE.len() * 8, "every flip compared");
}

#[test]
fn a_payload_changed_with_a_matching_trailer_fails_its_section_hash() {
    let mut v = ONE.to_vec();
    let at = HEADER_LEN + ENTRY_LEN; // first payload byte
    v[at] ^= 0x01;
    let v = retrail(v);
    let view = ContainerView::open(&v).expect("header, table and trailer are intact");
    assert_eq!(
        view.section(0),
        Err(ContainerError::SectionHash { index: 0 })
    );
    assert_eq!(
        Container::from_bytes(&v),
        Err(ContainerError::SectionHash { index: 0 })
    );
}

#[test]
fn only_the_requested_section_is_checked() {
    let mut v = DUP_TAG.to_vec();
    // second payload ("b") sits right after the first ("a")
    let at = HEADER_LEN + 3 * ENTRY_LEN + 1;
    assert_eq!(v[at], b'b');
    v[at] = b'c';
    let v = retrail(v);
    let view = ContainerView::open(&v).unwrap();
    assert_eq!(view.section(0).unwrap(), b"a");
    assert_eq!(view.section(2).unwrap(), b"note");
    assert_eq!(
        view.section(1),
        Err(ContainerError::SectionHash { index: 1 })
    );
}

// ---------------------------------------------------------------- (3) versions, unknown sections

fn with_u16(fixture: &[u8], at: usize, value: u16) -> Vec<u8> {
    let mut v = fixture.to_vec();
    v[at..at + 2].copy_from_slice(&value.to_le_bytes());
    retrail(v)
}

#[test]
fn major_versions_other_than_one_are_refused() {
    for major in [0u16, 2, 0xFFFF] {
        let v = with_u16(ONE, 8, major);
        assert_eq!(
            Container::from_bytes(&v),
            Err(ContainerError::UnsupportedMajor { major }),
            "major {major}"
        );
    }
}

#[test]
fn any_minor_version_is_read() {
    let v = with_u16(ONE, 10, 7);
    let c = Container::from_bytes(&v).unwrap();
    assert_eq!(c.minor, 7);
    assert_eq!(c.to_bytes(), v, "the minor version is written back");
}

#[test]
fn an_unknown_critical_section_is_refused() {
    // flags of the second entry: critical bit on
    let at = HEADER_LEN + ENTRY_LEN + 4;
    let mut v = UNKNOWN_NONCRITICAL.to_vec();
    v[at] |= 1;
    let v = retrail(v);
    assert_eq!(
        ContainerView::open(&v).err(),
        Some(ContainerError::UnknownCritical { tag: Tag(*b"ZZZZ") })
    );
    assert_eq!(
        Container::from_bytes(&v),
        Err(ContainerError::UnknownCritical { tag: Tag(*b"ZZZZ") })
    );
}

#[test]
fn an_unknown_non_critical_section_is_kept_and_written_back() {
    let c = Container::from_bytes(UNKNOWN_NONCRITICAL).unwrap();
    assert_eq!(
        c.sections[1],
        Section {
            tag: Tag(*b"ZZZZ"),
            critical: false,
            payload: b"keep me".to_vec()
        }
    );
    assert_eq!(c.to_bytes(), UNKNOWN_NONCRITICAL);
}

#[test]
fn undefined_flag_bits_are_refused() {
    let at = HEADER_LEN + 4;
    for bit in 1..16u16 {
        let v = with_u16(ONE, at, 1 | (1 << bit));
        assert_eq!(
            Container::from_bytes(&v),
            Err(ContainerError::Reserved),
            "bit {bit}"
        );
    }
}

// ---------------------------------------------------------------- references and law ids

#[test]
fn references_resolve_to_the_shared_shape_section() {
    let view = ContainerView::open(BUNDLE).unwrap();
    let refs = view.references().unwrap();
    assert_eq!(refs.len(), 2);
    let (shape, _) = bundle_parts();
    for r in &refs {
        assert_eq!(r.target, sha(&shape));
        assert_eq!(r.kind, Tag::SHAP);
        assert_eq!(r.builder, BUILDER);
        assert_eq!(r.section, 0, "both rows name the one shape section");
    }
    assert_eq!(view.references_with_builders(&[BUILDER]).unwrap(), refs);
}

#[test]
fn a_reference_to_a_missing_section_is_refused() {
    let missing = [0x5A; 32];
    let c = build(&[
        (Tag::PHYS, true, b"w"),
        (
            Tag::SREF,
            true,
            &sref_payload(&[(missing, *b"SHAP", BUILDER)]),
        ),
    ]);
    let bytes = c.to_bytes();
    assert_eq!(
        Container::from_bytes(&bytes),
        Err(ContainerError::DanglingSection { target: missing })
    );
    assert_eq!(
        ContainerView::open(&bytes).unwrap().references(),
        Err(ContainerError::DanglingSection { target: missing })
    );
}

#[test]
fn a_reference_to_a_section_of_another_kind_is_refused() {
    let world = b"w".to_vec();
    let c = build(&[
        (Tag::PHYS, true, &world),
        (
            Tag::SREF,
            true,
            &sref_payload(&[(sha(&world), *b"SHAP", BUILDER)]),
        ),
    ]);
    assert_eq!(
        Container::from_bytes(&c.to_bytes()),
        Err(ContainerError::DanglingSection {
            target: sha(&world)
        })
    );
}

#[test]
fn an_unknown_builder_is_refused_when_the_caller_lists_builders() {
    let view = ContainerView::open(BUNDLE).unwrap();
    assert_eq!(
        view.references_with_builders(&[b"other/v1"]),
        Err(ContainerError::UnknownBuilder { row: 0 })
    );
}

#[test]
fn malformed_reference_and_law_id_tables_are_refused() {
    for (tag, payload) in [
        (Tag::SREF, vec![1, 0, 0, 0, 0, 0, 0, 0]), // one row, no bytes
        (Tag::SREF, vec![0, 0, 0, 0, 0, 0, 0]),    // short count
        (Tag::SREF, {
            let mut p = sref_payload(&[([0; 32], *b"SHAP", BUILDER)]);
            p.push(0); // trailing byte
            p
        }),
        (Tag::LIDS, vec![1, 0, 0, 0, 0, 0, 0, 0]),
        (Tag::LIDS, lids_payload(&[(9, [0; 32], SEM)])), // index past the table
        (Tag::SREF, (u64::MAX).to_le_bytes().to_vec()),  // count larger than the payload
    ] {
        let c = build(&[(Tag::RAW, true, b"r"), (tag, true, &payload)]);
        assert_eq!(
            Container::from_bytes(&c.to_bytes()),
            Err(ContainerError::Malformed { tag }),
            "{tag:?} {payload:?}"
        );
    }
}

#[test]
fn law_ids_under_other_semantics_are_refused_by_the_checked_reader() {
    let mut v = BUNDLE.to_vec();
    v[16] ^= 0xFF; // header semantics id only
    let v = retrail(v);
    let view = ContainerView::open(&v).unwrap();
    let mut header = SEM;
    header[0] ^= 0xFF;
    let mismatch = ContainerError::SemanticsMismatch {
        container: header,
        section: SEM,
    };
    assert_eq!(view.law_ids(), Err(mismatch));
    assert_eq!(Container::from_bytes(&v), Err(mismatch));
    let rows = view.law_ids_unchecked_semantics().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].semantics_id, SEM);
    assert_eq!(view.semantics_id(), header);
}

fn fitted_law() -> SignalLaw {
    let points: Vec<(f64, f64)> = (0..16)
        .map(|i| (f64::from(i), 2.0 * f64::from(i) + 1.0))
        .collect();
    SignalLaw::fit_polynomial(&points, 1, Provenance::new("test", "linear")).unwrap()
}

#[test]
fn law_ids_are_compared_with_the_recomputed_identifier() {
    let law = fitted_law();
    let id = law.law_id(&SEMANTICS_ID);
    let c = {
        let mut c = Container::new(SEMANTICS_ID);
        c.push(Tag::LAW, true, b"law-bytes".to_vec());
        c.push(Tag::LIDS, true, lids_payload(&[(0, id, SEMANTICS_ID)]));
        c
    };
    let bytes = c.to_bytes();
    let view = ContainerView::open(&bytes).unwrap();
    view.verify_law_ids(|i, payload, sem| {
        assert_eq!((i, payload), (0, &b"law-bytes"[..]));
        Some(law.law_id(sem))
    })
    .unwrap();
    assert_eq!(
        view.verify_law_ids(|_, _, sem| {
            let mut other = law.law_id(sem);
            other[31] ^= 1;
            Some(other)
        }),
        Err(ContainerError::LawIdMismatch { section: 0 })
    );
    assert_eq!(
        view.verify_law_ids(|_, _, _| None),
        Err(ContainerError::LawIdMismatch { section: 0 })
    );
}

// ---------------------------------------------------------------- (4) earlier ALICE_ZIP files

#[test]
fn earlier_alice_zip_files_read_as_two_sections() {
    for (v, header_len) in [(LEGACY_V1, 65usize), (LEGACY_V2, 66)] {
        assert_eq!(detect(v), Format::LegacyAliceZip);
        let c = read_legacy_alice_zip(v).unwrap();
        assert_eq!(c.semantics_id, [0; 32]);
        assert_eq!(c.sections.len(), 2);
        assert_eq!(c.sections[0].tag, Tag::PROV);
        assert_eq!(c.sections[0].payload, &v[..header_len]);
        assert_eq!(c.sections[1].tag, Tag::RAW);
        assert_eq!(c.sections[1].payload, &v[header_len..]);
        assert_eq!(read_any(v).unwrap(), c, "read_any dispatches by the magic");
    }
}

#[test]
fn values_the_earlier_readers_let_through_are_refused() {
    assert_eq!(
        read_legacy_alice_zip(LEGACY_MAJOR2),
        Err(ContainerError::LegacyVersion { major: 2, minor: 0 })
    );
    assert_eq!(
        read_legacy_alice_zip(LEGACY_UNKNOWN_PAYLOAD),
        Err(ContainerError::LegacyField {
            field: "payload_type",
            value: 0x7F
        })
    );
    let mut v = LEGACY_V2.to_vec();
    v[11] = 0x06; // file_type
    assert_eq!(
        read_legacy_alice_zip(&v),
        Err(ContainerError::LegacyField {
            field: "file_type",
            value: 0x06
        })
    );
    let mut v = LEGACY_V2.to_vec();
    v[12] = 4; // engine index (four engines: 0..=3)
    assert_eq!(
        read_legacy_alice_zip(&v),
        Err(ContainerError::LegacyField {
            field: "engine",
            value: 4
        })
    );
    let mut v = LEGACY_V2.to_vec();
    v.push(0); // payload longer than compressed_size
    assert_eq!(read_legacy_alice_zip(&v), Err(ContainerError::Layout));
    let v = &LEGACY_V2[..LEGACY_V2.len() - 1];
    assert_eq!(read_legacy_alice_zip(v), Err(ContainerError::Layout));
    let mut v = LEGACY_V1.to_vec();
    v[10] = 2; // 1.2 was never written
    assert_eq!(
        read_legacy_alice_zip(&v),
        Err(ContainerError::LegacyVersion { major: 1, minor: 2 })
    );
}

#[test]
fn formats_are_told_apart_by_their_first_bytes() {
    assert_eq!(detect(BUNDLE), Format::Container);
    assert_eq!(detect(LEGACY_V1), Format::LegacyAliceZip);
    // an earlier viewer file starts with the five bytes "ALICE" too
    let viewer = b"ALICE\x01\x00\x00rest-of-a-viewer-file-that-is-long-enough-to-be-a-header....";
    assert_eq!(detect(viewer), Format::Unknown);
    assert_eq!(read_any(viewer), Err(ContainerError::BadMagic));
    assert_eq!(
        detect(b"ALICE_ZI"),
        Format::Unknown,
        "a prefix of the magic is not a match"
    );
    assert_eq!(read_any(BUNDLE).unwrap(), bundle());
}

// ---------------------------------------------------------------- degenerate input

fn no_panic<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> T {
    std::panic::catch_unwind(f).expect("must return an error, not panic")
}

#[test]
fn degenerate_inputs_return_the_stated_error() {
    let header_only = &ONE[..HEADER_LEN];
    let mut huge_count = EMPTY.to_vec();
    huge_count[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut offset_overflow = ONE.to_vec();
    offset_overflow[HEADER_LEN + 8..HEADER_LEN + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut length_past_end = ONE.to_vec();
    length_past_end[HEADER_LEN + 16..HEADER_LEN + 24].copy_from_slice(&1_000u64.to_le_bytes());
    let mut extra = ONE.to_vec();
    extra.insert(ONE.len() - TRAILER_LEN, 0);
    let cases: Vec<(&str, Vec<u8>, ContainerError)> = vec![
        ("empty", vec![], ContainerError::TooShort),
        ("seven bytes", ONE[..7].to_vec(), ContainerError::TooShort),
        (
            "header without trailer",
            header_only.to_vec(),
            ContainerError::TooShort,
        ),
        ("count u64::MAX", huge_count, ContainerError::Layout),
        ("offset overflow", offset_overflow, ContainerError::Layout),
        (
            "length past the end",
            length_past_end,
            ContainerError::Layout,
        ),
        (
            "extra byte before the trailer",
            extra,
            ContainerError::Layout,
        ),
    ];
    for (name, bytes, want) in cases {
        let got = no_panic(|| Container::from_bytes(&bytes));
        assert_eq!(got, Err(want), "{name}");
        let got = no_panic(|| ContainerView::open(&bytes).err());
        assert_eq!(got, Some(want), "{name} (view)");
    }
    for (name, bytes) in [("empty", &b""[..]), ("short", &LEGACY_V2[..64])] {
        assert_eq!(
            no_panic(|| read_legacy_alice_zip(bytes)),
            Err(ContainerError::TooShort),
            "{name}"
        );
    }
}

#[test]
fn errors_display_without_panicking() {
    for e in [
        ContainerError::TooShort,
        ContainerError::UnknownCritical { tag: Tag(*b"ZZZZ") },
        ContainerError::LegacyField {
            field: "engine",
            value: 9,
        },
    ] {
        assert!(!e.to_string().is_empty());
    }
}
