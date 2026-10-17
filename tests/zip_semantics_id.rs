//! `alice_zip::law::SEMANTICS_ID` is the fold of `alice_zip::semantics::ZIP_SEMANTICS_PINS`.
//!
//! Every entry is checked against its source: each golden digest is read from the test of
//! `tests/determinism_golden.rs` it is named after (the file, not a copy), each tag entry is
//! the SHA-256 of the tag the crate uses, and the det-math entry is the resolved
//! `alice_det_math::SEMANTICS_ID`. Then every row of `ZIP_SEMANTICS_BY_DET_MATH` is
//! recomputed with the fold written here.

use alice_det_math::SemanticsCheck;
use alice_zip::law::{check_semantics, PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};
use alice_zip::semantics::{semantics_id_for, ZIP_SEMANTICS_BY_DET_MATH, ZIP_SEMANTICS_PINS};
use sha2::{Digest, Sha256};

const GOLDEN_SOURCE: &str = include_str!("determinism_golden.rs");

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// the digest the test `name` of determinism_golden.rs passes to assert_golden: the first
/// 64-hex-digit string literal after `fn name(`
fn golden_of(name: &str) -> [u8; 32] {
    let at = GOLDEN_SOURCE
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("no test {name} in determinism_golden.rs"));
    let rest = &GOLDEN_SOURCE[at..];
    let end = rest[1..].find("\nfn ").map_or(rest.len(), |e| e + 1);
    let body = &rest[..end];
    let lit = body
        .split('"')
        .find(|t| t.len() == 64 && t.bytes().all(|c| c.is_ascii_hexdigit()))
        .unwrap_or_else(|| panic!("test {name} records no digest"));
    unhex(lit)
}

/// the fold of alice-det-math's SEMANTICS_ID: sort by name, then for each entry the name
/// length (u32 big-endian), the name and the digest, under SHA-256
fn fold(entries: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut e = entries.to_vec();
    e.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for w in e.windows(2) {
        assert_ne!(w[0].0, w[1].0, "repeated name");
    }
    let mut h = Sha256::new();
    for (name, d) in &e {
        assert!(!name.is_empty() && name.is_ascii());
        h.update(u32::try_from(name.len()).unwrap().to_be_bytes());
        h.update(name.as_bytes());
        h.update(d);
    }
    h.finalize().into()
}

#[test]
fn every_entry_matches_its_source() {
    let tags: [(&str, &[u8]); 5] = [
        ("tag law::LAW_ID_DOMAIN", alice_zip::law::LAW_ID_DOMAIN),
        ("tag law::SIGNAL_LAW_KIND", alice_zip::law::SIGNAL_LAW_KIND),
        ("tag int2d::ZERO_LAW_KIND", alice_zip::int2d::ZERO_LAW_KIND),
        (
            "tag int2d::PLANE_PER_TILE_KIND",
            alice_zip::int2d::PLANE_PER_TILE_KIND,
        ),
        (
            "tag int2d::DOMAIN_DIGEST_DOMAIN",
            alice_zip::int2d::DOMAIN_DIGEST_DOMAIN,
        ),
    ];
    let mut checked = 0;
    for (name, digest) in ZIP_SEMANTICS_PINS {
        if let Some(test) = name.strip_prefix("golden ") {
            assert_eq!(hex(digest), hex(&golden_of(test)), "{name}");
        } else if let Some((_, tag)) = tags.iter().find(|(n, _)| n == name) {
            let want: [u8; 32] = Sha256::digest(tag).into();
            assert_eq!(*digest, want, "{name}");
        } else {
            assert_eq!(
                *name, "alice-det-math",
                "an entry of an unknown kind: {name}"
            );
            assert_eq!(*digest, alice_det_math::SEMANTICS_ID);
        }
        checked += 1;
    }
    // every golden test that records a generator's bits is an entry, and every tag is
    let goldens = ZIP_SEMANTICS_PINS
        .iter()
        .filter(|(n, _)| n.starts_with("golden "))
        .count();
    assert_eq!(goldens, 5);
    for (tag, _) in tags {
        assert!(
            ZIP_SEMANTICS_PINS.iter().any(|(n, _)| *n == tag),
            "{tag} is not an entry"
        );
    }
    assert_eq!(checked, ZIP_SEMANTICS_PINS.len());
    assert!(checked >= 10);
}

#[test]
fn every_row_is_the_fold_of_the_pins() {
    assert!(!ZIP_SEMANTICS_BY_DET_MATH.is_empty());
    for (det_math, id) in ZIP_SEMANTICS_BY_DET_MATH {
        let entries: Vec<(&str, [u8; 32])> = ZIP_SEMANTICS_PINS
            .iter()
            .map(|&(n, d)| {
                if n == "alice-det-math" {
                    (n, *det_math)
                } else {
                    (n, d)
                }
            })
            .collect();
        assert_eq!(
            hex(id),
            hex(&fold(&entries)),
            "row for det-math {}",
            hex(det_math)
        );
        assert_eq!(semantics_id_for(det_math), *id);
    }
    // the build's identifier is the row of the resolved det-math, and the doc's hex form
    assert_eq!(
        SEMANTICS_ID,
        semantics_id_for(&alice_det_math::SEMANTICS_ID)
    );
    let doc = include_str!("../src/semantics.rs");
    assert!(
        doc.contains(&hex(&SEMANTICS_ID)),
        "the module doc shows another identifier"
    );
}

#[test]
fn the_identifier_moves_with_each_entry() {
    let base: Vec<(&str, [u8; 32])> = ZIP_SEMANTICS_PINS.to_vec();
    let id = fold(&base);
    assert_eq!(id, SEMANTICS_ID);
    for i in 0..base.len() {
        let mut changed = base.clone();
        changed[i].1[0] ^= 1;
        assert_ne!(
            fold(&changed),
            id,
            "entry {} does not reach the identifier",
            base[i].0
        );
        let mut dropped = base.clone();
        dropped.remove(i);
        assert_ne!(
            fold(&dropped),
            id,
            "dropping {} leaves the identifier",
            base[i].0
        );
    }
}

#[test]
fn check_semantics_reads_this_and_previous_identifiers() {
    assert_eq!(check_semantics(&SEMANTICS_ID), SemanticsCheck::Verified);
    assert!(!PREVIOUS_SEMANTICS_IDS.is_empty());
    assert!(!PREVIOUS_SEMANTICS_IDS.contains(&SEMANTICS_ID));
    for p in PREVIOUS_SEMANTICS_IDS {
        assert_eq!(check_semantics(p), SemanticsCheck::VerifiedPrevious(*p));
    }
    // the identifiers this crate published up to 0.10 (alice-det-math's) are previous ones
    for old in [
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
        "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000",
    ] {
        assert!(PREVIOUS_SEMANTICS_IDS.contains(&unhex(old)), "{old}");
    }
    assert_eq!(check_semantics(&[7; 32]), SemanticsCheck::Mismatch);
}
