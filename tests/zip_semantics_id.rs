//! `alice_zip::law::SEMANTICS_ID` is the fold of `alice_zip::semantics::ZIP_SEMANTICS_PINS`.
//!
//! Every entry is checked against its source: each golden digest is read from the test of
//! `tests/determinism_golden.rs` it is named after (the file, not a copy), each tag entry is
//! the SHA-256 of the tag the crate uses, and each det-math entry is the resolved
//! `alice_det_math::function_pin`. Then `ZIP_SEMANTICS_ID` is recomputed with the fold written
//! here.

use alice_det_math::SemanticsCheck;
use alice_zip::law::{check_semantics, PREVIOUS_SEMANTICS_IDS, SEMANTICS_ID};
use alice_zip::semantics::{
    DET_MATH_FUNCTIONS, DET_MATH_PINS, ZIP_SEMANTICS_ID, ZIP_SEMANTICS_PINS,
};
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
    let tags: [(&str, &[u8]); 6] = [
        ("tag law::LAW_ID_DOMAIN", alice_zip::law::LAW_ID_DOMAIN),
        ("tag law::SIGNAL_LAW_KIND", alice_zip::law::SIGNAL_LAW_KIND),
        ("tag int2d::ZERO_LAW_KIND", alice_zip::int2d::ZERO_LAW_KIND),
        (
            "tag int2d::PLANE_PER_TILE_KIND",
            alice_zip::int2d::PLANE_PER_TILE_KIND,
        ),
        (
            "tag int2d::CAUSAL_PREV_KIND",
            alice_zip::int2d::CAUSAL_PREV_KIND,
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
        } else {
            let (_, tag) = tags
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("an entry of an unknown kind: {name}"));
            let want: [u8; 32] = Sha256::digest(tag).into();
            assert_eq!(*digest, want, "{name}");
        }
        checked += 1;
    }
    // every golden test that records a generator's bits is an entry, and every tag is
    let goldens = ZIP_SEMANTICS_PINS
        .iter()
        .filter(|(n, _)| n.starts_with("golden "))
        .count();
    assert_eq!(goldens, 14);
    for (tag, _) in tags {
        assert!(
            ZIP_SEMANTICS_PINS.iter().any(|(n, _)| *n == tag),
            "{tag} is not an entry"
        );
    }
    assert_eq!(checked, ZIP_SEMANTICS_PINS.len());
    assert!(checked >= 19);
}

#[test]
fn every_det_math_entry_is_the_resolved_function() {
    assert!(!DET_MATH_PINS.is_empty());
    assert_eq!(DET_MATH_FUNCTIONS.len(), DET_MATH_PINS.len());
    for (want, (name, digest)) in DET_MATH_FUNCTIONS.iter().zip(DET_MATH_PINS) {
        assert_eq!(want, name);
        assert_eq!(*digest, alice_det_math::function_pin(name), "{name}");
    }
}

/// the fold's entries: a det-math entry goes in under `det-math <function>`
fn entries() -> Vec<(String, [u8; 32])> {
    ZIP_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| (n.to_string(), d))
        .chain(
            DET_MATH_PINS
                .iter()
                .map(|&(n, d)| (format!("det-math {n}"), d)),
        )
        .collect()
}

#[test]
fn the_identifier_is_the_fold_of_the_pins() {
    let e = entries();
    let named: Vec<(&str, [u8; 32])> = e.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    assert_eq!(hex(&ZIP_SEMANTICS_ID), hex(&fold(&named)));
    assert_eq!(SEMANTICS_ID, ZIP_SEMANTICS_ID);
    let doc = include_str!("../src/semantics.rs");
    assert!(
        doc.contains(&format!("//! {}", hex(&SEMANTICS_ID))),
        "the module doc shows another identifier"
    );
}

#[test]
fn the_identifier_moves_with_each_entry() {
    let e = entries();
    let base: Vec<(&str, [u8; 32])> = e.iter().map(|(n, d)| (n.as_str(), *d)).collect();
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
    // the identifiers this crate published up to 0.10 (alice-det-math's) are previous ones, and the
    // last one that folded det-math's whole identifier
    for old in [
        "06c11067caed2f27b3323c25b023b1237fb5f5936bf86624b2dfb33759183461",
        "51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e",
        "65ba78c17a0cab6d1717270ad7ceca8856c293425f92289de943767ecb1ae883",
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
        "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000",
    ] {
        assert!(PREVIOUS_SEMANTICS_IDS.contains(&unhex(old)), "{old}");
    }
    assert_eq!(check_semantics(&[7; 32]), SemanticsCheck::Mismatch);
}

/// `tag <module>::<NAME>` of every `pub const <NAME>_KIND` in the given sources (file name, text)
fn kind_tags(sources: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (file, text) in sources {
        let module = file.trim_end_matches(".rs");
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("pub const ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            if name.ends_with("_KIND") {
                out.push(format!("tag {module}::{name}"));
            }
        }
    }
    out.sort();
    out
}

/// the kind tags that are not an entry of `pins`
fn missing_kind_tags(sources: &[(String, String)], pins: &[(&str, [u8; 32])]) -> Vec<String> {
    kind_tags(sources)
        .into_iter()
        .filter(|t| !pins.iter().any(|(n, _)| n == t))
        .collect()
}

fn src_files() -> Vec<(String, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "rs") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            out.push((name, std::fs::read_to_string(&p).unwrap()));
        }
    }
    out
}

#[test]
fn every_law_kind_tag_of_the_source_is_an_entry() {
    // a law kind added to the source without its entry would leave that law outside the identifier
    let sources = src_files();
    let tags = kind_tags(&sources);
    assert!(
        tags.len() >= 4,
        "found {} kind tags in src/ (a scan that finds none checks nothing)",
        tags.len()
    );
    assert_eq!(
        missing_kind_tags(&sources, ZIP_SEMANTICS_PINS),
        Vec::<String>::new()
    );
}

#[test]
fn a_kind_tag_without_its_entry_is_reported() {
    let mut sources = src_files();
    sources.push((
        "planted.rs".to_string(),
        "pub const MADE_UP_KIND: &[u8] = b\"made-up/v1\";\n".to_string(),
    ));
    assert_eq!(
        missing_kind_tags(&sources, ZIP_SEMANTICS_PINS),
        vec!["tag planted::MADE_UP_KIND".to_string()]
    );
    // and one taken out of the pins is reported too
    let pins: Vec<(&str, [u8; 32])> = ZIP_SEMANTICS_PINS
        .iter()
        .copied()
        .filter(|(n, _)| *n != "tag int2d::CAUSAL_PREV_KIND")
        .collect();
    assert_eq!(
        missing_kind_tags(&src_files(), &pins),
        vec!["tag int2d::CAUSAL_PREV_KIND".to_string()]
    );
}
