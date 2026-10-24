//! The v2 law reference (`law_ref::LawRef`) against an independent implementation
//!
//! `tests/data/law_ref/v2_probes.txt` holds `ref` probes (a kind, a semantics id and named,
//! typed rows, with the bytes and the law identifier `scripts/law_ref_v2_reference.py` computes)
//! and `reject` probes (bytes a reader must refuse, with the reason). This test reads the same
//! file: every `ref` must encode to its bytes and decode back, and every `reject` must be refused
//! for its reason. CI runs the script with `--check` against the same file.
//!
//! The law identifier inside the bytes is recomputed by the reader, so a reference whose rows,
//! names, kind or semantics differ from what its identifier names is refused (four probes change
//! one of them under an unchanged identifier).
//!
//! Author: Moroya Sakamoto

use alice_zip::law_ref::{LawRef, LawRefError, Value, MAX_ROWS};

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd hex {s:?}");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn utf8(s: &str) -> String {
    String::from_utf8(unhex(s)).unwrap()
}

#[derive(Default)]
struct Probe {
    reject: bool,
    name: String,
    kind: Vec<u8>,
    semantics: [u8; 32],
    rows: Vec<(String, Value)>,
    bytes: Vec<u8>,
    expect_bytes: String,
    expect_id: String,
    expect_error: String,
}

fn probes() -> Vec<Probe> {
    let text = include_str!("data/law_ref/v2_probes.txt");
    let mut out = Vec::new();
    let mut cur: Option<Probe> = None;
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        match word {
            "ref" | "reject" => {
                cur = Some(Probe {
                    reject: word == "reject",
                    name: rest.to_string(),
                    ..Probe::default()
                });
            }
            "kind" => cur.as_mut().unwrap().kind = unhex(rest),
            "semantics" => cur.as_mut().unwrap().semantics = unhex(rest).try_into().unwrap(),
            "row" => {
                let mut parts = rest.splitn(3, ' ');
                let name = utf8(parts.next().unwrap());
                let t = parts.next().unwrap();
                let v = parts.next().unwrap_or("");
                let value = match t {
                    "bytes" => Value::Bytes(unhex(v)),
                    "u32" => Value::U32(v.parse().unwrap()),
                    "u64" => Value::U64(v.parse().unwrap()),
                    "str" => Value::Str(utf8(v)),
                    "id" => Value::Id(unhex(v).try_into().unwrap()),
                    other => panic!("unknown value type {other}"),
                };
                cur.as_mut().unwrap().rows.push((name, value));
            }
            "bytes" => cur.as_mut().unwrap().bytes = unhex(rest),
            "expect_bytes" => cur.as_mut().unwrap().expect_bytes = rest.to_string(),
            "expect_id" => cur.as_mut().unwrap().expect_id = rest.to_string(),
            "expect_error" => cur.as_mut().unwrap().expect_error = rest.to_string(),
            "end" => out.push(cur.take().unwrap()),
            other => panic!("unknown line {other:?}"),
        }
    }
    assert!(cur.is_none(), "the last probe has no end");
    out
}

fn reason(e: &LawRefError) -> &'static str {
    match e {
        LawRefError::BadMagic => "BadMagic",
        LawRefError::UnsupportedVersion(_) => "UnsupportedVersion",
        LawRefError::Truncated => "Truncated",
        LawRefError::EmptyKind => "EmptyKind",
        LawRefError::KindTooLong(_) => "KindTooLong",
        LawRefError::NotUtf8 => "NotUtf8",
        LawRefError::TooManyRows(_) => "TooManyRows",
        LawRefError::EmptyName => "EmptyName",
        LawRefError::NameTooLong(_) => "NameTooLong",
        LawRefError::DuplicateName(_) => "DuplicateName",
        LawRefError::UnknownTag(_) => "UnknownTag",
        LawRefError::TrailingBytes => "TrailingBytes",
        LawRefError::LawIdMismatch { .. } => "LawIdMismatch",
    }
}

#[test]
fn every_reference_encodes_to_the_bytes_the_independent_implementation_gives() {
    let probes = probes();
    let refs: Vec<&Probe> = probes.iter().filter(|p| !p.reject).collect();
    assert_eq!(refs.len(), 11, "the ref probes changed in number");
    for p in refs {
        let r = LawRef::new(&p.kind, p.semantics, p.rows.clone()).unwrap();
        assert_eq!(hex(&r.to_bytes()), p.expect_bytes, "{}", p.name);
        assert_eq!(hex(r.law_id()), p.expect_id, "{}", p.name);
        let back = LawRef::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back, r, "{}: does not decode to itself", p.name);
        assert_eq!(back.kind(), &p.kind[..]);
        assert_eq!(back.semantics_id(), &p.semantics);
        assert_eq!(back.rows(), &p.rows[..]);
    }
}

#[test]
fn every_rejected_byte_string_is_refused_for_its_reason() {
    let probes = probes();
    let rejects: Vec<&Probe> = probes.iter().filter(|p| p.reject).collect();
    assert_eq!(rejects.len(), 18, "the reject probes changed in number");
    for p in rejects {
        let e = LawRef::from_bytes(&p.bytes).expect_err(&p.name);
        assert_eq!(reason(&e), p.expect_error, "{}: {e:?}", p.name);
    }
}

#[test]
fn a_mismatched_law_id_names_both_identifiers() {
    let r = LawRef::new(b"probe/law", [0x11; 32], vec![("x".into(), Value::U32(5))]).unwrap();
    let mut b = r.to_bytes();
    // the law id follows magic (4), version (2), kind length (1), kind (9) and semantics (32)
    b[48] ^= 1;
    match LawRef::from_bytes(&b) {
        Err(LawRefError::LawIdMismatch { stated, computed }) => {
            assert_eq!(&computed, r.law_id());
            assert_ne!(stated, computed);
            assert_eq!(stated[0], r.law_id()[0] ^ 1);
        }
        other => panic!("expected LawIdMismatch, got {other:?}"),
    }
}

#[test]
fn the_constructor_refuses_what_the_reader_refuses() {
    let row = |n: &str| (n.to_string(), Value::U32(0));
    assert_eq!(
        LawRef::new(b"", [0; 32], vec![]).unwrap_err(),
        LawRefError::EmptyKind
    );
    assert_eq!(
        LawRef::new(&[b'k'; 256], [0; 32], vec![]).unwrap_err(),
        LawRefError::KindTooLong(256)
    );
    assert_eq!(
        LawRef::new(&[0xff], [0; 32], vec![]).unwrap_err(),
        LawRefError::NotUtf8
    );
    assert_eq!(
        LawRef::new(b"k", [0; 32], vec![row("a"), row("a")]).unwrap_err(),
        LawRefError::DuplicateName("a".into())
    );
    assert_eq!(
        LawRef::new(b"k", [0; 32], vec![row("")]).unwrap_err(),
        LawRefError::EmptyName
    );
    assert_eq!(
        LawRef::new(b"k", [0; 32], vec![row(&"n".repeat(256))]).unwrap_err(),
        LawRefError::NameTooLong(256)
    );
    let many: Vec<_> = (0..=MAX_ROWS).map(|i| row(&format!("r{i}"))).collect();
    assert_eq!(
        LawRef::new(b"k", [0; 32], many).unwrap_err(),
        LawRefError::TooManyRows(MAX_ROWS + 1)
    );
}

#[test]
fn the_law_id_is_the_v2_law_identifier_of_the_rows() {
    // the rows enter the identifier as (name as a str, value), in order, after kind and semantics
    use alice_zip::law_id::LawIdHasher;
    let r = LawRef::new(
        b"probe/law",
        [0x11; 32],
        vec![("x".into(), Value::U32(5)), ("s".into(), Value::Str("ab".into()))],
    )
    .unwrap();
    let id = LawIdHasher::new(b"probe/law", &[0x11; 32])
        .str("x")
        .u32(5)
        .str("s")
        .str("ab")
        .finish();
    assert_eq!(r.law_id(), &id);
}
