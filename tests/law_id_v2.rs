//! The v2 law identifier (`law_id::LawIdHasher`) against an independent implementation
//!
//! `tests/data/law_id/v2_probes.txt` holds probes (a kind, a semantics id, a list of typed
//! values) and the identifier `scripts/law_id_v2_reference.py` computes for each (standard-library
//! Python, no shared code). This test reads the same file and must compute every `expect` line;
//! CI runs the script with `--check` against the same file. So two implementations agree on the
//! bytes, and the file pins them.
//!
//! The probes separate what the encoding must keep apart: `u32 5` and `u64 5`, ("ab", "c") and
//! ("a", "bc"), bytes and a string of the same content, and the empty values. Floats enter by
//! their bits: -0.0 and +0.0 differ, f32 1.5 and f64 1.5 differ, the canonical NaN is accepted and
//! every other NaN is refused (an `expect refused NonCanonicalNan` line).
//!
//! Author: Moroya Sakamoto

use alice_det_math::law_bits::NonCanonicalNan;
use alice_zip::law_id::{LawIdHasher, LAW_ID_DOMAIN_V2};

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

struct Probe {
    name: String,
    kind: Vec<u8>,
    semantics: [u8; 32],
    values: Vec<(String, String)>,
    expect: String,
}

fn probes() -> Vec<Probe> {
    let text = include_str!("data/law_id/v2_probes.txt");
    let mut out = Vec::new();
    let mut cur: Option<Probe> = None;
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        match word {
            "probe" => {
                cur = Some(Probe {
                    name: rest.to_string(),
                    kind: Vec::new(),
                    semantics: [0; 32],
                    values: Vec::new(),
                    expect: String::new(),
                });
            }
            "kind" => cur.as_mut().unwrap().kind = unhex(rest),
            "semantics" => {
                cur.as_mut().unwrap().semantics = unhex(rest).try_into().unwrap();
            }
            "value" => {
                let (t, v) = rest.split_once(' ').unwrap_or((rest, ""));
                cur.as_mut()
                    .unwrap()
                    .values
                    .push((t.to_string(), v.to_string()));
            }
            "expect" => cur.as_mut().unwrap().expect = rest.to_string(),
            "end" => out.push(cur.take().unwrap()),
            other => panic!("unknown line {other:?}"),
        }
    }
    assert!(cur.is_none(), "the last probe has no end");
    out
}

fn compute(p: &Probe) -> Result<[u8; 32], NonCanonicalNan> {
    let mut h = LawIdHasher::new(&p.kind, &p.semantics);
    for (t, v) in &p.values {
        h = match t.as_str() {
            "f64" => h.f64(f64::from_bits(u64::from_str_radix(v, 16).unwrap()))?,
            "f32" => h.f32(f32::from_bits(u32::from_str_radix(v, 16).unwrap()))?,
            "bytes" => h.bytes(&unhex(v)),
            "u32" => h.u32(v.parse().unwrap()),
            "u64" => h.u64(v.parse().unwrap()),
            "str" => h.str(core::str::from_utf8(&unhex(v)).unwrap()),
            "id" => h.id(&unhex(v).try_into().unwrap()),
            other => panic!("{}: unknown value type {other}", p.name),
        };
    }
    Ok(h.finish())
}

fn shown(r: Result<[u8; 32], NonCanonicalNan>) -> String {
    match r {
        Ok(id) => hex(&id),
        Err(_) => "refused NonCanonicalNan".into(),
    }
}

#[test]
fn every_probe_gives_the_identifier_the_independent_implementation_gives() {
    let probes = probes();
    assert_eq!(probes.len(), 23, "the probe file changed size");
    for p in &probes {
        assert_eq!(shown(compute(p)), p.expect, "{}", p.name);
    }
    let refused = probes
        .iter()
        .filter(|p| p.expect.starts_with("refused"))
        .count();
    assert_eq!(refused, 5, "the refused probes changed in number");
}

#[test]
fn the_encoding_keeps_apart_what_it_must() {
    let probes = probes();
    let id = |name: &str| {
        let p = probes.iter().find(|p| p.name == name).unwrap();
        compute(p).unwrap()
    };
    assert_ne!(id("u32 five"), id("u64 five"));
    assert_ne!(id("str ab then c"), id("str a then bc"));
    assert_ne!(id("f64 plus zero"), id("f64 minus zero"));
    assert_ne!(id("f32 plus zero"), id("f32 minus zero"));
    assert_ne!(id("f32 one and a half"), id("f64 one and a half"));
    let accepted: Vec<&Probe> = probes
        .iter()
        .filter(|p| !p.expect.starts_with("refused"))
        .collect();
    let mut all: Vec<[u8; 32]> = accepted.iter().map(|p| compute(p).unwrap()).collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), accepted.len(), "two probes share an identifier");
}

#[test]
fn the_domain_names_its_version() {
    assert_eq!(LAW_ID_DOMAIN_V2, b"alice-law-id/v2");
}

#[test]
fn a_refused_nan_names_its_bits() {
    let h = LawIdHasher::new(b"k", &[0; 32]);
    let bits = 0x7ff8_0000_0000_0001_u64;
    match h.clone().f64(f64::from_bits(bits)) {
        Err(NonCanonicalNan::F64(b)) => assert_eq!(b, bits),
        other => panic!(
            "expected the f64 NaN refused, got {:?}",
            other.map(LawIdHasher::finish)
        ),
    }
    match h.f32(f32::from_bits(0xffc0_0000)) {
        Err(NonCanonicalNan::F32(b)) => assert_eq!(b, 0xffc0_0000),
        other => panic!(
            "expected the f32 NaN refused, got {:?}",
            other.map(LawIdHasher::finish)
        ),
    }
}
