//! A law identifier read from a container is trusted only under the
//! arithmetic it was computed with.
//!
//! A container is data: it opens and can be inspected whatever arithmetic
//! stamped it (a migration tool reads a file written by an older release).
//! What must not happen is a law identifier being used as a claim about
//! bits that the running build no longer produces. The trusting readers
//! (`law_ids_for_this_arithmetic`, `verify_law_ids`) compare the header's
//! semantics id with the running one through `alice_det_math::check_semantics`
//! and refuse with `NotThisArithmetic`; the structural readers (`law_ids`,
//! `law_ids_unchecked_semantics`, `Container::from_bytes`) do not.

use std::fs;
use std::path::Path;

use alice_zip::container::{Container, ContainerError, ContainerView, Tag};
use alice_zip::law::{Provenance, SignalLaw, SEMANTICS_ID};

/// The semantics id `alice-det-math` 0.4.0 published (`alice-zip` 0.9.0 and
/// earlier stamped it).
const SEMANTICS_ID_0_4_0: [u8; 32] = [
    0xd2, 0x20, 0x9b, 0x30, 0xf6, 0xf1, 0xf4, 0x5b, 0xaa, 0x1b, 0x63, 0x8b, 0xcd, 0xfe, 0xe3, 0x4a,
    0xc6, 0x47, 0x73, 0xb2, 0xe6, 0x3b, 0x9c, 0x08, 0x3b, 0x2e, 0x77, 0xaf, 0xc6, 0x91, 0x39, 0x8e,
];

/// The independent reference's fixture stamps this id (tests/container_oracle.rs).
const FIXTURE_SEMANTICS_ID: [u8; 32] = [0x11; 32];

fn law() -> SignalLaw {
    let points: Vec<(f64, f64)> = (0..16)
        .map(|i| (f64::from(i), 2.0 * f64::from(i) + 1.0))
        .collect();
    SignalLaw::fit_polynomial(&points, 1, Provenance::new("test", "linear")).unwrap()
}

fn lids_payload(rows: &[(u64, [u8; 32], [u8; 32])]) -> Vec<u8> {
    let mut p = (rows.len() as u64).to_le_bytes().to_vec();
    for (section, id, sem) in rows {
        p.extend_from_slice(&section.to_le_bytes());
        p.extend_from_slice(id);
        p.extend_from_slice(sem);
    }
    p
}

/// A container whose header and LIDS row both carry `semantics`, with the
/// law identifier computed under that same id.
fn stamped(semantics: [u8; 32]) -> Vec<u8> {
    let l = law();
    let mut c = Container::new(semantics);
    c.push(Tag::LAW, true, b"law-bytes".to_vec());
    c.push(
        Tag::LIDS,
        true,
        lids_payload(&[(1, l.law_id(&semantics), semantics)]),
    );
    c.to_bytes()
}

fn recompute(_: usize, _: &[u8], sem: &[u8; 32]) -> Option<[u8; 32]> {
    Some(law().law_id(sem))
}

#[test]
fn a_container_stamped_with_the_running_arithmetic_is_trusted() {
    let bytes = stamped(SEMANTICS_ID);
    let view = ContainerView::open(&bytes).unwrap();
    let rows = view.law_ids_for_this_arithmetic().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].semantics_id, SEMANTICS_ID);
    assert_eq!(view.verify_law_ids(recompute), Ok(()));
}

#[test]
fn a_container_stamped_by_0_4_0_opens_but_its_law_ids_are_not_trusted() {
    let bytes = stamped(SEMANTICS_ID_0_4_0);
    let refused = ContainerError::NotThisArithmetic {
        container: SEMANTICS_ID_0_4_0,
        running: SEMANTICS_ID,
    };
    // structural readers: the file is data and stays readable
    let c = Container::from_bytes(&bytes).unwrap();
    assert_eq!(c.semantics_id, SEMANTICS_ID_0_4_0);
    let view = ContainerView::open(&bytes).unwrap();
    assert_eq!(view.law_ids().unwrap().len(), 1);
    assert_eq!(view.law_ids_unchecked_semantics().unwrap().len(), 1);
    // trusting readers: refused up front, not one LawIdMismatch per row
    assert_eq!(view.law_ids_for_this_arithmetic(), Err(refused));
    assert_eq!(view.verify_law_ids(recompute), Err(refused));
    // the refusal does not depend on the recompute callback being consulted
    let mut called = false;
    let _ = view.verify_law_ids(|_, _, _| {
        called = true;
        None
    });
    assert!(
        !called,
        "verify_law_ids consulted the callback before refusing"
    );
}

#[test]
fn the_reference_fixture_opens_and_is_not_trusted_either() {
    let bytes = stamped(FIXTURE_SEMANTICS_ID);
    let view = ContainerView::open(&bytes).unwrap();
    assert!(view.law_ids().is_ok());
    assert_eq!(
        view.law_ids_for_this_arithmetic(),
        Err(ContainerError::NotThisArithmetic {
            container: FIXTURE_SEMANTICS_ID,
            running: SEMANTICS_ID,
        })
    );
}

/// Names of the `pub fn` items in `src` whose signature (from `pub fn` up to
/// the opening `{` or the `;` of a declaration, across lines) mentions
/// `LawIdEntry`. Line endings are normalised first, so a checkout with CRLF
/// reads the same as one with LF.
fn law_id_readers(src: &str) -> Vec<String> {
    let src = src.replace("\r\n", "\n");
    let mut found = Vec::new();
    let mut rest = src.as_str();
    while let Some(at) = rest.find("pub fn ") {
        let after = &rest[at + "pub fn ".len()..];
        let end = after.find(['{', ';']).unwrap_or(after.len());
        let signature = &after[..end];
        if signature.contains("LawIdEntry") {
            let name: String = signature
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            found.push(name);
        }
        rest = &after[end..];
    }
    found.sort();
    found
}

/// The body of `pub fn {name}` in `src`, from its signature to the closing
/// brace at the method's indentation (LF-normalised).
fn method_body(src: &str, name: &str) -> String {
    let src = src.replace("\r\n", "\n");
    let from = src
        .find(&format!("pub fn {name}"))
        .unwrap_or_else(|| panic!("`pub fn {name}` exists"));
    let tail = &src[from..];
    let end = tail.find("\n    }\n").expect("end of the fn");
    tail[..end].to_owned()
}

#[test]
fn the_reader_scan_sees_one_line_and_multi_line_signatures_with_crlf() {
    let src = "impl V {\r\n    pub fn one(&self) -> Vec<LawIdEntry> {\r\n    }\r\n    pub fn \
               two(\r\n        &self,\r\n    ) -> Result<\r\n        Vec<LawIdEntry>,\r\n        E,\r\n    > {\r\n    }\r\n    \
               pub fn other(&self) -> u8 {\r\n    }\r\n}\r\n";
    assert_eq!(
        law_id_readers(src),
        vec!["one".to_owned(), "two".to_owned()]
    );
    assert!(method_body(src, "one").contains("Vec<LawIdEntry>"));
}

/// Every `pub fn` of the container module that returns law identifier rows
/// is listed here with its role. A new one fails this test until it is
/// classified, so a path that hands out law ids cannot appear without a
/// decision on whether it trusts them.
#[test]
fn every_reader_of_law_ids_is_classified() {
    let src = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/container.rs"))
        .expect("read src/container.rs");
    let found = law_id_readers(&src);
    // (name, trusts the ids as claims about the running arithmetic)
    let classified = [
        ("law_ids", false),
        ("law_ids_for_this_arithmetic", true),
        ("law_ids_unchecked_semantics", false),
    ];
    let names: Vec<String> = classified.iter().map(|(n, _)| (*n).to_owned()).collect();
    assert!(
        !found.is_empty(),
        "compared nothing: no reader of law ids found"
    );
    assert_eq!(
        found, names,
        "classify every pub fn that returns LawIdEntry"
    );
    // the trusting readers go through the arithmetic check
    for (name, trusts) in classified {
        if trusts {
            assert!(
                method_body(&src, name).contains("check_semantics"),
                "{name} trusts law ids but does not call check_semantics"
            );
        }
    }
    assert!(
        method_body(&src, "verify_law_ids").contains("self.law_ids_for_this_arithmetic()"),
        "verify_law_ids must read the rows through the trusting reader"
    );
}
