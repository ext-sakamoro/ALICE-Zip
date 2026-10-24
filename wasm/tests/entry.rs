//! The browser entry gives what the core gives: the same container fields and the same refusals as
//! `alice_zip::container::ContainerView`, and Ed25519 verification as alice-sign's strict check.
//!
//! Each test runs on the host (`cargo test`) and in Node (`cargo test --target wasm32-unknown-unknown`, with
//! wasm-bindgen-test's runner). Oracles: the core's own reading of the same bytes (ContainerView, the reference
//! the entry must equal) and RFC 8032 §7.1 test 1 for the signature.

use alice_zip::container::{Container, ContainerView, Tag};
use alice_zip_wasm::{verify_ed25519, Opened};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// A container with a law section (critical), a raw section, and a section of a format's own tag
fn sample(own_critical: bool) -> Vec<u8> {
    let mut c = Container::new([0x5a; 32]);
    c.push(Tag::LAW, true, b"law bytes".to_vec());
    c.push(Tag::RAW, false, vec![1, 2, 3, 4, 5]);
    c.push(
        Tag(*b"XTRA"),
        own_critical,
        b"a format's own section".to_vec(),
    );
    c.to_bytes()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn the_entry_reads_the_fields_the_core_reads() {
    let bytes = sample(false);
    let core = ContainerView::open(&bytes).unwrap();
    let entry = Opened::open(bytes.clone(), Vec::new()).unwrap();
    assert_eq!(entry.semantics_id(), core.semantics_id().to_vec());
    assert_eq!(entry.len(), core.len());
    // the three pushed sections and the format identifier the core writes first
    assert_eq!(entry.len(), 4);
    assert_eq!(entry.format_status(), format!("{:?}", core.format_status()));
    for i in 0..core.len() {
        assert_eq!(entry.tag(i).unwrap(), core.tag(i).0.to_vec(), "section {i}");
        assert_eq!(entry.critical(i).unwrap(), core.critical(i), "section {i}");
        assert_eq!(
            entry.section(i).unwrap(),
            core.section(i).unwrap().to_vec(),
            "section {i}"
        );
    }
    let raw = (0..core.len()).find(|&i| core.tag(i) == Tag::RAW).unwrap();
    assert_eq!(entry.section(raw).unwrap(), vec![1, 2, 3, 4, 5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_critical_tag_of_its_own_is_read_only_when_declared() {
    let bytes = sample(true);
    let refused = Opened::open(bytes.clone(), Vec::new())
        .err()
        .expect("an undeclared critical tag");
    assert_eq!(
        refused,
        ContainerView::open(&bytes).err().unwrap().to_string()
    );
    // declared: the tags are 4 bytes each, concatenated
    let opened = Opened::open(bytes.clone(), b"XTRA".to_vec()).unwrap();
    let core = ContainerView::open_with_known(&bytes, &[Tag(*b"XTRA")]).unwrap();
    let own = (0..core.len())
        .find(|&i| core.tag(i) == Tag(*b"XTRA"))
        .unwrap();
    assert_eq!(opened.tag(own).unwrap(), b"XTRA".to_vec());
    assert!(opened.critical(own).unwrap());
    // a known list that is not whole 4-byte tags is refused, not cut
    assert!(Opened::open(bytes, b"YOI".to_vec()).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn damaged_bytes_are_refused_as_the_core_refuses_them() {
    let good = sample(false);
    let mut cases = vec![Vec::new(), good[..good.len() - 1].to_vec()];
    let mut trailer = good.clone();
    *trailer.last_mut().unwrap() ^= 1;
    cases.push(trailer);
    let mut body = good.clone();
    let mid = body.len() / 2;
    body[mid] ^= 1;
    cases.push(body);
    for bytes in cases {
        let core = ContainerView::open(&bytes)
            .err()
            .expect("the core refuses it");
        assert_eq!(
            Opened::open(bytes, Vec::new()).err(),
            Some(core.to_string())
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn an_index_past_the_sections_is_an_error_not_a_trap() {
    let opened = Opened::open(sample(false), Vec::new()).unwrap();
    let n = opened.len();
    assert!(opened.tag(n - 1).is_ok());
    assert!(opened.tag(n).is_err());
    assert!(opened.critical(n).is_err());
    assert!(opened.section(n).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ed25519_verifies_rfc_8032_test_1_strictly() {
    let pk = hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    let sig = hex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    );
    assert!(verify_ed25519(&pk, b"", &sig));
    let mut bad = sig.clone();
    bad[0] ^= 1;
    assert!(!verify_ed25519(&pk, b"", &bad));
    assert!(!verify_ed25519(&pk, b"x", &sig));
    assert!(!verify_ed25519(&pk[..31], b"", &sig));
    assert!(!verify_ed25519(&pk, b"", &sig[..63]));
}
