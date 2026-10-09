//! Builds a container that bundles a shape, a state that refers to it, a law
//! and the law's identifier, reads it back, and checks the references and
//! the identifier.
//!
//! ```text
//! cargo run --example container_roundtrip
//! ```

use alice_zip::container::{read_any, Container, ContainerView, Tag};
use alice_zip::law::{Provenance, SignalLaw, SEMANTICS_ID};

const BUILDER: &[u8] = b"example/shape/v1";

fn main() {
    let points: Vec<(f64, f64)> = (0..32)
        .map(|i| (f64::from(i), 0.5 * f64::from(i) - 3.0))
        .collect();
    let law = SignalLaw::fit_polynomial(&points, 1, Provenance::new("example", "least squares"))
        .expect("fit");

    let mut c = Container::new(SEMANTICS_ID);
    let shape = c.push(Tag::SHAP, true, b"a shape description".to_vec());
    c.push(
        Tag::PHYS,
        true,
        b"a state that uses the shape twice".to_vec(),
    );
    let mut sref = 2u64.to_le_bytes().to_vec();
    for _ in 0..2 {
        sref.extend_from_slice(&shape);
        sref.extend_from_slice(&Tag::SHAP.0);
        sref.extend_from_slice(
            &u16::try_from(BUILDER.len())
                .expect("short name")
                .to_le_bytes(),
        );
        sref.extend_from_slice(BUILDER);
    }
    c.push(Tag::SREF, true, sref);
    c.push(Tag::LAW, true, b"law payload".to_vec());
    let mut lids = 1u64.to_le_bytes().to_vec();
    lids.extend_from_slice(&3u64.to_le_bytes());
    lids.extend_from_slice(&law.law_id(&SEMANTICS_ID));
    lids.extend_from_slice(&SEMANTICS_ID);
    c.push(Tag::LIDS, true, lids);

    let bytes = c.to_bytes();
    assert_eq!(read_any(&bytes).expect("read back"), c);

    let view = ContainerView::open(&bytes).expect("header, table and trailer");
    let refs = view
        .references_with_builders(&[BUILDER])
        .expect("references resolve");
    assert!(
        refs.iter().all(|r| r.section == 0),
        "one shape, stored once"
    );
    view.verify_law_ids(|_, _, semantics| Some(law.law_id(semantics)))
        .expect("law identifier matches");

    let mut damaged = bytes.clone();
    damaged[60] ^= 1;
    assert!(read_any(&damaged).is_err(), "a changed byte is refused");

    let id: String = c.id().iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "{} bytes, {} sections, {} references, id {id}",
        bytes.len(),
        view.len(),
        refs.len()
    );
}
