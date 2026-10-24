//! The browser entry of ALICE-Zip (wasm32-unknown-unknown, through wasm-bindgen).
//!
//! A web page opens a container and checks a signature with the same code the host uses: [`Opened`] is the core's
//! `ContainerView` over bytes the page hands in (the same checks, the same fields, the core's error text), and
//! [`verify_ed25519`] is alice-sign's strict Ed25519 verification. Nothing here parses or hashes on its own.
//!
//! Build: `cargo build --release --target wasm32-unknown-unknown` in this directory, then `wasm-bindgen` for
//! the page. The functions are plain Rust as well, so the tests run on the host and in Node alike.

use alice_zip::container::{ContainerView, Tag};
use wasm_bindgen::prelude::wasm_bindgen;

/// A container that opened: its bytes and the tags declared as known when it was opened.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Opened {
    bytes: Vec<u8>,
    known: Vec<Tag>,
}

#[wasm_bindgen]
impl Opened {
    /// Open `bytes` with the core's checks; `known` is the format's own tags, 4 bytes each, concatenated (a
    /// critical section of an undeclared tag is refused). The error is the core's message.
    pub fn open(bytes: Vec<u8>, known: Vec<u8>) -> Result<Opened, String> {
        if !known.len().is_multiple_of(4) {
            return Err(format!(
                "the known tags are {} bytes, not whole 4-byte tags",
                known.len()
            ));
        }
        let known: Vec<Tag> = known.as_chunks::<4>().0.iter().map(|c| Tag(*c)).collect();
        ContainerView::open_with_known(&bytes, &known).map_err(|e| e.to_string())?;
        Ok(Self { bytes, known })
    }

    fn view(&self) -> ContainerView<'_> {
        // (opened once in `open`; the bytes and the known tags are unchanged since)
        ContainerView::open_with_known(&self.bytes, &self.known)
            .unwrap_or_else(|e| unreachable!("the bytes opened before: {e}"))
    }

    fn index(&self, index: usize) -> Result<usize, String> {
        let n = self.view().len();
        if index < n {
            Ok(index)
        } else {
            Err(format!("section {index} of {n}"))
        }
    }

    /// The container's semantics id (32 bytes).
    #[wasm_bindgen(js_name = semanticsId)]
    #[must_use]
    pub fn semantics_id(&self) -> Vec<u8> {
        self.view().semantics_id().to_vec()
    }

    /// The number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.view().len()
    }

    /// Whether the container has no sections.
    #[wasm_bindgen(js_name = isEmpty)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.view().is_empty()
    }

    /// The format status, as the core names it (`Debug` of `FormatStatus`).
    #[wasm_bindgen(js_name = formatStatus)]
    #[must_use]
    pub fn format_status(&self) -> String {
        format!("{:?}", self.view().format_status())
    }

    /// The tag of section `index` (4 bytes).
    pub fn tag(&self, index: usize) -> Result<Vec<u8>, String> {
        let i = self.index(index)?;
        Ok(self.view().tag(i).0.to_vec())
    }

    /// Whether section `index` is critical.
    pub fn critical(&self, index: usize) -> Result<bool, String> {
        let i = self.index(index)?;
        Ok(self.view().critical(i))
    }

    /// The payload of section `index`, checked against its digest by the core.
    pub fn section(&self, index: usize) -> Result<Vec<u8>, String> {
        let i = self.index(index)?;
        self.view()
            .section(i)
            .map(<[u8]>::to_vec)
            .map_err(|e| e.to_string())
    }
}

/// Whether `signature` (64 bytes) is `public_key`'s (32 bytes) Ed25519 signature of `message`, by alice-sign's
/// strict verification; a key or signature of another length is not.
#[wasm_bindgen(js_name = verifyEd25519)]
#[must_use]
pub fn verify_ed25519(public_key: &[u8], message: &[u8], signature: &[u8]) -> bool {
    let (Ok(pk), Ok(sig)) = (
        <[u8; 32]>::try_from(public_key),
        <[u8; 64]>::try_from(signature),
    ) else {
        return false;
    };
    alice_sign::PublicKey(pk).verify_strict(message, &alice_sign::Signature(sig))
}
