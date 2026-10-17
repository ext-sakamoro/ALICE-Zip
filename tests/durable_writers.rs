//! Every function of this crate that writes a durable form, and how each one
//! identifies the meaning of what it writes (ALICE-Law G8)
//!
//! The list is count-locked against the source: a public function whose name
//! says it writes (compress / encode / to_bytes / write / save / serialize /
//! store / dump / export / pack) and that is not in `WRITERS` fails, and so
//! does an entry the source no longer has. Each decision is then checked:
//! the bytes carry the identifier, or the module is shown to have nothing
//! the identifier would describe.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    /// The container format: section 0 is `FMID` (the format's identifier)
    /// and the header carries the semantics identifier the writer evaluated with
    FormatAndSemanticsId,
    /// A residual container: the model's semantics identifier at bytes 2..34
    SemanticsId,
    /// Integer arithmetic only: no floating-point type and no arithmetic
    /// identifier in the module; the archive names its law by its own
    /// `law_id` (kind, coefficients, domain digest) and a format version
    IntegerOnly,
    /// A byte compressor of the caller's bytes: nothing in the output depends
    /// on an arithmetic, so there is no meaning beyond the bytes to identify
    NoLaw,
}

use Decision::{FormatAndSemanticsId, IntegerOnly, NoLaw, SemanticsId};

/// (source file under src/, function, decision)
const WRITERS: [(&str, &str, Decision); 13] = [
    ("container.rs", "to_bytes", FormatAndSemanticsId),
    ("compression.rs", "compress_residual_quantized", SemanticsId),
    (
        "compression.rs",
        "compress_residual_quantized_with",
        SemanticsId,
    ),
    ("compression.rs", "compress_residual_lossless", SemanticsId),
    (
        "compression.rs",
        "compress_residual_lossless_with",
        SemanticsId,
    ),
    ("compression.rs", "compress_residual_xor", SemanticsId),
    ("compression.rs", "compress_residual_xor_with", SemanticsId),
    ("int2d.rs", "to_bytes", IntegerOnly),
    ("int2d.rs", "encode", IntegerOnly),
    ("int2d.rs", "encode_with", IntegerOnly),
    ("compression.rs", "zlib_compress", NoLaw),
    ("compression.rs", "lzma_compress", NoLaw),
    ("lz77.rs", "lz77_encode", NoLaw),
];

const WRITE_WORDS: [&str; 10] = [
    "compress", "encode", "to_bytes", "write", "save", "serial", "store", "dump", "export", "pack",
];
const READ_WORDS: [&str; 2] = ["decompress", "decode"];

fn sources(dir: &Path, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let rel = path
                .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, std::fs::read_to_string(&path).unwrap()));
        }
    }
}

/// `(file, name)` of every public function whose name says it writes
fn writers_in_source() -> BTreeSet<(String, String)> {
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    assert!(
        files.len() > 10,
        "read too few source files: {}",
        files.len()
    );
    let mut found = BTreeSet::new();
    for (file, text) in &files {
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("pub fn ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let writes = WRITE_WORDS.iter().any(|w| name.contains(w));
            let reads = READ_WORDS.iter().any(|w| name.contains(w));
            if writes && !reads {
                found.insert((file.clone(), name));
            }
        }
    }
    found
}

#[test]
fn the_writer_list_is_every_writer_in_the_source() {
    let listed: BTreeSet<(String, String)> = WRITERS
        .iter()
        .map(|(f, n, _)| ((*f).to_string(), (*n).to_string()))
        .collect();
    assert_eq!(listed.len(), WRITERS.len(), "an entry is listed twice");
    let found = writers_in_source();
    let unlisted: Vec<_> = found.difference(&listed).collect();
    let gone: Vec<_> = listed.difference(&found).collect();
    assert!(
        unlisted.is_empty() && gone.is_empty(),
        "writers without an identifier decision: {unlisted:?}; listed but not in the source: {gone:?}"
    );
    assert_eq!(found.len(), 13);
}

#[test]
fn the_container_writer_carries_the_format_and_semantics_identifiers() {
    use alice_zip::container::{FORMAT_ID, HEADER_LEN};
    let mut c = alice_zip::container::Container::new(alice_zip::law::SEMANTICS_ID);
    c.push(alice_zip::container::Tag::RAW, true, b"r".to_vec());
    let bytes = c.to_bytes();
    assert_eq!(bytes[16..48], alice_zip::law::SEMANTICS_ID);
    assert_eq!(&bytes[HEADER_LEN..HEADER_LEN + 4], b"FMID");
    let at = HEADER_LEN + 2 * alice_zip::container::ENTRY_LEN;
    assert_eq!(bytes[at..at + 32], FORMAT_ID);
}

#[test]
fn every_residual_writer_records_the_semantics_identifier() {
    use alice_zip::compression::{
        compress_residual_lossless_with, compress_residual_quantized_with,
        compress_residual_xor_with, ResidualCodec,
    };
    let x: Vec<f32> = (0..64u16).map(f32::from).collect();
    let codec = ResidualCodec::Deflate;
    let all = [
        compress_residual_quantized_with(&x, 8, codec, 6).unwrap(),
        compress_residual_lossless_with(&x, codec, 6).unwrap(),
        compress_residual_xor_with(&x, &x, codec, 6).unwrap(),
        alice_zip::compression::compress_residual_quantized(&x, 16, 6).unwrap(),
        alice_zip::compression::compress_residual_lossless(&x, 6).unwrap(),
        alice_zip::compression::compress_residual_xor(&x, &x, 6).unwrap(),
    ];
    assert_eq!(
        all.len(),
        WRITERS.iter().filter(|w| w.2 == SemanticsId).count()
    );
    for c in all {
        assert_eq!(c[2..34], alice_zip::law::SEMANTICS_ID);
    }
}

/// Code of `text` without line comments (`//`, `///`, `//!`)
fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_integer_only_module_has_no_float_and_no_arithmetic_identifier() {
    let files: BTreeSet<&str> = WRITERS
        .iter()
        .filter(|w| w.2 == IntegerOnly)
        .map(|w| w.0)
        .collect();
    assert_eq!(files.len(), 1);
    for file in files {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
        let code = code_only(&std::fs::read_to_string(path).unwrap());
        let words: Vec<&str> = code
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|w| !w.is_empty())
            .collect();
        assert!(words.len() > 1000, "{file}: read too little");
        for banned in [
            "f32",
            "f64",
            "alice_det_math",
            "SEMANTICS_ID",
            "check_semantics",
        ] {
            assert!(!words.contains(&banned), "{file} uses {banned}");
        }
        // what it takes from the rest of the crate: a byte-string domain and the
        // byte compressor, both without an arithmetic
        let mut used = BTreeSet::new();
        for (i, _) in code.match_indices("crate::") {
            let path: String = code[i..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            used.insert(path);
        }
        let allowed: BTreeSet<String> = [
            "crate::law::LAW_ID_DOMAIN",
            "crate::compression::zlib_compress",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        assert_eq!(used, allowed, "{file}: crate items used");
    }
}

#[test]
fn the_byte_compressors_take_and_return_only_bytes() {
    // NoLaw entries: their signature has no floating-point type, so nothing a
    // law computes can reach their output except as the caller's bytes
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let mut checked = 0;
    for (file, name, decision) in WRITERS {
        if decision != NoLaw {
            continue;
        }
        let text = &files.iter().find(|(f, _)| f == file).unwrap().1;
        let at = text.find(&format!("pub fn {name}(")).unwrap();
        let sig: String = text[at..].chars().take_while(|c| *c != '{').collect();
        assert!(sig.contains("&[u8]"), "{file} {name}: {sig}");
        assert!(
            !sig.contains("f32") && !sig.contains("f64"),
            "{file} {name}: {sig}"
        );
        checked += 1;
    }
    assert_eq!(checked, 3);
}
