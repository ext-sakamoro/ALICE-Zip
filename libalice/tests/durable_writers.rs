//! Every function of the CLI crate whose name says it writes, and how it
//! identifies the meaning of what it writes (ALICE-Law G8)
//!
//! Count-locked against the source: a `fn` outside test code whose name says
//! it writes (compress / encode / to_bytes / write / save / serial / store /
//! dump / export / pack, not decompress / decode) and that is not in
//! `WRITERS` fails, and so does an entry the source no longer has. The
//! decisions that put an identifier in the bytes are checked here or name the
//! test that checks them.

use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    /// Writes a residual file with `"semantics"` (versions 5 to 7)
    SemanticsId,
    /// Builds a residual that carries this crate's identifier
    /// (`metadata.semantics`), written by `ResidualData::to_bytes`
    Stamped,
    /// Writes the core crate's residual containers, which carry the
    /// identifier (`alice_zip::compression`, see its tests)
    CrateContainer,
    /// The fixed 66-byte `ALICE_ZIP` header: it labels the payload, whose
    /// format carries the identity
    HeaderLabels,
    /// The caller's values or bytes, stored as they are (or quantised by
    /// basic IEEE operations): decoding regenerates nothing from a law
    Literal,
    /// Returns a value in memory and writes nothing durable
    NotDurable,
}

use Decision::{CrateContainer, HeaderLabels, Literal, NotDurable, SemanticsId, Stamped};

/// (file under src/, function, decision)
const WRITERS: [(&str, &str, Decision); 19] = [
    ("residual.rs", "to_bytes", SemanticsId),
    ("residual.rs", "compress_residual_delta", Stamped),
    ("residual.rs", "choose_compression", Stamped),
    ("residual.rs", "compress_original", Stamped),
    ("residual.rs", "compress_with_method", NotDurable), // private, stamped by its callers
    ("residual.rs", "xz_compress", Literal),
    ("residual.rs", "bit_delta_encode", Literal),
    ("codec_bridge.rs", "compress_subband", CrateContainer),
    ("codec_bridge.rs", "compress_all_subbands", CrateContainer),
    ("lib.rs", "residual_compress", CrateContainer),
    ("lib.rs", "residual_compress_lossless", CrateContainer),
    ("ffi.rs", "alice_residual_compress", CrateContainer),
    ("format.rs", "to_bytes", HeaderLabels),
    ("lib.rs", "lzma_compress", Literal),
    ("lib.rs", "zlib_compress", Literal),
    ("ffi.rs", "alice_lzma_compress", Literal),
    ("ffi.rs", "alice_zlib_compress", Literal),
    ("bin/alice.rs", "cmd_compress", Literal), // the ALZ file: zlib or 8/16-bit quantisation of the input
    ("media/audio.rs", "write_wav_f32", Literal),
];
// further names the scan finds that are not writers
const NOT_WRITERS: [(&str, &str, &str); 2] = [
    (
        "bin/alice.rs",
        "to_bytes",
        "the ALZ header of cmd_compress (magic, version, mode, shape, size)",
    ),
    (
        "analyzer.rs",
        "compress",
        "returns an in-memory CompressedPayload; nothing in this crate writes it",
    ),
];

const WORDS: [&str; 10] = [
    "compress", "encode", "to_bytes", "write", "save", "serial", "store", "dump", "export", "pack",
];

fn sources(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, root, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path).unwrap();
            // the unit tests at the end of a file are not production code
            let code = match text.find("#[cfg(test)]\nmod tests") {
                Some(i) => text[..i].to_owned(),
                None => text,
            };
            out.push((rel, code));
        }
    }
}

/// (file, name) pairs, and (file, code) of every source file
type Scan = (BTreeSet<(String, String)>, Vec<(String, String)>);

fn writers_in_source() -> Scan {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&root, &root, &mut files);
    assert!(files.len() > 10, "read too few files: {}", files.len());
    let mut found = BTreeSet::new();
    for (file, text) in &files {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            let rest = [
                "pub fn ",
                "fn ",
                "pub(crate) fn ",
                "pub extern \"C\" fn ",
                "pub unsafe extern \"C\" fn ",
                "extern \"C\" fn ",
            ]
            .iter()
            .find_map(|p| t.strip_prefix(p));
            let Some(rest) = rest else { continue };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let test_fn = i > 0
                && lines[..i]
                    .iter()
                    .rev()
                    .take(3)
                    .any(|l| l.trim() == "#[test]");
            if test_fn || name.starts_with("test_") {
                continue;
            }
            if WORDS.iter().any(|w| name.contains(w))
                && !name.contains("decompress")
                && !name.contains("decode")
            {
                found.insert((file.clone(), name));
            }
        }
    }
    (found, files)
}

/// Where `fn name(` or `fn name<` starts in `text`
fn fn_at(text: &str, name: &str) -> usize {
    [format!("fn {name}("), format!("fn {name}<")]
        .iter()
        .find_map(|p| text.find(p.as_str()))
        .unwrap_or_else(|| panic!("fn {name} not found"))
}

#[test]
fn the_writer_list_is_every_writer_in_the_source() {
    let (found, _) = writers_in_source();
    let listed: BTreeSet<(String, String)> = WRITERS
        .iter()
        .map(|(f, n, _)| ((*f).to_owned(), (*n).to_owned()))
        .chain(
            NOT_WRITERS
                .iter()
                .map(|(f, n, _)| ((*f).to_owned(), (*n).to_owned())),
        )
        .collect();
    assert_eq!(
        listed.len(),
        WRITERS.len() + NOT_WRITERS.len(),
        "an entry is listed twice"
    );
    let unlisted: Vec<_> = found.difference(&listed).collect();
    let gone: Vec<_> = listed.difference(&found).collect();
    assert!(
        unlisted.is_empty() && gone.is_empty(),
        "without a decision: {unlisted:?}; listed but not in the source: {gone:?}"
    );
    assert_eq!(found.len(), 21);
}

#[test]
fn the_stamped_constructors_carry_this_crates_identifier() {
    use alice_core::residual::{
        choose_compression, compress_original, compress_residual_delta, ResidualCompressionMethod,
    };
    use alice_zip::law::SEMANTICS_ID;
    let x = [1.0f32, 2.5, -3.0, 4.0];
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let built = [
        compress_residual_delta(&x),
        choose_compression(&x, false),
        choose_compression(&[], false),
        compress_original(
            &bytes,
            "float32",
            &[1.0, 2.0, -3.0, 4.5],
            ResidualCompressionMethod::Lzma,
        )
        .unwrap(),
    ];
    assert_eq!(
        built.len(),
        WRITERS.iter().filter(|w| w.2 == Stamped).count() + 1
    );
    for rd in built {
        assert_eq!(rd.metadata.semantics, Some(SEMANTICS_ID));
        let b = rd.to_bytes();
        let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        let h = std::str::from_utf8(&b[4..4 + n]).unwrap();
        assert!(h.contains("\"semantics\":"), "{h}");
    }
}

#[test]
fn the_crate_container_writers_call_the_core_crates_writers() {
    // their bytes are the core crate's residual containers, whose identifier
    // is checked by alice-zip's tests/residual_semantics_id.rs
    let (_, files) = writers_in_source();
    let mut checked = 0;
    for (file, name, decision) in WRITERS {
        if decision != CrateContainer {
            continue;
        }
        let text = &files.iter().find(|(f, _)| f == file).unwrap().1;
        let at = fn_at(text, name);
        let body: String = text[at..].lines().take(40).collect::<Vec<_>>().join("\n");
        // compress_all_subbands goes through compress_subband
        assert!(
            body.contains("compress_residual_") || body.contains("compress_subband("),
            "{file} {name}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[test]
fn the_literal_writers_regenerate_nothing_from_a_law() {
    let (_, files) = writers_in_source();
    let mut checked = 0;
    for (file, name, decision) in WRITERS {
        if decision != Literal {
            continue;
        }
        let text = &files.iter().find(|(f, _)| f == file).unwrap().1;
        let at = fn_at(text, name);
        let body: String = text[at..].lines().take(80).collect::<Vec<_>>().join("\n");
        for law in ["generators::", "reconstruct(", "generate_"] {
            assert!(!body.contains(law), "{file} {name} calls {law}");
        }
        checked += 1;
    }
    assert_eq!(checked, 8);
}
