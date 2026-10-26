//! Byte-stream compression wrappers and the `.alice` residual container.
//!
//! Thin adapters over [`flate2`](https://crates.io/crates/flate2) (zlib, always
//! with `std`) and, behind the `lzma` feature, over
//! [`lzma-rs`](https://crates.io/crates/lzma-rs) (LZMA) plus the quantised /
//! lossless residual containers that the `.alice` file format and
//! `alice-edge` coefficient batches persist. Every function here has a single
//! implementation for the whole ecosystem (`libalice` re-exports this module).
//!
//! # Residual container formats
//!
//! All multi-byte fields are **little-endian**. `codec` is
//! [`ResidualCodec`]'s tag (0 = deflate, 1 = LZMA) and is recorded in the
//! container, so a reader never has to guess which compressor produced the
//! payload and the default can change without orphaning what is on disk.
//!
//! | container | layout |
//! |-----------|--------|
//! | quantised (`compress_residual_quantized`) | `0xFB` · `codec: u8` · `semantics: [u8; 32]` · `bits: u8` (8 or 16) · `min: f64` · `scale: f64` · `len: u32` · codec(quantised bytes) |
//! | lossless (`compress_residual_lossless`) | `0xFA` · `codec: u8` · `semantics: [u8; 32]` · `len: u32` · codec(`f32` LE samples) |
//! | xor (`compress_residual_xor`) | `0xF9` · `codec: u8` · `semantics: [u8; 32]` · `len: u32` · codec(`u32` LE `bits ^ bits`) |
//!
//! # The semantics identifier
//!
//! A residual is only meaningful against the model it was taken from, and the
//! model's samples depend on the arithmetic that evaluated it. Every container
//! records `semantics`, the identifier the model was evaluated under
//! ([`crate::law::SEMANTICS_ID`] for a model evaluated by this crate's laws).
//! A reader compares the **stored** identifier with
//! [`crate::law::check_semantics`]: its own identifier or a verified earlier
//! one ([`ResidualStatus::Verified`] / [`ResidualStatus::VerifiedPrevious`])
//! decodes; any other is refused with [`ResidualIdError::Mismatch`], which
//! names both identifiers, before anything is decoded.
//!
//! # Containers written before the identifier
//!
//! They still decode, and [`residual_container_status`] reports them as
//! [`ResidualStatus::Legacy`]; the `*_strict` decoders refuse them with
//! [`ResidualIdError::Legacy`]. They are: `0xFC` · `codec` · `bits` · `min` ·
//! `scale` · `len` · .. (quantised), `0xFD` · `codec` · `len` · .. (lossless)
//! and `0xFE` · `codec` · `len` · .. (xor), written from 0.8.0 to 0.10; and,
//! from before the codec byte existed, `0xFF` · `len: u32` · LZMA(`f32` LE)
//! for the lossless container and `bits: u8` · `min: f64` · `scale: f64` ·
//! `len: u32` · LZMA(..) for the quantised one (reading either needs the
//! `lzma` feature). Containers written from 0.11.0 onwards need 0.11.0 or
//! later to read.
//!
//! # Example
//!
//! ```
//! use alice_zip::compression::{zlib_compress, zlib_decompress};
//!
//! let payload = b"the quick brown fox jumps over the lazy dog".repeat(4);
//! let compressed = zlib_compress(&payload, 6).unwrap();
//! let decompressed = zlib_decompress(&compressed).unwrap();
//! assert_eq!(payload, decompressed);
//! ```

use core::fmt;

use flate2::read::{DeflateDecoder, ZlibDecoder};
use flate2::write::{DeflateEncoder, ZlibEncoder};
use flate2::Compression;
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};

pub use crate::quantize::{dequantize_16bit, dequantize_8bit, quantize_16bit, quantize_8bit};

/// Compress `data` with zlib (deflate + zlib wrapper) at the given level.
///
/// `level` follows the standard zlib range `0..=9` (0 = store, 9 = max
/// compression, 6 = default). Values outside the range are clamped.
///
/// # Errors
///
/// Returns [`io::Error`] on internal encoder failure (usually memory).
pub fn zlib_compress(data: &[u8], level: u32) -> io::Result<Vec<u8>> {
    let level = level.min(9);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data)?;
    encoder.finish()
}

/// Decompress zlib-formatted `data` produced by [`zlib_compress`] (or any
/// zlib-compliant encoder).
///
/// # Errors
///
/// Returns [`io::Error`] if the input is not valid zlib format or the payload
/// is truncated.
pub fn zlib_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------- lzma

/// Compress `data` with LZMA (`.lzma` legacy stream, `lzma-rs` fixed settings;
/// `_preset` is accepted for API compatibility and ignored)
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
#[cfg(feature = "lzma")]
pub fn lzma_compress(data: &[u8], _preset: u32) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_compress(&mut io::Cursor::new(data), &mut output)
        .map_err(|e| io::Error::other(format!("LZMA compress error: {e}")))?;
    Ok(output)
}

/// Decompress an LZMA stream produced by [`lzma_compress`]
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when `data` is not a valid LZMA stream
#[cfg(feature = "lzma")]
pub fn lzma_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    lzma_rs::lzma_decompress(&mut io::Cursor::new(data), &mut output).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("LZMA decompress error: {e}"),
        )
    })?;
    Ok(output)
}

// ------------------------------------------------- codec bytes identifier

/// Domain of [`deflate_bytes_id`] and `lzma_bytes_id`
pub const CODEC_BYTES_DOMAIN: &[u8] = b"alice-zip/codec-bytes/v1";

/// The bytes the deflate paths ([`zlib_compress`] and the residual containers' raw deflate)
/// produce for the probe inputs ([`codec_probe_inputs`] at [`CODEC_PROBE_LEVELS`]), folded
/// into one identifier
///
/// Decoding stays compatible across encoder versions; the encoded bytes need not. A caller
/// that keeps a compressed size or compressed bytes as a fact (a capacity counted in
/// compressed bytes, a digest of a compressed file) depends on the encoder giving the same
/// bytes on every target and with the `flate2` / `miniz_oxide` versions and features its own
/// build resolves. [`check_codec_bytes`] recomputes this value in the caller's build.
pub const DEFLATE_BYTES_ID: [u8; 32] = [
    0x06, 0x59, 0xce, 0x52, 0xa0, 0xec, 0x10, 0x4a, 0x35, 0x08, 0x64, 0x05, 0xa7, 0xab, 0x83, 0x73,
    0x65, 0xb0, 0x6e, 0xc0, 0x08, 0x8b, 0xe0, 0xbc, 0x61, 0xc3, 0x35, 0xe3, 0xd9, 0x95, 0xbe, 0x7d,
];

/// The bytes `lzma_compress` produces for the probe inputs, folded into one identifier
/// (see [`DEFLATE_BYTES_ID`])
///
/// Defined in every build; `lzma_bytes_id` and its check need the `lzma` feature
pub const LZMA_BYTES_ID: [u8; 32] = [
    0xa9, 0x39, 0xb1, 0xff, 0x04, 0x75, 0x45, 0xb8, 0xa5, 0x7d, 0x77, 0x4e, 0xca, 0x0c, 0xe5, 0xdc,
    0x3d, 0x6a, 0x3d, 0x28, 0x0f, 0xd5, 0x74, 0xdd, 0x68, 0x97, 0xdd, 0xd7, 0x15, 0x0d, 0x29, 0xbd,
];

/// The deflate levels of [`DEFLATE_BYTES_ID`] (each for zlib and for raw deflate)
pub const CODEC_PROBE_LEVELS: [u32; 4] = [0, 1, 6, 9];

/// The probe inputs of the codec identifiers, as `(name, bytes)` in the order they are folded
///
/// Built from integer arithmetic and literals only, so they are the same bytes everywhere.
#[must_use]
pub fn codec_probe_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut lcg = Vec::with_capacity(65_536);
    let mut x: u32 = 0x1234_5678;
    for _ in 0..65_536 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        lcg.extend_from_slice(&[(x >> 24) as u8]);
    }
    let ramp: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    vec![
        ("empty", Vec::new()),
        ("one-byte", b"a".to_vec()),
        (
            "english",
            b"the quick brown fox jumps over the lazy dog. ".repeat(64),
        ),
        (
            "japanese",
            "準備で手が絵の具だらけになった。\n".repeat(48).into_bytes(),
        ),
        ("ramp", ramp),
        ("lcg-64k", lcg),
    ]
}

fn fold_bytes(h: &mut Sha256, b: &[u8]) {
    h.update((b.len() as u64).to_be_bytes());
    h.update(b);
}

/// The outputs folded into [`DEFLATE_BYTES_ID`], in order, as `(label, compressed bytes)`
/// with labels `zlib/<level>/<probe>` and `deflate/<level>/<probe>`
///
/// # Errors
///
/// The encoder's error
pub fn deflate_probe_outputs() -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    for (name, bytes) in codec_probe_inputs() {
        for level in CODEC_PROBE_LEVELS {
            let zlib = zlib_compress(&bytes, level)?;
            let raw = codec_compress(&bytes, ResidualCodec::Deflate, level)?;
            out.extend([
                (format!("zlib/{level}/{name}"), zlib),
                (format!("deflate/{level}/{name}"), raw),
            ]);
        }
    }
    Ok(out)
}

/// The outputs folded into [`LZMA_BYTES_ID`], in order, as `(lzma/<probe>, compressed bytes)`
///
/// # Errors
///
/// The encoder's error
#[cfg(feature = "lzma")]
pub fn lzma_probe_outputs() -> io::Result<Vec<(String, Vec<u8>)>> {
    codec_probe_inputs()
        .into_iter()
        .map(|(name, bytes)| Ok((format!("lzma/{name}"), lzma_compress(&bytes, 6)?)))
        .collect()
}

/// Folds `(label, bytes)` outputs under `codec` into an identifier
///
/// Every field is length-prefixed (`u64` big-endian), so moving bytes between labels and
/// outputs, or between two outputs, gives another identifier.
#[must_use]
pub fn fold_codec_outputs(codec: &[u8], outputs: &[(String, Vec<u8>)]) -> [u8; 32] {
    let mut h = Sha256::new();
    fold_bytes(&mut h, CODEC_BYTES_DOMAIN);
    fold_bytes(&mut h, codec);
    h.update((outputs.len() as u64).to_be_bytes());
    for (label, bytes) in outputs {
        fold_bytes(&mut h, label.as_bytes());
        fold_bytes(&mut h, bytes);
    }
    h.finalize().into()
}

/// [`DEFLATE_BYTES_ID`] as this build's encoder computes it
///
/// # Errors
///
/// The encoder's error
pub fn deflate_bytes_id() -> io::Result<[u8; 32]> {
    Ok(fold_codec_outputs(b"deflate", &deflate_probe_outputs()?))
}

/// [`LZMA_BYTES_ID`] as this build's encoder computes it
///
/// # Errors
///
/// The encoder's error
#[cfg(feature = "lzma")]
pub fn lzma_bytes_id() -> io::Result<[u8; 32]> {
    Ok(fold_codec_outputs(b"lzma", &lzma_probe_outputs()?))
}

/// This build's encoder does not give the recorded bytes, or failed
#[derive(Debug)]
#[non_exhaustive]
pub enum CodecBytesError {
    /// The identifier this build computed differs from the recorded one
    Mismatch {
        /// `"deflate"` or `"lzma"`
        codec: &'static str,
        /// The recorded identifier
        expected: [u8; 32],
        /// The identifier this build computed
        actual: [u8; 32],
    },
    /// The encoder failed on a probe input
    Encode(io::Error),
}

impl fmt::Display for CodecBytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch {
                codec,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{codec} encoder bytes differ from the recorded ones: recorded "
                )?;
                for b in expected {
                    write!(f, "{b:02x}")?;
                }
                f.write_str(", this build ")?;
                for b in actual {
                    write!(f, "{b:02x}")?;
                }
                Ok(())
            }
            Self::Encode(e) => write!(f, "encoder failed on a probe input: {e}"),
        }
    }
}

impl std::error::Error for CodecBytesError {}

fn check_codec(
    codec: &'static str,
    expected: [u8; 32],
    actual: io::Result<[u8; 32]>,
) -> Result<&'static str, CodecBytesError> {
    let actual = actual.map_err(CodecBytesError::Encode)?;
    if actual == expected {
        Ok(codec)
    } else {
        Err(CodecBytesError::Mismatch {
            codec,
            expected,
            actual,
        })
    }
}

/// Checks that this build's encoders give the recorded bytes: [`DEFLATE_BYTES_ID`], and
/// [`LZMA_BYTES_ID`] with the `lzma` feature
///
/// A caller that treats compressed sizes or bytes as facts runs this in its own tests on
/// every target it ships, since the encoder versions and features are those of its build.
///
/// # Errors
///
/// Returns the codecs it compared, in order (`["deflate"]`, or `["deflate", "lzma"]` with the
/// `lzma` feature), so a caller can assert that its check compared what it ships
///
/// # Errors
///
/// [`CodecBytesError::Mismatch`] naming both identifiers, or the encoder's error
pub fn check_codec_bytes() -> Result<Vec<&'static str>, CodecBytesError> {
    let mut compared = Vec::new();
    compared.extend([check_codec(
        "deflate",
        DEFLATE_BYTES_ID,
        deflate_bytes_id(),
    )?]);
    #[cfg(feature = "lzma")]
    {
        compared.extend([check_codec("lzma", LZMA_BYTES_ID, lzma_bytes_id())?]);
    }
    Ok(compared)
}

#[cfg(test)]
mod codec_bytes_tests {
    use super::*;

    #[test]
    fn check_codec_names_both_identifiers_on_a_mismatch() {
        let actual = deflate_bytes_id().unwrap();
        let mut expected = actual;
        expected[31] ^= 1;
        let e = check_codec("deflate", expected, Ok(actual)).unwrap_err();
        let text = e.to_string();
        let hex = |b: &[u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert!(matches!(
            e,
            CodecBytesError::Mismatch {
                codec: "deflate",
                ..
            }
        ));
        assert!(
            text.contains(&hex(&expected)) && text.contains(&hex(&actual)),
            "{text}"
        );
        assert!(check_codec("deflate", actual, Ok(actual)).is_ok());
    }
}

// ------------------------------------------------- residual containers

/// Which compressor a residual container's payload went through.
///
/// The byte is stored in the container, so a reader never has to guess and the
/// default can change without orphaning what is already on disk.
///
/// ⚠️ [`Self::Lzma`] is `lzma-rs`, which is pure Rust but has a weak encoder:
/// measured on a 100,000-sample sine residual it produced **46,916 bytes where
/// [`Self::Deflate`] produced 8,430** (5.6x worse), and on the raw samples
/// 316,528 against zlib's 8,742 (36x worse). It is kept so that containers
/// written by earlier versions keep decoding, and for callers who need an LZMA
/// stream specifically. New data should use the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResidualCodec {
    /// Raw deflate via `flate2` (pure-Rust `miniz_oxide` backend) — the default
    Deflate,
    /// LZMA via `lzma-rs` (`lzma` feature) — legacy, see the type's note
    Lzma,
}

impl ResidualCodec {
    const DEFLATE_TAG: u8 = 0;
    const LZMA_TAG: u8 = 1;

    const fn tag(self) -> u8 {
        match self {
            Self::Deflate => Self::DEFLATE_TAG,
            Self::Lzma => Self::LZMA_TAG,
        }
    }

    fn from_tag(tag: u8) -> io::Result<Self> {
        match tag {
            Self::DEFLATE_TAG => Ok(Self::Deflate),
            Self::LZMA_TAG => Ok(Self::Lzma),
            other => Err(invalid(format!(
                "unknown residual codec tag {other} (this container was written \
                 by a newer version of alice-zip)"
            ))),
        }
    }
}

/// The codec new containers are written with
///
/// Deflate, because it is pure Rust, already a dependency, and measurably
/// stronger than the LZMA implementation this crate can reach without a C
/// toolchain (see [`ResidualCodec`]).
#[must_use]
pub const fn residual_codec_default() -> ResidualCodec {
    ResidualCodec::Deflate
}

/// Marker of the quantised container, version 1 (codec-tagged)
///
/// Version 0 wrote the caller's `bits` argument as its first byte, so the two
/// versions are told apart by that byte: `0xFC` means v1, anything else means
/// v0.
///
/// ⚠️ The one ambiguous case is a v0 container written with `bits == 0xFC`.
/// Nothing in this repository wrote one — the only widths the dequantiser
/// distinguishes are 8 and 16, every caller passes one of those, and the v1
/// writer normalises the value — but a v0 container built by hand with that
/// width would be misread. There is no byte the v0 layout reserves, so this
/// cannot be ruled out by the format; it is stated rather than hidden.
const QUANTIZED_V1_MARKER: u8 = 0xFC;
/// Marker of the lossless (subtraction) container, version 1
const LOSSLESS_V1_MARKER: u8 = 0xFD;
/// Marker of the xor container
const XOR_MARKER: u8 = 0xFE;
/// Marker of the lossless container, version 0 (LZMA payload, no codec byte)
const LOSSLESS_V0_MARKER: u8 = 0xFF;
/// Marker of the quantised container with the semantics identifier
///
/// Like [`QUANTIZED_V1_MARKER`], told apart from a version 0 container by its
/// first byte; a version 0 container written with `bits == 0xFB` would be
/// misread in the same way.
const QUANTIZED_V2_MARKER: u8 = 0xFB;
/// Marker of the lossless container with the semantics identifier
const LOSSLESS_V2_MARKER: u8 = 0xFA;
/// Marker of the xor container with the semantics identifier
const XOR_V2_MARKER: u8 = 0xF9;
/// Length of the semantics identifier the current containers carry
const SEMANTICS_LEN: usize = 32;

/// `marker · codec · len: u32`
const V1_HEADER: usize = 1 + 1 + 4;
/// Extra bytes the quantised v1 container carries between codec and length:
/// `bits: u8 · min: f64 · scale: f64`
const QUANTIZED_V1_EXTRA: usize = 1 + 8 + 8;
/// v0 lossless: `0xFF · len: u32`
const LOSSLESS_V0_HEADER: usize = 1 + 4;
/// v0 quantised: `bits · min: f64 · scale: f64 · len: u32`
const QUANTIZED_V0_HEADER: usize = 1 + 8 + 8 + 4;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// How a residual container identifies the arithmetic of its model (see the
/// module documentation)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResidualStatus {
    /// The stored identifier is this build's ([`crate::law::SEMANTICS_ID`]).
    Verified,
    /// The stored identifier is an earlier one this build reproduces bit for
    /// bit ([`crate::law::PREVIOUS_SEMANTICS_IDS`]).
    VerifiedPrevious([u8; 32]),
    /// Written before containers carried an identifier (alice-zip 0.10 and
    /// earlier).
    Legacy,
}

/// Why a residual container's identifier was refused; carried inside the
/// [`io::Error`] the decoders return (`io::Error::get_ref` and
/// `downcast_ref::<ResidualIdError>()` recover it)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResidualIdError {
    /// The stored identifier is neither this build's nor a verified earlier
    /// one.
    Mismatch {
        /// The identifier in the container.
        stored: [u8; 32],
        /// This build's identifier ([`crate::law::SEMANTICS_ID`]).
        running: [u8; 32],
    },
    /// A strict decoder was given a container written before the identifier.
    Legacy,
}

impl fmt::Display for ResidualIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { stored, running } => {
                write!(f, "residual container: model evaluated under semantics ")?;
                hex(f, stored)?;
                write!(f, ", this build reproduces ")?;
                hex(f, running)?;
                write!(f, " and its verified predecessors only")
            }
            Self::Legacy => write!(
                f,
                "residual container written before the semantics identifier, refused by a strict read"
            ),
        }
    }
}

fn hex(f: &mut fmt::Formatter<'_>, bytes: &[u8; 32]) -> fmt::Result {
    for b in bytes {
        write!(f, "{b:02x}")?;
    }
    Ok(())
}

impl std::error::Error for ResidualIdError {}

/// The status of a stored identifier, or the mismatch error
fn status_of(stored: &[u8]) -> io::Result<ResidualStatus> {
    let mut id = [0u8; SEMANTICS_LEN];
    id.copy_from_slice(&stored[..SEMANTICS_LEN]);
    match crate::law::check_semantics(&id) {
        alice_det_math::SemanticsCheck::Verified => Ok(ResidualStatus::Verified),
        alice_det_math::SemanticsCheck::VerifiedPrevious(p) => {
            Ok(ResidualStatus::VerifiedPrevious(p))
        }
        alice_det_math::SemanticsCheck::Mismatch => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            ResidualIdError::Mismatch {
                stored: id,
                running: crate::law::SEMANTICS_ID,
            },
        )),
    }
}

/// How the residual container `data` identifies the arithmetic of its model,
/// read from the header without decoding the payload
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when `data` is empty or too short for its
/// header, or carries a marker this version does not know, and
/// [`ResidualIdError::Mismatch`] (inside the [`io::Error`]) when the stored
/// identifier is not one this build reproduces
pub fn residual_container_status(data: &[u8]) -> io::Result<ResidualStatus> {
    match data.first() {
        None => Err(invalid("empty container")),
        Some(&QUANTIZED_V2_MARKER | &LOSSLESS_V2_MARKER | &XOR_V2_MARKER) => {
            if data.len() < 2 + SEMANTICS_LEN {
                return Err(invalid("container too short for its semantics identifier"));
            }
            status_of(&data[2..2 + SEMANTICS_LEN])
        }
        Some(&QUANTIZED_V1_MARKER | &LOSSLESS_V1_MARKER | &XOR_MARKER | &LOSSLESS_V0_MARKER)
        | Some(&8 | &16) => Ok(ResidualStatus::Legacy),
        Some(other) => Err(invalid(format!("unknown container marker 0x{other:02X}"))),
    }
}

/// `data`'s status, refusing [`ResidualStatus::Legacy`]
fn strict(data: &[u8]) -> io::Result<()> {
    if residual_container_status(data)? == ResidualStatus::Legacy {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            ResidualIdError::Legacy,
        ));
    }
    Ok(())
}

/// `semantics · extra`, the bytes the current containers put between the codec
/// byte and the length
fn with_semantics(extra: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SEMANTICS_LEN + extra.len());
    out.extend_from_slice(&crate::law::SEMANTICS_ID);
    out.extend_from_slice(extra);
    out
}

/// [`unpack_v1`] for a container with the semantics identifier: checks the
/// identifier first and returns `(codec, extra after the identifier, payload)`
fn unpack_v2(
    data: &[u8],
    marker: u8,
    extra_len: usize,
) -> io::Result<(ResidualCodec, &[u8], &[u8])> {
    let (codec, extra, payload) = unpack_v1(data, marker, SEMANTICS_LEN + extra_len)?;
    status_of(&extra[..SEMANTICS_LEN])?;
    Ok((codec, &extra[SEMANTICS_LEN..], payload))
}

/// Compress `payload` with `codec`
///
/// `level` is the deflate level `0..=9` (values above 9 are clamped to 9;
/// **0 means store**, as in [`zlib_compress`]); the LZMA path has fixed
/// settings and ignores it.
fn codec_compress(payload: &[u8], codec: ResidualCodec, level: u32) -> io::Result<Vec<u8>> {
    match codec {
        ResidualCodec::Deflate => {
            // `min(9)` で揃える — `zlib_compress` も同じで、level 0 は store
            // ⚠️ 以前は `clamp(1, 9)` で floor していたので **level 0 が黙って 1 に
            //    上がり**、同 module の 2 本の公開 API が同じ引数に別の意味を与えて
            //    いた (実測: 400,000 byte に対し `zlib_compress(.., 0)` は 400,071 byte
            //    = store なのに、容器の level 0 は 31,607 byte = level 1 と byte 一致)
            //    「無圧縮を頼んだのに圧縮される」形なので floor を外した
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level.min(9)));
            encoder.write_all(payload)?;
            encoder.finish()
        }
        #[cfg(feature = "lzma")]
        ResidualCodec::Lzma => lzma_compress(payload, level),
        #[cfg(not(feature = "lzma"))]
        ResidualCodec::Lzma => Err(invalid(
            "this container needs the `lzma` feature (payload is an LZMA stream)",
        )),
    }
}

fn codec_decompress(payload: &[u8], codec: ResidualCodec) -> io::Result<Vec<u8>> {
    match codec {
        ResidualCodec::Deflate => {
            let mut out = Vec::new();
            DeflateDecoder::new(payload).read_to_end(&mut out)?;
            Ok(out)
        }
        #[cfg(feature = "lzma")]
        ResidualCodec::Lzma => lzma_decompress(payload),
        #[cfg(not(feature = "lzma"))]
        ResidualCodec::Lzma => Err(invalid(
            "this container needs the `lzma` feature (payload is an LZMA stream)",
        )),
    }
}

/// Write `marker · codec · len: u32 · payload`
fn pack_v1(marker: u8, codec: ResidualCodec, payload: &[u8], extra: &[u8]) -> io::Result<Vec<u8>> {
    let len =
        u32::try_from(payload.len()).map_err(|_| invalid("compressed payload exceeds 4 GiB"))?;
    let mut out = Vec::with_capacity(V1_HEADER + extra.len() + payload.len());
    out.push(marker);
    out.push(codec.tag());
    out.extend_from_slice(extra);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// Read back the `marker · codec · [extra] · len: u32` header and return
/// `(codec, payload)`
fn unpack_v1(
    data: &[u8],
    marker: u8,
    extra_len: usize,
) -> io::Result<(ResidualCodec, &[u8], &[u8])> {
    let header = V1_HEADER + extra_len;
    if data.len() < header {
        return Err(invalid(format!(
            "container too short: need at least {header} bytes, got {}",
            data.len()
        )));
    }
    if data[0] != marker {
        return Err(invalid(format!(
            "wrong container marker: expected 0x{marker:02X}, got 0x{:02X}",
            data[0]
        )));
    }
    let codec = ResidualCodec::from_tag(data[1])?;
    let extra = &data[2..2 + extra_len];
    let len_at = 2 + extra_len;
    let payload_len = u32::from_le_bytes(
        data[len_at..len_at + 4]
            .try_into()
            .map_err(|_| invalid("payload length"))?,
    ) as usize;
    let end = header
        .checked_add(payload_len)
        .ok_or_else(|| invalid("payload length overflow"))?;
    if data.len() < end {
        return Err(invalid(format!(
            "container truncated: expected {end} bytes, got {}",
            data.len()
        )));
    }
    Ok((codec, extra, &data[header..end]))
}

/// The codec a residual container was written with
///
/// Works for every container this module produces. Version 0 containers
/// (written before the codec byte existed) report [`ResidualCodec::Lzma`],
/// which is what they are.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when `data` is too short or carries a marker
/// or codec tag this version does not know
pub fn residual_container_codec(data: &[u8]) -> io::Result<ResidualCodec> {
    match data.first() {
        None => Err(invalid("empty container")),
        Some(&LOSSLESS_V0_MARKER) => Ok(ResidualCodec::Lzma),
        Some(&8 | &16) => Ok(ResidualCodec::Lzma), // v0 quantised: first byte is `bits`
        Some(
            &QUANTIZED_V1_MARKER | &LOSSLESS_V1_MARKER | &XOR_MARKER | &QUANTIZED_V2_MARKER
            | &LOSSLESS_V2_MARKER | &XOR_V2_MARKER,
        ) => {
            let tag = *data
                .get(1)
                .ok_or_else(|| invalid("container has no codec byte"))?;
            ResidualCodec::from_tag(tag)
        }
        Some(other) => Err(invalid(format!("unknown container marker 0x{other:02X}"))),
    }
}

/// Quantise `residual` to `bits` (16, or 8 for any other value) and compress it
/// into the quantised residual container with the default codec
///
/// This container is **lossy**: the quantisation step discards bits before the
/// payload is compressed. Use [`compress_residual_xor`] when the samples have
/// to come back unchanged.
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
pub fn compress_residual_quantized(residual: &[f32], bits: u8, level: u32) -> io::Result<Vec<u8>> {
    compress_residual_quantized_with(residual, bits, residual_codec_default(), level)
}

/// [`compress_residual_quantized`] with an explicit codec
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure, or when `codec` needs a Cargo
/// feature that is not enabled
pub fn compress_residual_quantized_with(
    residual: &[f32],
    bits: u8,
    codec: ResidualCodec,
    level: u32,
) -> io::Result<Vec<u8>> {
    let (quantized, min_val, scale) = if bits == 16 {
        quantize_16bit(residual)
    } else {
        quantize_8bit(residual)
    };
    let payload = codec_compress(&quantized, codec, level)?;
    let mut extra = Vec::with_capacity(QUANTIZED_V1_EXTRA);
    extra.push(if bits == 16 { 16 } else { 8 });
    extra.extend_from_slice(&min_val.to_le_bytes());
    extra.extend_from_slice(&scale.to_le_bytes());
    debug_assert_eq!(extra.len(), QUANTIZED_V1_EXTRA);
    pack_v1(
        QUANTIZED_V2_MARKER,
        codec,
        &payload,
        &with_semantics(&extra),
    )
}

/// [`decompress_residual_quantized`], refusing a container written before the
/// semantics identifier ([`ResidualIdError::Legacy`])
///
/// # Errors
///
/// As [`decompress_residual_quantized`], and [`ResidualIdError::Legacy`]
/// (inside the [`io::Error`])
pub fn decompress_residual_quantized_strict(data: &[u8]) -> io::Result<Vec<f32>> {
    strict(data)?;
    decompress_residual_quantized(data)
}

/// Inverse of [`compress_residual_quantized`]
///
/// Accepts the current container and the ones earlier releases wrote
/// ([`ResidualStatus::Legacy`]: `0xFC`, and version 0 whose first byte is 8 or
/// 16 with an LZMA payload); [`residual_container_status`] tells them apart.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the header is truncated, the declared
/// payload length exceeds `data`, or the payload is not a valid stream for the
/// codec the header names; [`ResidualIdError::Mismatch`] (inside the
/// [`io::Error`]) when the stored semantics identifier is not one this build
/// reproduces
pub fn decompress_residual_quantized(data: &[u8]) -> io::Result<Vec<f32>> {
    let (bits, min_val, scale, quantized) = match data.first() {
        Some(&QUANTIZED_V2_MARKER) => {
            let (codec, extra, payload) = unpack_v2(data, QUANTIZED_V2_MARKER, QUANTIZED_V1_EXTRA)?;
            let bits = extra[0];
            let min_val = f64::from_le_bytes(extra[1..9].try_into().map_err(|_| invalid("min"))?);
            let scale = f64::from_le_bytes(extra[9..17].try_into().map_err(|_| invalid("scale"))?);
            (bits, min_val, scale, codec_decompress(payload, codec)?)
        }
        Some(&QUANTIZED_V1_MARKER) => {
            let (codec, extra, payload) = unpack_v1(data, QUANTIZED_V1_MARKER, QUANTIZED_V1_EXTRA)?;
            let bits = extra[0];
            let min_val = f64::from_le_bytes(extra[1..9].try_into().map_err(|_| invalid("min"))?);
            let scale = f64::from_le_bytes(extra[9..17].try_into().map_err(|_| invalid("scale"))?);
            (bits, min_val, scale, codec_decompress(payload, codec)?)
        }
        _ => decompress_quantized_v0(data)?,
    };
    Ok(if bits == 16 {
        dequantize_16bit(&quantized, min_val, scale)
    } else {
        dequantize_8bit(&quantized, min_val, scale)
    })
}

/// Version 0 quantised container: `bits · min: f64 · scale: f64 · len: u32 · LZMA(..)`
fn decompress_quantized_v0(data: &[u8]) -> io::Result<(u8, f64, f64, Vec<u8>)> {
    if data.len() < QUANTIZED_V0_HEADER {
        return Err(invalid(format!(
            "Data too short for residual header (need at least {QUANTIZED_V0_HEADER} bytes)"
        )));
    }
    let bits = data[0];
    let min_val = f64::from_le_bytes(data[1..9].try_into().map_err(|_| invalid("min_val"))?);
    let scale = f64::from_le_bytes(data[9..17].try_into().map_err(|_| invalid("scale"))?);
    let compressed_len = u32::from_le_bytes(
        data[17..21]
            .try_into()
            .map_err(|_| invalid("compressed_len"))?,
    ) as usize;
    let end = QUANTIZED_V0_HEADER
        .checked_add(compressed_len)
        .ok_or_else(|| invalid("compressed_len overflow"))?;
    if data.len() < end {
        return Err(invalid(format!(
            "Data truncated: expected {end} bytes, got {}",
            data.len()
        )));
    }
    let quantized = codec_decompress(&data[QUANTIZED_V0_HEADER..end], ResidualCodec::Lzma)?;
    Ok((bits, min_val, scale, quantized))
}

/// Compress the raw little-endian `f32` samples of `residual` with the default
/// codec
///
/// ⚠️ This container stores the residual array itself exactly, but a *pipeline*
/// built on it is **not** bit-exact: the caller computes `residual = original -
/// model` in `f32`, and where `|original| << |model|` that subtraction rounds
/// the original away, so `model + residual` does not return it. Measured on a
/// 100,000-sample sine fitted by [`crate::generators::analyze_signal`], 99
/// samples — every zero crossing — came back different.
/// [`compress_residual_xor`] has no such gap and is what a lossless pipeline
/// should use.
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure
pub fn compress_residual_lossless(residual: &[f32], level: u32) -> io::Result<Vec<u8>> {
    compress_residual_lossless_with(residual, residual_codec_default(), level)
}

/// [`compress_residual_lossless`] with an explicit codec
///
/// # Errors
///
/// Returns [`io::Error`] on encoder failure, or when `codec` needs a Cargo
/// feature that is not enabled
pub fn compress_residual_lossless_with(
    residual: &[f32],
    codec: ResidualCodec,
    level: u32,
) -> io::Result<Vec<u8>> {
    let bytes: Vec<u8> = residual.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let payload = codec_compress(&bytes, codec, level)?;
    pack_v1(LOSSLESS_V2_MARKER, codec, &payload, &with_semantics(&[]))
}

/// [`decompress_residual_lossless`], refusing a container written before the
/// semantics identifier ([`ResidualIdError::Legacy`])
///
/// # Errors
///
/// As [`decompress_residual_lossless`], and [`ResidualIdError::Legacy`]
/// (inside the [`io::Error`])
pub fn decompress_residual_lossless_strict(data: &[u8]) -> io::Result<Vec<f32>> {
    strict(data)?;
    decompress_residual_lossless(data)
}

/// Inverse of [`compress_residual_lossless`]
///
/// Accepts the current container and the ones earlier releases wrote
/// ([`ResidualStatus::Legacy`]: `0xFD`, and `0xFF` with an LZMA payload and no
/// codec byte); [`residual_container_status`] tells them apart.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the marker is unknown, the header or
/// payload is truncated, or the payload is not a valid stream for the codec the
/// header names; [`ResidualIdError::Mismatch`] (inside the [`io::Error`]) when
/// the stored semantics identifier is not one this build reproduces
pub fn decompress_residual_lossless(data: &[u8]) -> io::Result<Vec<f32>> {
    let bytes = match data.first() {
        None => return Err(invalid("empty container")),
        Some(&LOSSLESS_V2_MARKER) => {
            let (codec, _, payload) = unpack_v2(data, LOSSLESS_V2_MARKER, 0)?;
            codec_decompress(payload, codec)?
        }
        Some(&LOSSLESS_V1_MARKER) => {
            let (codec, _, payload) = unpack_v1(data, LOSSLESS_V1_MARKER, 0)?;
            codec_decompress(payload, codec)?
        }
        Some(&LOSSLESS_V0_MARKER) => {
            if data.len() < LOSSLESS_V0_HEADER {
                return Err(invalid(format!(
                    "Data too short for lossless header (need at least \
                     {LOSSLESS_V0_HEADER} bytes)"
                )));
            }
            let compressed_len = u32::from_le_bytes(
                data[1..5]
                    .try_into()
                    .map_err(|_| invalid("compressed_len"))?,
            ) as usize;
            let end = LOSSLESS_V0_HEADER
                .checked_add(compressed_len)
                .ok_or_else(|| invalid("compressed_len overflow"))?;
            if data.len() < end {
                return Err(invalid(format!(
                    "Data truncated: expected {end} bytes, got {}",
                    data.len()
                )));
            }
            codec_decompress(&data[LOSSLESS_V0_HEADER..end], ResidualCodec::Lzma)?
        }
        Some(other) => {
            return Err(invalid(format!(
                "Invalid lossless marker: expected 0x{LOSSLESS_V2_MARKER:02X}, \
                 0x{LOSSLESS_V1_MARKER:02X} or 0x{LOSSLESS_V0_MARKER:02X}, got 0x{other:02X}"
            )))
        }
    };
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "payload is {} bytes, not a whole number of f32 samples",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Store `original` as the **bit pattern xor** against `model` — exact for
/// every finite input
///
/// `original[i].to_bits() ^ model[i].to_bits()` is reversible by construction,
/// so [`decompress_residual_xor`] returns the original samples bit for bit:
/// signed zeros, denormals, and values far smaller than their model all
/// survive, where `original - model` in `f32` loses them. ⚠️ **Every** bit
/// pattern survives, not only the finite ones — infinities and NaN round trip
/// too, since nothing about `bits ^ bits` looks at what the bits mean.
///
/// ⚠️ It buys exactness, not size. When the model is close to the signal the
/// high bits agree and the xor is mostly zero bytes, which is usually smaller
/// than the subtraction residual; on a degree-3 polynomial, and on short
/// signals, it is larger. Pick by whether the samples have to come back
/// unchanged.
///
/// `model` must have the same length as `original`, and the same `model` must
/// be passed to the decoder — it is the other half of the data.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the lengths differ, or [`io::Error`] on
/// encoder failure
pub fn compress_residual_xor(original: &[f32], model: &[f32], level: u32) -> io::Result<Vec<u8>> {
    compress_residual_xor_with(original, model, residual_codec_default(), level)
}

/// [`compress_residual_xor`] with an explicit codec
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the lengths differ, or [`io::Error`] on
/// encoder failure / a codec whose Cargo feature is not enabled
pub fn compress_residual_xor_with(
    original: &[f32],
    model: &[f32],
    codec: ResidualCodec,
    level: u32,
) -> io::Result<Vec<u8>> {
    if original.len() != model.len() {
        return Err(invalid(format!(
            "model has {} samples but the signal has {}",
            model.len(),
            original.len()
        )));
    }
    let bytes: Vec<u8> = original
        .iter()
        .zip(model.iter())
        .flat_map(|(o, m)| (o.to_bits() ^ m.to_bits()).to_le_bytes())
        .collect();
    let payload = codec_compress(&bytes, codec, level)?;
    pack_v1(XOR_V2_MARKER, codec, &payload, &with_semantics(&[]))
}

/// [`decompress_residual_xor`], refusing a container written before the
/// semantics identifier ([`ResidualIdError::Legacy`])
///
/// # Errors
///
/// As [`decompress_residual_xor`], and [`ResidualIdError::Legacy`] (inside the
/// [`io::Error`])
pub fn decompress_residual_xor_strict(data: &[u8], model: &[f32]) -> io::Result<Vec<f32>> {
    strict(data)?;
    decompress_residual_xor(data, model)
}

/// Inverse of [`compress_residual_xor`] — needs the same `model`, evaluated
/// under an identifier this build reproduces (see the module documentation)
///
/// Accepts the current container and the `0xFE` one 0.8.0 to 0.10 wrote
/// ([`ResidualStatus::Legacy`]).
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] when the marker is wrong, the container is
/// truncated, the payload is not a valid stream, or the sample count does not
/// match `model`; [`ResidualIdError::Mismatch`] (inside the [`io::Error`])
/// when the stored semantics identifier is not one this build reproduces,
/// checked before the payload is decoded
pub fn decompress_residual_xor(data: &[u8], model: &[f32]) -> io::Result<Vec<f32>> {
    let (codec, _, payload) = match data.first() {
        Some(&XOR_V2_MARKER) => unpack_v2(data, XOR_V2_MARKER, 0)?,
        _ => unpack_v1(data, XOR_MARKER, 0)?,
    };
    let bytes = codec_decompress(payload, codec)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "payload is {} bytes, not a whole number of samples",
            bytes.len()
        )));
    }
    if bytes.len() / 4 != model.len() {
        return Err(invalid(format!(
            "container holds {} samples but the model has {}",
            bytes.len() / 4,
            model.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .zip(model.iter())
        .map(|(c, m)| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ m.to_bits()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(8);
        let compressed = zlib_compress(&payload, 6).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(payload, decompressed);
        assert!(
            compressed.len() < payload.len(),
            "compression should shrink"
        );
    }

    #[test]
    fn roundtrip_empty() {
        let compressed = zlib_compress(&[], 6).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn roundtrip_binary() {
        let payload: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let compressed = zlib_compress(&payload, 9).unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(payload, decompressed);
    }

    #[test]
    fn level_clamping() {
        let payload = b"hello world";
        let over_max = zlib_compress(payload, 42).unwrap();
        let normal = zlib_compress(payload, 9).unwrap();
        assert_eq!(over_max, normal, "level > 9 should clamp to 9");
    }

    #[test]
    fn level_zero_stores() {
        let payload = b"abcdef".repeat(16);
        let stored = zlib_compress(&payload, 0).unwrap();
        let decompressed = zlib_decompress(&stored).unwrap();
        assert_eq!(payload, decompressed);
    }

    #[test]
    fn decompress_invalid_returns_error() {
        let garbage = vec![0xff, 0xfe, 0xfd, 0xfc];
        assert!(zlib_decompress(&garbage).is_err());
    }

    #[cfg(feature = "lzma")]
    mod lzma {
        use super::super::*;

        #[test]
        fn lzma_roundtrip_and_invalid() {
            let payload = b"alice alice alice alice".repeat(32);
            let c = lzma_compress(&payload, 6).unwrap();
            assert!(c.len() < payload.len());
            assert_eq!(lzma_decompress(&c).unwrap(), payload);
            assert!(lzma_decompress(b"not lzma").is_err());
            assert_eq!(
                lzma_decompress(&lzma_compress(&[], 6).unwrap()).unwrap(),
                Vec::<u8>::new()
            );
        }

        #[test]
        fn quantized_container_roundtrip_both_widths() {
            let residual: Vec<f32> = (0..300)
                .map(|i| alice_det_math::sin(i as f32 * 0.37) * 5.0)
                .collect();
            for bits in [8u8, 16, 0, 255] {
                let c = compress_residual_quantized(&residual, bits, 6).unwrap();
                // layout: marker, codec, semantics, then the normalised width.
                // Any `bits` other than 16 means 8, and the container says so
                // instead of echoing whatever the caller passed.
                assert_eq!(c[0], QUANTIZED_V2_MARKER);
                assert_eq!(c[1], residual_codec_default().tag());
                assert_eq!(c[2..34], crate::law::SEMANTICS_ID);
                assert_eq!(c[34], if bits == 16 { 16 } else { 8 });
                let out = decompress_residual_quantized(&c).unwrap();
                assert_eq!(out.len(), residual.len());
                let tol = if bits == 16 {
                    10.0 / 65535.0
                } else {
                    10.0 / 255.0
                };
                for (a, b) in out.iter().zip(&residual) {
                    assert!((a - b).abs() <= tol / 2.0 + 1e-5, "bits={bits}: {a} vs {b}");
                }
            }
            assert!(decompress_residual_quantized(&[16; 20]).is_err());
            let mut truncated = compress_residual_quantized(&residual, 8, 6).unwrap();
            truncated.truncate(60);
            assert!(decompress_residual_quantized(&truncated).is_err());
        }

        #[test]
        fn lossless_container_is_bit_exact() {
            let residual: Vec<f32> = vec![0.0, -0.0, 1.5e-30, f32::MAX, -7.25, 3.0e10];
            let c = compress_residual_lossless(&residual, 6).unwrap();
            assert_eq!(c[0], LOSSLESS_V2_MARKER);
            let out = decompress_residual_lossless(&c).unwrap();
            assert_eq!(out.len(), residual.len());
            for (a, b) in out.iter().zip(&residual) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert!(decompress_residual_lossless(&[0x00, 0, 0, 0, 0]).is_err());
            assert!(decompress_residual_lossless(&[0xFF, 9, 0, 0, 0, 1]).is_err());
        }
    }
}
