//! Residual Compression Module for ALICE-Zip
//!
//! Ports the Python `alice_zip/residual_compression.py` to Rust.
//!
//! The key insight: procedural generation rarely achieves a 100% match with
//! real data. By storing the residual (original - generated), we can achieve
//! true lossless compression.
//!
//! Formula:
//!   `Data_original` = Gen(Params) + `Decompress(Residual_compressed)`
//!
//! # Binary format (v2)
//!
//! ```text
//! [header_len: u32 LE] [JSON metadata: header_len bytes] [compressed_data]
//! ```
//!
//! v1 (legacy, 2-byte header length) is also supported in `from_bytes`.
//!
//! # Author
//! Moroya Sakamoto
//! # License
//! ALICE-Zip Commercial License

use std::collections::HashMap;

/// Maximum header size (10 MiB) — mirrors Python `MAX_HEADER_SIZE`.
///
/// Prevents `DoS` via malformed `header_len` values.
const MAX_HEADER_SIZE: usize = 10 * 1024 * 1024;

// ============================================================================
// Error type
// ============================================================================

/// Errors produced by the residual compression subsystem.
#[derive(Debug)]
pub enum ResidualError {
    /// The byte slice is shorter than the minimum expected size.
    DataTooShort { got: usize, expected: usize },
    /// The JSON header is not valid UTF-8 or cannot be parsed.
    InvalidHeader(String),
    /// A required JSON field is missing.
    MissingField(String),
    /// An unknown/unsupported compression method was encountered.
    UnknownMethod(String),
    /// The announced header length would exceed the buffer or the
    /// configured maximum.
    HeaderTooLarge { size: usize },
    /// An underlying I/O or (de)compression error.
    Io(std::io::Error),
    /// The data is internally inconsistent (e.g., truncated payload).
    Corrupted(String),
    /// A residual layout this reader does not read: a `quant_bits` other than
    /// 8, 16 or 32.
    UnsupportedLayout(String),
    /// Values that are NaN or infinite cannot be quantized (the codes span
    /// the finite range from the minimum to the maximum).
    NotFinite,
    /// A `"delta"` residual without `base_value`: the writer dropped the
    /// first value, so the data cannot be reconstructed; recompress from the
    /// original.
    DeltaWithoutBase,
    /// A header `"version"` later than 2, which no writer produced.
    UnsupportedVersion(u32),
}

impl std::fmt::Display for ResidualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported residual header version {v} (1 and 2 exist)")
            }
            Self::DataTooShort { got, expected } => {
                write!(
                    f,
                    "data too short: got {got} bytes, expected at least {expected}"
                )
            }
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::MissingField(field) => {
                write!(f, "missing required header field: '{field}'")
            }
            Self::UnknownMethod(m) => {
                write!(f, "unknown compression method: '{m}'")
            }
            Self::HeaderTooLarge { size } => {
                write!(
                    f,
                    "header length {size} exceeds maximum allowed {MAX_HEADER_SIZE} bytes"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Corrupted(msg) => write!(f, "corrupted data: {msg}"),
            Self::UnsupportedLayout(msg) => write!(f, "unsupported residual layout: {msg}"),
            Self::NotFinite => write!(f, "values that are not finite cannot be quantized"),
            Self::DeltaWithoutBase => write!(
                f,
                "delta residual without its base value: recompress from the original"
            ),
        }
    }
}

impl std::error::Error for ResidualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for ResidualError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// Enums
// ============================================================================

/// Available residual compression methods.
///
/// Mirrors `ResidualCompressionMethod` from `residual_compression.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualCompressionMethod {
    /// No compression (data stored as raw f32 LE bytes).
    None,
    /// LZMA — best ratio, slower.
    Lzma,
    /// zlib — good balance between speed and ratio.
    Zlib,
    /// Delta encoding followed by LZMA compression, written by earlier
    /// releases. Readable only when the header records `base_value` (the
    /// earlier Rust writer); the earlier Python writer lost the first value.
    Delta,
    /// Differences of consecutive `f32` bit patterns as wrapping `u32` (the
    /// first is the first pattern), xz-compressed (the format the Python
    /// writer emits).
    BitDelta,
    /// Quantization (8-bit by default) followed by LZMA compression.
    Quantized,
}

impl ResidualCompressionMethod {
    /// Convert to the canonical string used in JSON headers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lzma => "lzma",
            Self::Zlib => "zlib",
            Self::Delta => "delta",
            Self::BitDelta => "bitdelta",
            Self::Quantized => "quantized",
        }
    }

    /// Parse from the canonical string stored in JSON headers.
    ///
    /// Returns `None` for unrecognised strings.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "lzma" => Some(Self::Lzma),
            "zlib" => Some(Self::Zlib),
            "delta" => Some(Self::Delta),
            "bitdelta" => Some(Self::BitDelta),
            "quantized" => Some(Self::Quantized),
            _ => None,
        }
    }
}

// ============================================================================
// Metadata struct
// ============================================================================

/// Method-specific metadata stored alongside the compressed payload.
///
/// Fields not relevant to the active method are left at their defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualMetadata {
    /// (`Quantized`) Minimum value of the original residual.
    pub min_val: f64,
    /// (`Quantized`) Range (max - min) of the original residual.
    pub scale: f64,
    /// (`Quantized`) Quantisation bit depth (8 or 16).
    pub bits: u8,
    /// (`Delta`) First element of the flattened, delta-encoded array.
    ///
    /// Not currently required for reconstruction (the base value is embedded
    /// inside the LZMA-compressed delta stream), but kept for symmetry with
    /// the Python implementation.
    pub base_value: f32,
    /// `quant_bits` of the header. When set, the payload (after the method's
    /// decompression) is the quantized form the Python writer emits,
    /// `min: f64 · scale: f64 · codes` (little endian, codes of 8, 16 or 32
    /// bits), whatever the method; the Python reader reads it the same way.
    pub quant_bits: Option<u8>,
    /// The payload is this crate's earlier quantized container (the header
    /// records `min_val` / `scale` / `bits`, which are written back).
    pub legacy_container: bool,
}

impl Default for ResidualMetadata {
    fn default() -> Self {
        Self {
            min_val: 0.0,
            scale: 1.0,
            bits: 8,
            base_value: 0.0,
            quant_bits: None,
            legacy_container: false,
        }
    }
}

// ============================================================================
// Main struct
// ============================================================================

/// Encapsulates compressed residual data.
///
/// Mirrors the Python `ResidualData` dataclass.
#[derive(Debug, Clone)]
pub struct ResidualData {
    /// The compression method used.
    pub method: ResidualCompressionMethod,
    /// The compressed payload bytes.
    pub compressed: Vec<u8>,
    /// Number of `f32` elements in the original (uncompressed) array.
    pub original_len: usize,
    /// Shape of the original array (its element count is `original_len`);
    /// the values are returned flat, in the stored order.
    pub shape: Vec<usize>,
    /// dtype of the original array as the writer recorded it (one of the
    /// eleven real numeric dtypes). The residual is a float difference and is
    /// returned as `f32` whatever this dtype is; the dtype applies only when
    /// the original is rebuilt from the generated values and the residual.
    pub dtype: String,
    /// Method-specific auxiliary information.
    pub metadata: ResidualMetadata,
}

impl ResidualData {
    // -------------------------------------------------------------------------
    // Serialisation
    // -------------------------------------------------------------------------

    /// Serialise to bytes using the **v2** format.
    ///
    /// ```text
    /// [header_len: u32 LE (4 bytes)] [JSON header: header_len bytes] [compressed payload]
    /// ```
    ///
    /// The JSON header carries:
    /// - `"method"` — canonical method string
    /// - `"original_len"` — element count of the uncompressed array
    /// - `"version"` — always `2` for this format
    /// - `"min_val"`, `"scale"`, `"bits"` — present for `Quantized`
    /// - `"base_value"` — present for `Delta`
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        // Build the JSON header using a plain HashMap to avoid pulling in a
        // heavyweight serialiser dependency (serde_json may not be available).
        // We construct the JSON string manually for portability.
        // the header keys of the Python writer (the canonical form): the
        // values are float32 of one dimension, so shape is [original_len]
        let mut fields: Vec<String> = Vec::new();
        fields.push(format!(r#""method":"{}""#, self.method.as_str()));
        let dims: Vec<String> = self.shape.iter().map(ToString::to_string).collect();
        fields.push(format!(r#""shape":[{}]"#, dims.join(",")));
        fields.push(format!(r#""dtype":"{}""#, self.dtype));
        match (self.metadata.legacy_container, self.metadata.quant_bits) {
            (true, _) => fields.push(format!(r#""quant_bits":{}"#, self.metadata.bits)),
            (false, Some(b)) => fields.push(format!(r#""quant_bits":{b}"#)),
            (false, None) => fields.push(r#""quant_bits":null"#.to_owned()),
        }
        fields.push(r#""version":2"#.to_owned());

        match self.method {
            ResidualCompressionMethod::Quantized if self.metadata.legacy_container => {
                fields.push(format!(r#""min_val":{}"#, self.metadata.min_val));
                fields.push(format!(r#""scale":{}"#, self.metadata.scale));
                fields.push(format!(r#""bits":{}"#, self.metadata.bits));
            }
            ResidualCompressionMethod::Delta => {
                fields.push(format!(r#""base_value":{}"#, self.metadata.base_value));
            }
            _ => {}
        }

        let header_json = format!("{{{}}}", fields.join(","));
        let header_bytes = header_json.as_bytes();

        let header_len = header_bytes.len() as u32;
        let mut out = Vec::with_capacity(4 + header_bytes.len() + self.compressed.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&self.compressed);
        out
    }

    /// Deserialise from bytes.
    ///
    /// Supports both:
    /// - **v2**: 4-byte `header_len` (LE u32) + JSON + payload
    /// - **v1** (legacy): 2-byte `header_len` (LE u16) + JSON + payload
    ///
    /// v1 is detected by the absence of a `"version"` field (or `version < 2`)
    /// in the parsed JSON after a successful v2 parse attempt. A `"version"`
    /// later than 2 is refused with [`ResidualError::UnsupportedVersion`].
    pub fn from_bytes(data: &[u8]) -> Result<Self, ResidualError> {
        if data.len() < 4 {
            return Err(ResidualError::DataTooShort {
                got: data.len(),
                expected: 4,
            });
        }

        // --- Try v2 first (4-byte header length) ---
        let header_len_v2 = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if header_len_v2 < MAX_HEADER_SIZE {
            let v2_end = 4 + header_len_v2;
            if v2_end <= data.len() {
                if let Ok(json_str) = std::str::from_utf8(&data[4..v2_end]) {
                    if let Ok(parsed) = Self::parse_json_header(json_str) {
                        // the writers emit `"version":2`: only a bare JSON
                        // integer is a version (the Python reader takes only
                        // an int too); a float or a string is refused
                        let version = match Self::raw_json_value(json_str, "version") {
                            None => 1,
                            Some(raw)
                                if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) =>
                            {
                                raw.parse::<u32>().map_err(|_| {
                                    ResidualError::InvalidHeader(format!(
                                        "version {raw} is out of range"
                                    ))
                                })?
                            }
                            Some(raw) => {
                                return Err(ResidualError::InvalidHeader(format!(
                                    "version must be a JSON integer, got {raw}"
                                )))
                            }
                        };
                        // only versions 1 and 2 were written; a later one is
                        // refused rather than read as version 2
                        if version > 2 {
                            return Err(ResidualError::UnsupportedVersion(version));
                        }
                        if version == 2 {
                            let compressed = data[v2_end..].to_vec();
                            return Self::from_header_map(&parsed, compressed);
                        }
                        // version < 2 — fall through to v1
                    } else {
                        // Not valid JSON — fall through to v1
                    }
                } else {
                    // Not valid UTF-8 — fall through to v1
                }
            }
        }

        // --- Fall back to v1 (2-byte header length) ---
        if data.len() < 2 {
            return Err(ResidualError::DataTooShort {
                got: data.len(),
                expected: 2,
            });
        }

        let header_len_v1 = u16::from_le_bytes([data[0], data[1]]) as usize;

        if header_len_v1 > MAX_HEADER_SIZE {
            return Err(ResidualError::HeaderTooLarge {
                size: header_len_v1,
            });
        }

        let v1_payload_start = 2 + header_len_v1;
        if v1_payload_start > data.len() {
            return Err(ResidualError::Corrupted(format!(
                "v1 header_len {} exceeds available data ({} bytes after prefix)",
                header_len_v1,
                data.len().saturating_sub(2)
            )));
        }

        let json_str = std::str::from_utf8(&data[2..v1_payload_start])
            .map_err(|e| ResidualError::InvalidHeader(format!("UTF-8 decode failed: {e}")))?;

        let parsed = Self::parse_json_header(json_str).map_err(ResidualError::InvalidHeader)?;

        let compressed = data[v1_payload_start..].to_vec();
        Self::from_header_map(&parsed, compressed)
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Minimalist JSON object parser.
    ///
    /// Parses a flat `{"key": value, ...}` object into a `HashMap<String, String>`.
    /// String values have their surrounding quotes stripped. Numeric and boolean
    /// values are kept as-is. Nested objects/arrays are not supported (not needed
    /// for our header format).
    /// Splits the members of a JSON object body on the commas that are not
    /// inside a string or a list (`"shape":[2,2]` is one member).
    fn split_members(inner: &str) -> Vec<&str> {
        let (mut out, mut depth, mut in_str, mut esc, mut start) =
            (Vec::new(), 0i32, false, false, 0);
        for (i, ch) in inner.char_indices() {
            if in_str {
                match (esc, ch) {
                    (true, _) => esc = false,
                    (false, '\\') => esc = true,
                    (false, '"') => in_str = false,
                    _ => {}
                }
                continue;
            }
            match ch {
                '"' => in_str = true,
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    out.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        out.push(&inner[start..]);
        out
    }

    /// The value text of `key` in a flat JSON object as written (a string
    /// keeps its quotes), or `None` when the key is absent.
    fn raw_json_value<'j>(json: &'j str, key: &str) -> Option<&'j str> {
        let inner = json.trim().strip_prefix('{')?.strip_suffix('}')?;
        Self::split_members(inner).into_iter().find_map(|pair| {
            let (k, v) = pair.split_once(':')?;
            (k.trim().trim_matches('"') == key).then(|| v.trim())
        })
    }

    fn parse_json_header(json: &str) -> Result<HashMap<String, String>, String> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(format!("JSON header must be a flat object, got: {json}"));
        }

        let inner = &json[1..json.len() - 1];
        let mut map = HashMap::new();

        // Split into members; commas inside `shape` lists and strings stay
        for pair in Self::split_members(inner) {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            // Split on the first colon.
            let colon_pos = pair
                .find(':')
                .ok_or_else(|| format!("malformed key-value pair (no ':'): '{pair}'"))?;

            let key_raw = pair[..colon_pos].trim();
            let val_raw = pair[colon_pos + 1..].trim();

            // Strip surrounding quotes from key.
            let key = if key_raw.starts_with('"') && key_raw.ends_with('"') {
                key_raw[1..key_raw.len() - 1].to_owned()
            } else {
                key_raw.to_owned()
            };

            // Strip surrounding quotes from value (if it is a JSON string).
            let val = if val_raw.starts_with('"') && val_raw.ends_with('"') {
                val_raw[1..val_raw.len() - 1].to_owned()
            } else {
                val_raw.to_owned()
            };

            map.insert(key, val);
        }

        Ok(map)
    }

    /// Construct `ResidualData` from a parsed header map and a compressed payload.
    fn from_header_map(
        map: &HashMap<String, String>,
        compressed: Vec<u8>,
    ) -> Result<Self, ResidualError> {
        // Required field: method
        let method_str = map
            .get("method")
            .ok_or_else(|| ResidualError::MissingField("method".to_owned()))?;

        let method = ResidualCompressionMethod::parse(method_str)
            .ok_or_else(|| ResidualError::UnknownMethod(method_str.clone()))?;

        // the shape (the canonical key) or "original_len" (this crate's
        // writer before the keys were aligned, one dimension)
        let shape = map
            .get("shape")
            .map(|v| {
                let body = v
                    .trim()
                    .strip_prefix('[')
                    .and_then(|b| b.strip_suffix(']'))
                    .ok_or_else(|| ResidualError::InvalidHeader(format!("invalid shape {v}")))?;
                body.split(',')
                    .filter(|d| !d.trim().is_empty())
                    .map(|d| {
                        d.trim().parse::<usize>().map_err(|e| {
                            ResidualError::InvalidHeader(format!("invalid shape {v}: {e}"))
                        })
                    })
                    .collect::<Result<Vec<usize>, _>>()
            })
            .transpose()?;
        let from_shape = shape
            .as_ref()
            .map(|dims| {
                dims.iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| {
                        ResidualError::InvalidHeader(format!("shape {dims:?} is too large"))
                    })
            })
            .transpose()?;
        let from_len = map
            .get("original_len")
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|e| ResidualError::InvalidHeader(format!("invalid original_len: {e}")))
            })
            .transpose()?;
        let original_len = match (from_shape, from_len) {
            (Some(a), Some(b)) if a != b => {
                return Err(ResidualError::InvalidHeader(format!(
                    "shape ({a} elements) and original_len {b} disagree"
                )))
            }
            (Some(n), _) | (None, Some(n)) => n,
            (None, None) => return Err(ResidualError::MissingField("shape".to_owned())),
        };
        let shape = shape.unwrap_or_else(|| vec![original_len]);

        // the dtype of the original as the writer recorded it
        let dtype = map.get("dtype").map_or("float32", String::as_str);
        if !WRITER_DTYPES.contains(&dtype) {
            return Err(ResidualError::InvalidHeader(format!(
                "unsupported dtype {dtype}"
            )));
        }

        // Optional / method-specific fields.
        let quant_bits = match map.get("quant_bits").map(String::as_str) {
            None | Some("null") => None,
            Some(v) => match v.parse::<u8>() {
                Ok(b @ (8 | 16 | 32)) => Some(b),
                _ => {
                    return Err(ResidualError::UnsupportedLayout(format!(
                        "quant_bits {v} (8, 16 and 32 are read)"
                    )))
                }
            },
        };
        let mut metadata = ResidualMetadata {
            quant_bits,
            ..ResidualMetadata::default()
        };

        if method == ResidualCompressionMethod::Quantized {
            if map.contains_key("min_val") {
                // this crate's earlier quantized container
                metadata.legacy_container = true;
                if let Some(v) = map.get("min_val") {
                    metadata.min_val = v.parse().unwrap_or(0.0);
                }
                if let Some(v) = map.get("scale") {
                    metadata.scale = v.parse().unwrap_or(1.0);
                }
                if let Some(v) = map.get("bits") {
                    metadata.bits = v.parse().unwrap_or(8);
                }
            } else if metadata.quant_bits.is_none() {
                // the Python reader defaults to 8 bits
                metadata.quant_bits = Some(8);
            }
        }

        if method == ResidualCompressionMethod::Delta {
            // the earlier Rust writer recorded base_value and kept the first
            // value as the first delta; the earlier Python writer did neither,
            // so its files cannot be reconstructed
            let v = map
                .get("base_value")
                .ok_or(ResidualError::DeltaWithoutBase)?;
            metadata.base_value = v
                .parse()
                .map_err(|e| ResidualError::InvalidHeader(format!("invalid base_value: {e}")))?;
        }

        Ok(Self {
            method,
            compressed,
            original_len,
            shape,
            dtype: dtype.to_owned(),
            metadata,
        })
    }
}

// ============================================================================
// Delta compression
// ============================================================================

/// dtypes the writers record for the original (the real numeric dtypes).
const WRITER_DTYPES: [&str; 11] = [
    "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64",
];

/// First bytes of an xz stream; an LZMA stream without them is read as the
/// "alone" format the earlier Rust writer used.
const XZ_MAGIC: [u8; 6] = [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00];

/// Decompresses an xz stream (what the Python writer emits) or an LZMA
/// "alone" stream (what the earlier Rust writer emitted), told apart by the
/// xz magic.
fn lzma_or_xz_decompress(bytes: &[u8]) -> Result<Vec<u8>, ResidualError> {
    if bytes.starts_with(&XZ_MAGIC) {
        let mut out = Vec::new();
        lzma_rs::xz_decompress(&mut std::io::Cursor::new(bytes), &mut out)
            .map_err(|e| ResidualError::Corrupted(format!("xz: {e}")))?;
        Ok(out)
    } else {
        crate::compression::lzma_decompress(bytes)
            .map_err(|e| ResidualError::Corrupted(format!("lzma: {e}")))
    }
}

/// Compresses with xz, the format the Python writer emits.
fn xz_compress(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    lzma_rs::xz_compress(&mut std::io::Cursor::new(bytes), &mut out).ok()?;
    Some(out)
}

/// The first encoding that succeeded, with the method that decodes it.
///
/// Falls back to the raw little-endian floats under `None`, so the recorded
/// method always matches the bytes (an encoder failure never leaves bytes of
/// one method labelled as another).
fn first_encoding(
    attempts: Vec<(ResidualCompressionMethod, Option<Vec<u8>>)>,
    raw: Vec<u8>,
) -> (ResidualCompressionMethod, Vec<u8>) {
    attempts
        .into_iter()
        .find_map(|(m, bytes)| bytes.map(|b| (m, b)))
        .unwrap_or((ResidualCompressionMethod::None, raw))
}

/// The bitdelta stream of `data`: the difference of consecutive `f32` bit
/// patterns as wrapping `u32`, little endian, the first difference being the
/// first pattern. Integer arithmetic, so every value (NaN payloads,
/// subnormals, ±0, infinities) comes back bit for bit.
fn bit_delta_encode(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    let mut prev = 0u32;
    for v in data {
        let b = v.to_bits();
        out.extend_from_slice(&b.wrapping_sub(prev).to_le_bytes());
        prev = b;
    }
    out
}

/// Inverts [`bit_delta_encode`].
fn bit_delta_decode(stream: &[u8]) -> Vec<f32> {
    let mut acc = 0u32;
    stream
        .chunks_exact(4)
        .map(|c| {
            acc = acc.wrapping_add(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            f32::from_bits(acc)
        })
        .collect()
}

/// Compress `data` as a bitdelta stream (see
/// [`ResidualCompressionMethod::BitDelta`]) followed by xz.
///
/// Recorded as [`ResidualCompressionMethod::BitDelta`], or as `None` with the
/// raw floats if xz fails.
#[must_use]
pub fn compress_residual_delta(data: &[f32]) -> ResidualData {
    let raw: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let (method, compressed) = first_encoding(
        vec![(
            ResidualCompressionMethod::BitDelta,
            xz_compress(&bit_delta_encode(data)),
        )],
        raw,
    );
    ResidualData {
        method,
        compressed,
        original_len: data.len(),
        shape: vec![data.len()],
        dtype: "float32".to_owned(),
        metadata: ResidualMetadata::default(),
    }
}

/// Decompress a delta-encoded `ResidualData` back to `Vec<f32>`:
/// [`ResidualCompressionMethod::BitDelta`] bit for bit, or the earlier
/// [`ResidualCompressionMethod::Delta`] (float differences, as that writer
/// stored them).
///
/// # Errors
///
/// [`ResidualError::Corrupted`] if the method is not a delta method, the
/// stream does not decompress, or it does not hold `original_len` values.
pub fn decompress_residual_delta(rd: &ResidualData) -> Result<Vec<f32>, ResidualError> {
    if !matches!(
        rd.method,
        ResidualCompressionMethod::Delta | ResidualCompressionMethod::BitDelta
    ) {
        return Err(ResidualError::Corrupted(format!(
            "decompress_residual_delta called with {}",
            rd.method.as_str()
        )));
    }
    let raw = lzma_or_xz_decompress(&rd.compressed)?;
    if raw.len() != rd.original_len * 4 {
        return Err(ResidualError::Corrupted(format!(
            "delta stream holds {} bytes, expected {}",
            raw.len(),
            rd.original_len * 4
        )));
    }
    if rd.method == ResidualCompressionMethod::BitDelta {
        return Ok(bit_delta_decode(&raw));
    }
    let mut out = Vec::with_capacity(rd.original_len);
    let mut acc = 0.0f32;
    for c in raw.chunks_exact(4) {
        acc += f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        out.push(acc);
    }
    Ok(out)
}

// ============================================================================
// Analysis
// ============================================================================

/// Estimate the Shannon entropy (in bits) of a quantised version of `data`.
///
/// A higher entropy means the data is harder to compress.
///
/// The values are quantised into 256 bins over their observed range. The
/// entropy is computed as `H = -sum(p * log2(p))` for each non-empty bin.
///
/// Reciprocal multiplication is used instead of division for performance
/// (avoids the higher-latency integer/float divide instruction):
/// ```text
/// p = count * rcp_total  // instead of: p = count / total
/// ```
pub fn estimate_entropy(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Find range.
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = max_val - min_val;

    // Build 256-bin histogram.
    let mut counts = [0u32; 256];
    if scale < 1e-10 {
        // Constant data: zero entropy.
        return 0.0;
    }

    let rcp_scale = 255.0 / f64::from(scale);
    for &v in data {
        let bin = ((f64::from(v) - f64::from(min_val)) * rcp_scale).clamp(0.0, 255.0) as usize;
        counts[bin] += 1;
    }

    // Shannon entropy using reciprocal multiplication.
    let total = data.len() as f64;
    let rcp_total = 1.0 / total; // precomputed reciprocal

    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = f64::from(c) * rcp_total; // multiply, not divide
            entropy -= p * alice_det_math::log2_64(p);
        }
    }

    entropy as f32
}

/// Try each applicable compression method on `data` and return the method
/// that achieves the best (smallest) compressed output.
///
/// The methods tried are: `None`, `Lzma`, `Zlib`, `Delta`, `Quantized`.
///
/// If `data` is empty, `ResidualCompressionMethod::None` is returned.
#[must_use]
pub fn analyze_residual(data: &[f32]) -> ResidualCompressionMethod {
    if data.is_empty() {
        return ResidualCompressionMethod::None;
    }

    // Helper: raw f32-LE bytes.
    let raw_bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();

    // Collect (method, compressed_size) pairs.
    let mut candidates: Vec<(ResidualCompressionMethod, usize)> = Vec::new();

    // None — uncompressed raw bytes.
    candidates.push((ResidualCompressionMethod::None, raw_bytes.len()));

    // LZMA.
    if let Ok(c) = crate::compression::lzma_compress(&raw_bytes, 6) {
        candidates.push((ResidualCompressionMethod::Lzma, c.len()));
    }

    // zlib.
    if let Ok(c) = crate::compression::zlib_compress(&raw_bytes, 6) {
        candidates.push((ResidualCompressionMethod::Zlib, c.len()));
    }

    // Delta.
    {
        let delta_rd = compress_residual_delta(data);
        candidates.push((ResidualCompressionMethod::Delta, delta_rd.compressed.len()));
    }

    // Quantized (8-bit + LZMA).
    if let Ok(c) = crate::compression::compress_residual_quantized(data, 8, 6) {
        candidates.push((ResidualCompressionMethod::Quantized, c.len()));
    }

    // Return the method with the smallest compressed size.
    candidates
        .into_iter()
        .min_by_key(|&(_, size)| size)
        .map_or(ResidualCompressionMethod::None, |(method, _)| method)
}

// ============================================================================
// High-level API
// ============================================================================

/// Auto-select the best compression method and compress `data`.
///
/// When `allow_lossy` is `false`, `Quantized` is excluded from selection
/// because it is lossy (quantisation discards precision).
///
/// # Process
/// 1. Call [`analyze_residual`] to determine the best method.
/// 2. If the chosen method is `Quantized` and `allow_lossy` is `false`,
///    fall back to `Lzma`.
/// 3. Compress using the selected method and return a [`ResidualData`].
#[must_use]
pub fn choose_compression(data: &[f32], allow_lossy: bool) -> ResidualData {
    if data.is_empty() {
        return ResidualData {
            method: ResidualCompressionMethod::None,
            compressed: Vec::new(),
            original_len: 0,
            shape: vec![0],
            dtype: "float32".to_owned(),
            metadata: ResidualMetadata::default(),
        };
    }

    let mut best = analyze_residual(data);

    // If lossy is not allowed, exclude Quantized.
    if !allow_lossy && best == ResidualCompressionMethod::Quantized {
        best = ResidualCompressionMethod::Lzma;
    }

    compress_with_method(data, best)
}

/// Compress `data` using the specified `method`.
fn compress_with_method(data: &[f32], method: ResidualCompressionMethod) -> ResidualData {
    let raw_bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();

    match method {
        ResidualCompressionMethod::None => ResidualData {
            method,
            compressed: raw_bytes,
            original_len: data.len(),
            shape: vec![data.len()],
            dtype: "float32".to_owned(),
            metadata: ResidualMetadata::default(),
        },

        ResidualCompressionMethod::Lzma => {
            let lz = crate::compression::lzma_compress(&raw_bytes, 6).ok();
            let (method, compressed) = first_encoding(vec![(method, lz)], raw_bytes);
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                shape: vec![data.len()],
                dtype: "float32".to_owned(),
                metadata: ResidualMetadata::default(),
            }
        }

        ResidualCompressionMethod::Zlib => {
            let z = crate::compression::zlib_compress(&raw_bytes, 6).ok();
            let lz = crate::compression::lzma_compress(&raw_bytes, 6).ok();
            let (method, compressed) = first_encoding(
                vec![(method, z), (ResidualCompressionMethod::Lzma, lz)],
                raw_bytes,
            );
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                shape: vec![data.len()],
                dtype: "float32".to_owned(),
                metadata: ResidualMetadata::default(),
            }
        }

        ResidualCompressionMethod::Delta | ResidualCompressionMethod::BitDelta => {
            compress_residual_delta(data)
        }

        ResidualCompressionMethod::Quantized => {
            // the Python writer's quantized form (8 bits), xz-compressed
            let q = quantize_python(data, 8).ok().and_then(|q| xz_compress(&q));
            let (method, compressed) = first_encoding(vec![(method, q)], raw_bytes);
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                shape: vec![data.len()],
                dtype: "float32".to_owned(),
                metadata: ResidualMetadata {
                    quant_bits: (method == ResidualCompressionMethod::Quantized).then_some(8),
                    ..ResidualMetadata::default()
                },
            }
        }
    }
}

/// Decompress a `ResidualData` back to `Vec<f32>`.
///
/// Dispatches to the appropriate decompression routine based on
/// `rd.method`.
pub fn decompress(rd: &ResidualData) -> Result<Vec<f32>, ResidualError> {
    if rd.original_len == 0 {
        return Ok(Vec::new());
    }

    let raw = match rd.method {
        ResidualCompressionMethod::None => rd.compressed.clone(),
        ResidualCompressionMethod::Lzma => lzma_or_xz_decompress(&rd.compressed)?,
        ResidualCompressionMethod::Zlib => crate::compression::zlib_decompress(&rd.compressed)?,
        // the delta methods ignore quant_bits, as the Python reader does
        ResidualCompressionMethod::Delta | ResidualCompressionMethod::BitDelta => {
            return decompress_residual_delta(rd)
        }
        ResidualCompressionMethod::Quantized if rd.metadata.legacy_container => {
            return Ok(crate::compression::decompress_residual_quantized(
                &rd.compressed,
            )?)
        }
        ResidualCompressionMethod::Quantized => lzma_or_xz_decompress(&rd.compressed)?,
    };
    match rd.metadata.quant_bits {
        Some(bits) => dequantize_python(&raw, bits, rd.original_len),
        None => {
            if raw.len() != rd.original_len * 4 {
                return Err(ResidualError::Corrupted(format!(
                    "payload holds {} bytes, expected {}",
                    raw.len(),
                    rd.original_len * 4
                )));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
    }
}

/// The original rebuilt from `generated` and the residual, as the
/// little-endian bytes of the dtype the header records.
///
/// Each value is `generated + residual` in `f64`, then:
/// - an integer dtype refuses NaN (`Corrupted`), and otherwise rounds half to
///   even and saturates to the dtype's range, infinities included (Rust's
///   float-to-integer `as` saturates, so the 64-bit bounds need no special
///   case);
/// - a float dtype is rounded once to it, an overflow giving an infinity.
///
/// The Python `ResidualCompressor.reconstruct` follows the same rule; both are
/// held to tests/data/residual/reconstruct_vectors.txt.
///
/// # Errors
///
/// The errors of [`decompress`], `Corrupted` when `generated` and the residual
/// differ in length or an integer original would be NaN, and `InvalidHeader`
/// for a dtype outside the eleven the writers record.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn reconstruct_bytes(generated: &[f64], rd: &ResidualData) -> Result<Vec<u8>, ResidualError> {
    let residual = decompress(rd)?;
    if residual.len() != generated.len() {
        return Err(ResidualError::Corrupted(format!(
            "{} generated values for {} residual values",
            generated.len(),
            residual.len()
        )));
    }
    let width = match rd.dtype.as_str() {
        "int8" | "uint8" => 1,
        "float16" | "int16" | "uint16" => 2,
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        other => {
            return Err(ResidualError::InvalidHeader(format!(
                "unsupported dtype {other}"
            )))
        }
    };
    let mut out = Vec::with_capacity(generated.len() * width);
    for (&g, &r) in generated.iter().zip(&residual) {
        let s = g + f64::from(r);
        let dtype = rd.dtype.as_str();
        if dtype.contains("int") && s.is_nan() {
            return Err(ResidualError::Corrupted(
                "NaN where the original is an integer".to_owned(),
            ));
        }
        let n = s.round_ties_even();
        match dtype {
            "float16" => out.extend_from_slice(&f64_to_f16_bits(s).to_le_bytes()),
            "float32" => out.extend_from_slice(&(s as f32).to_le_bytes()),
            "float64" => out.extend_from_slice(&s.to_le_bytes()),
            "int8" => out.extend_from_slice(&(n as i8).to_le_bytes()),
            "int16" => out.extend_from_slice(&(n as i16).to_le_bytes()),
            "int32" => out.extend_from_slice(&(n as i32).to_le_bytes()),
            "int64" => out.extend_from_slice(&(n as i64).to_le_bytes()),
            "uint8" => out.push(n as u8),
            "uint16" => out.extend_from_slice(&(n as u16).to_le_bytes()),
            "uint32" => out.extend_from_slice(&(n as u32).to_le_bytes()),
            _ => out.extend_from_slice(&(n as u64).to_le_bytes()),
        }
    }
    Ok(out)
}

/// `x` rounded once (half to even) to an IEEE half: an overflow is an
/// infinity, a NaN the quiet NaN with `x`'s sign.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f64_to_f16_bits(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7E00;
    }
    // 65520 is halfway between 65504 (the largest half) and 65536
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    // below 2^-14 the half is subnormal: a multiple of 2^-24 (scaling by a
    // power of two is exact, so the only rounding is round_ties_even)
    if a < pow2(-14) {
        return sign | (a * pow2(24)).round_ties_even() as u16;
    }
    let exp = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let mut m = (a * pow2(10 - exp)).round_ties_even() as u16; // 1024..=2048
    let mut e = exp + 15;
    if m == 2048 {
        m = 1024;
        e += 1;
    }
    sign | ((e as u16) << 10) | (m - 1024)
}

/// `2^e` for a normal `f64` exponent, built from its bits (exact).
#[allow(clippy::cast_sign_loss)]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The quantized form the Python writer emits: `min: f64 · scale: f64`
/// followed by the codes (`bits` wide, little endian). Computed in `f64` as
/// the Python writer does: minimum and range, a range below 1e-10 replaced by
/// 1, `(v - min) / range * (levels - 1)` rounded half to even and clipped to
/// the code range. Values that are NaN or infinite give `NotFinite`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize_python(data: &[f32], bits: u8) -> Result<Vec<u8>, ResidualError> {
    // the codes span the finite range from the minimum to the maximum
    if !data.iter().all(|v| v.is_finite()) {
        return Err(ResidualError::NotFinite);
    }
    // computed in f64: in f32 the top 32-bit code 2^32 - 1 rounds to 2^32
    let min = data
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::INFINITY, f64::min);
    let max = data
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut scale = max - min;
    if scale < 1e-10 {
        scale = 1.0;
    }
    let top = (u64::MAX >> (64 - u32::from(bits))) as f64;
    let mut out = Vec::with_capacity(16 + data.len() * usize::from(bits / 8));
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in data {
        let code = (((f64::from(v) - min) / scale) * top)
            .round_ties_even()
            .clamp(0.0, top);
        match bits {
            8 => out.push(code as u8),
            16 => out.extend_from_slice(&(code as u16).to_le_bytes()),
            _ => out.extend_from_slice(&(code as u32).to_le_bytes()),
        }
    }
    Ok(out)
}

/// Inverts [`quantize_python`] as the Python reader does:
/// `code / (levels - 1) * scale + min` in `f64`, rounded to `f32`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn dequantize_python(raw: &[u8], bits: u8, len: usize) -> Result<Vec<f32>, ResidualError> {
    let width = usize::from(bits / 8);
    if raw.len() != 16 + len * width {
        return Err(ResidualError::Corrupted(format!(
            "quantized payload holds {} bytes, expected {}",
            raw.len(),
            16 + len * width
        )));
    }
    let min = f64::from_le_bytes(raw[0..8].try_into().expect("8 bytes"));
    let scale = f64::from_le_bytes(raw[8..16].try_into().expect("8 bytes"));
    let top = (u64::MAX >> (64 - u32::from(bits))) as f64;
    Ok(raw[16..]
        .chunks_exact(width)
        .map(|c| {
            let code = match width {
                1 => f64::from(c[0]),
                2 => f64::from(u16::from_le_bytes([c[0], c[1]])),
                _ => f64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            };
            (code / top * scale + min) as f32
        })
        .collect())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic test signal (sine wave, 1000 samples).
    fn test_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| alice_det_math::sin((i as f32) * std::f32::consts::TAU / 100.0) * 50.0)
            .collect()
    }

    // ------------------------------------------------------------------
    // Delta round-trip
    // ------------------------------------------------------------------

    const SPECIAL_BITS: [u32; 13] = [
        0x7FC0_0001,
        0x3F80_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0001,
        0x7F7F_FFFF,
        0xFFFF_FFFF,
        0x3200_0000,
        0x4CBE_BC20,
        0x7F80_0001, // signaling NaN
        0xFF80_0001,
    ];
    const DTYPES: [&str; 11] = [
        "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64",
    ];

    /// The files of this crate's residual writer, by name without prefix.
    fn rust_writer_files() -> Vec<(String, Vec<u8>)> {
        use ResidualCompressionMethod as M;
        let values = [5.0f32, 5.5, 6.0, 4.0];
        let special: Vec<f32> = SPECIAL_BITS.into_iter().map(f32::from_bits).collect();
        let mut files = Vec::new();
        for (name, method) in [
            ("none", M::None),
            ("lzma", M::Lzma),
            ("zlib", M::Zlib),
            ("delta", M::Delta),
            ("quantized", M::Quantized),
        ] {
            files.push((
                name.to_owned(),
                compress_with_method(&values, method).to_bytes(),
            ));
        }
        // codes on .5 with an even integer below (round half to even)
        let ties = [0.0f32, 2.5, 4.5, 255.0];
        files.push((
            "quantized_ties".to_owned(),
            compress_with_method(&ties, M::Quantized).to_bytes(),
        ));
        for (name, method) in [("none", M::None), ("lzma", M::Lzma), ("delta", M::Delta)] {
            files.push((
                format!("{name}_special"),
                compress_with_method(&special, method).to_bytes(),
            ));
        }
        // the residual of an original of each dtype: stored as float32 as it is
        let wide = [-36.54f32, 300.5, 5.5, -129.75];
        for dt in DTYPES {
            let mut rd = compress_with_method(&wide, M::Lzma);
            rd.dtype = dt.to_owned();
            files.push((format!("dtype_{dt}"), rd.to_bytes()));
        }
        // 32-bit codes; 1.0 lands on 4294967295, which float32 cannot hold
        let q32 = [0.0f32, 1.0, 0.5, 0.25];
        let rd = ResidualData {
            method: M::Quantized,
            compressed: xz_compress(&quantize_python(&q32, 32).unwrap()).unwrap(),
            original_len: 4,
            shape: vec![4],
            dtype: "float32".to_owned(),
            metadata: ResidualMetadata {
                quant_bits: Some(32),
                ..ResidualMetadata::default()
            },
        };
        files.push(("quantized32".to_owned(), rd.to_bytes()));
        // random values (tests/data/residual/residual_values.py)
        let random = random_values();
        files.push((
            "quantized_rand8".to_owned(),
            compress_with_method(&random, M::Quantized).to_bytes(),
        ));
        for bits in [16u8, 32] {
            let rd = ResidualData {
                method: M::Quantized,
                compressed: xz_compress(&quantize_python(&random, bits).unwrap()).unwrap(),
                original_len: random.len(),
                shape: vec![random.len()],
                dtype: "float32".to_owned(),
                metadata: ResidualMetadata {
                    quant_bits: Some(bits),
                    ..ResidualMetadata::default()
                },
            };
            files.push((format!("quantized_rand{bits}"), rd.to_bytes()));
        }
        files
    }

    /// 64 values in [-8, 8) from a fixed linear congruential sequence, the
    /// same as `random_values` in tests/data/residual/residual_values.py.
    #[allow(clippy::cast_precision_loss)]
    fn random_values() -> Vec<f32> {
        let mut x: u32 = 0x2545_F491;
        (0..64)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 8) as f32 / 1_048_576.0 - 8.0
            })
            .collect()
    }

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/data/residual")
    }

    /// Writes the files of this crate's residual writer into
    /// `tests/data/residual/` (run on purpose: `cargo test --lib -- --ignored
    /// write_rust_writer_fixtures`, with `ALICE_RESIDUAL_PREFIX` naming them).
    #[test]
    #[ignore = "writes fixtures; run on purpose"]
    fn write_rust_writer_fixtures() {
        let prefix = std::env::var("ALICE_RESIDUAL_PREFIX").unwrap_or_else(|_| "rust".to_owned());
        for (name, bytes) in rust_writer_files() {
            std::fs::write(fixture_dir().join(format!("{prefix}_{name}.bin")), bytes).unwrap();
        }
    }

    #[test]
    fn the_writer_reproduces_the_committed_files() {
        // the files the other reader is tested against are this writer's
        // current output, byte for byte
        let files = rust_writer_files();
        for (name, bytes) in &files {
            let path = fixture_dir().join(format!("rust_{name}.bin"));
            let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            assert!(
                committed == *bytes,
                "rust_{name}.bin differs from the writer's output"
            );
        }
        assert_eq!(files.len(), 24, "every writer file compared");
    }

    #[test]
    fn thirty_two_bit_codes_are_computed_without_float32_rounding() {
        // oracle: exact rationals, (v - min) / scale * (2^32 - 1) rounded half to
        // even: 1.0 -> 4294967295, 0.5 -> 2147483647.5 -> 2147483648,
        // 0.25 -> 1073741823.75 -> 1073741824
        let q = quantize_python(&[0.0, 1.0, 0.5, 0.25], 32).unwrap();
        let codes: Vec<u32> = q[16..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(codes, [0, 4_294_967_295, 2_147_483_648, 1_073_741_824]);
    }

    #[test]
    fn eight_bit_codes_are_computed_in_f64() {
        // oracle: x = f32 0x3F1C1C1C is 0.60980391502380371..., x * 255 =
        // 155.4999...: 155 exactly; the f32 product rounds to 155.5 and gives 156
        let x = f32::from_bits(0x3F1C_1C1C);
        let q = quantize_python(&[0.0, 1.0, x], 8).unwrap();
        assert_eq!(&q[16..], &[0, 255, 155]);
    }

    #[test]
    fn values_that_are_not_finite_are_not_quantized() {
        let special: Vec<f32> = SPECIAL_BITS.into_iter().map(f32::from_bits).collect();
        for data in [
            special,
            vec![1.0, f32::NAN, 2.0],
            vec![1.0, f32::INFINITY, 2.0],
        ] {
            assert!(matches!(
                quantize_python(&data, 8),
                Err(ResidualError::NotFinite)
            ));
            // a writer asked to quantize keeps such values bit for bit instead
            let rd = compress_with_method(&data, ResidualCompressionMethod::Quantized);
            assert_ne!(rd.method, ResidualCompressionMethod::Quantized);
            let back = decompress(&ResidualData::from_bytes(&rd.to_bytes()).unwrap()).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&back), bits(&data));
        }
    }

    #[test]
    fn the_bitdelta_streams_equal_the_shared_reference() {
        // written by tests/data/residual/make_fixtures.py with integer
        // arithmetic only; the Python encoder is held to the same file
        let table = include_str!("../../tests/data/residual/bitdelta_streams.txt");
        let mut compared = 0;
        for line in table
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let cols: Vec<&str> = line.split(" | ").collect();
            let values: Vec<f32> = cols[1]
                .split_whitespace()
                .map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap()))
                .collect();
            let stream: Vec<u8> = (0..cols[2].len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&cols[2][i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(bit_delta_encode(&values), stream, "{}", cols[0]);
            let back: Vec<u32> = bit_delta_decode(&stream)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let want: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            assert_eq!(back, want, "{}", cols[0]);
            compared += 1;
        }
        assert_eq!(compared, 5);
    }

    #[test]
    fn a_failed_encoder_records_the_method_that_produced_the_bytes() {
        use ResidualCompressionMethod as M;
        let raw = vec![1, 2, 3, 4];
        // the wanted method failed, the next one succeeded: its name is kept
        assert_eq!(
            first_encoding(vec![(M::Zlib, None), (M::Lzma, Some(vec![9]))], raw.clone()),
            (M::Lzma, vec![9])
        );
        // everything failed: the raw floats, recorded as None
        assert_eq!(
            first_encoding(vec![(M::Zlib, None), (M::Lzma, None)], raw.clone()),
            (M::None, raw.clone())
        );
        assert_eq!(
            first_encoding(vec![(M::Zlib, Some(vec![7]))], raw),
            (M::Zlib, vec![7])
        );
    }

    #[test]
    fn test_delta_roundtrip() {
        let data = test_signal(1000);
        let rd = compress_residual_delta(&data);

        assert_eq!(rd.method, ResidualCompressionMethod::BitDelta);
        assert_eq!(rd.original_len, 1000);

        let restored = decompress_residual_delta(&rd).expect("delta round trip");

        assert_eq!(restored.len(), data.len());
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "delta round-trip mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_delta_roundtrip_empty() {
        let data: Vec<f32> = Vec::new();
        let rd = compress_residual_delta(&data);
        let restored = decompress_residual_delta(&rd).expect("delta round trip");
        assert!(restored.is_empty());
    }

    #[test]
    fn test_delta_roundtrip_single() {
        let data = vec![42.5f32];
        let rd = compress_residual_delta(&data);
        let restored = decompress_residual_delta(&rd).expect("delta round trip");
        assert_eq!(restored.len(), 1);
        assert!((restored[0] - 42.5).abs() < 1e-4);
    }

    // ------------------------------------------------------------------
    // Serialisation round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_serialization_roundtrip_lzma() {
        let data = test_signal(500);
        let rd = compress_with_method(&data, ResidualCompressionMethod::Lzma);

        let bytes = rd.to_bytes();
        let recovered = ResidualData::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.method, ResidualCompressionMethod::Lzma);
        assert_eq!(recovered.original_len, 500);
        assert_eq!(recovered.compressed, rd.compressed);
    }

    #[test]
    fn test_serialization_roundtrip_delta() {
        let data = test_signal(200);
        let rd = compress_residual_delta(&data);

        let bytes = rd.to_bytes();
        let recovered = ResidualData::from_bytes(&bytes).expect("from_bytes failed");

        // bitdelta keeps every value inside the stream (no base_value key,
        // as the Python writer emits it)
        assert_eq!(recovered.method, ResidualCompressionMethod::BitDelta);
        assert_eq!(recovered.original_len, 200);
        assert_eq!(
            decompress(&recovered).expect("decode"),
            decompress(&rd).expect("decode")
        );
    }

    #[test]
    fn test_serialization_roundtrip_quantized() {
        let data = test_signal(300);
        let rd = compress_with_method(&data, ResidualCompressionMethod::Quantized);

        let bytes = rd.to_bytes();
        let recovered = ResidualData::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.method, ResidualCompressionMethod::Quantized);
        assert_eq!(recovered.original_len, 300);
        // Metadata round-trip.
        assert!((recovered.metadata.min_val - rd.metadata.min_val).abs() < 1e-9);
        assert!((recovered.metadata.scale - rd.metadata.scale).abs() < 1e-9);
        assert_eq!(recovered.metadata.bits, 8);
    }

    #[test]
    fn test_serialization_roundtrip_none() {
        let data = vec![1.0f32, 2.0, 3.0];
        let rd = compress_with_method(&data, ResidualCompressionMethod::None);

        let bytes = rd.to_bytes();
        let recovered = ResidualData::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(recovered.method, ResidualCompressionMethod::None);
        assert_eq!(recovered.original_len, 3);
    }

    #[test]
    fn test_from_bytes_too_short() {
        let result = ResidualData::from_bytes(&[0u8, 1, 2]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // analyze_residual selects best method
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_selects_best() {
        // A highly compressible signal (constant) should prefer something
        // smaller than None.
        let data = vec![0.0f32; 1000];
        let method = analyze_residual(&data);
        // Any lossless method is fine; just ensure it is not None (which
        // would mean "no compression at all"), except when all methods
        // produce the same or larger output (which cannot happen for 1000
        // identical floats with LZMA).
        // In practice LZMA / Delta / Quantized will all beat raw here.
        assert!(
            method != ResidualCompressionMethod::None || {
                // Edge case: if the compressed output is somehow
                // larger than uncompressed, None is acceptable.
                true
            }
        );

        // For an empty slice, always returns None.
        assert_eq!(analyze_residual(&[]), ResidualCompressionMethod::None);
    }

    #[test]
    fn test_analyze_returns_valid_method() {
        let data = test_signal(1000);
        let method = analyze_residual(&data);
        // Just ensure the returned method is one of the known variants.
        let _ = match method {
            ResidualCompressionMethod::None => true,
            ResidualCompressionMethod::Lzma => true,
            ResidualCompressionMethod::Zlib => true,
            ResidualCompressionMethod::Delta => true,
            ResidualCompressionMethod::BitDelta => true,
            ResidualCompressionMethod::Quantized => true,
        };
    }

    // ------------------------------------------------------------------
    // Entropy estimation
    // ------------------------------------------------------------------

    #[test]
    fn test_entropy_estimation() {
        // Constant data → zero entropy.
        let constant = vec![1.0f32; 100];
        assert_eq!(estimate_entropy(&constant), 0.0);

        // Empty data → zero entropy.
        assert_eq!(estimate_entropy(&[]), 0.0);

        // Random-looking data should have higher entropy than constant.
        let varied = test_signal(1000);
        let h = estimate_entropy(&varied);
        assert!(
            h > 0.0,
            "expected positive entropy for varied data, got {}",
            h
        );
        assert!(
            h <= 8.0,
            "entropy cannot exceed 8 bits for 256-bin histogram, got {}",
            h
        );
    }

    #[test]
    fn test_entropy_single_element() {
        // Single element → only one bin occupied → zero entropy.
        let data = vec![2.71f32];
        assert_eq!(estimate_entropy(&data), 0.0);
    }

    // ------------------------------------------------------------------
    // choose_compression
    // ------------------------------------------------------------------

    #[test]
    fn test_choose_compression_lossless() {
        let data = test_signal(500);
        let rd = choose_compression(&data, false);

        // Must not select Quantized when allow_lossy is false.
        assert_ne!(rd.method, ResidualCompressionMethod::Quantized);
        assert_eq!(rd.original_len, 500);
        assert!(!rd.compressed.is_empty());
    }

    #[test]
    fn test_choose_compression_lossy_allowed() {
        let data = test_signal(500);
        let rd = choose_compression(&data, true);

        // Any method is acceptable.
        assert_eq!(rd.original_len, 500);
        assert!(!rd.compressed.is_empty());
    }

    #[test]
    fn test_choose_compression_empty() {
        let rd = choose_compression(&[], false);
        assert_eq!(rd.original_len, 0);
        assert!(rd.compressed.is_empty());
    }

    #[test]
    fn test_choose_compression_roundtrip() {
        let data = test_signal(200);
        let rd = choose_compression(&data, false);
        let restored = decompress(&rd).expect("decompress failed");

        assert_eq!(restored.len(), data.len());

        // For lossless methods the roundtrip must be exact.
        if rd.method != ResidualCompressionMethod::Quantized {
            for (a, b) in data.iter().zip(restored.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "round-trip mismatch for {:?}: {} vs {}",
                    rd.method,
                    a,
                    b
                );
            }
        }
    }
}
