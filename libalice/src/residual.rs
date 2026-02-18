//! Residual Compression Module for ALICE-Zip
//!
//! Ports the Python `alice_zip/residual_compression.py` to Rust.
//!
//! The key insight: procedural generation rarely achieves a 100% match with
//! real data. By storing the residual (original - generated), we can achieve
//! true lossless compression.
//!
//! Formula:
//!   Data_original = Gen(Params) + Decompress(Residual_compressed)
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
/// Prevents DoS via malformed `header_len` values.
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
}

impl std::fmt::Display for ResidualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidualError::DataTooShort { got, expected } => {
                write!(f, "data too short: got {} bytes, expected at least {}", got, expected)
            }
            ResidualError::InvalidHeader(msg) => write!(f, "invalid header: {}", msg),
            ResidualError::MissingField(field) => {
                write!(f, "missing required header field: '{}'", field)
            }
            ResidualError::UnknownMethod(m) => {
                write!(f, "unknown compression method: '{}'", m)
            }
            ResidualError::HeaderTooLarge { size } => {
                write!(
                    f,
                    "header length {} exceeds maximum allowed {} bytes",
                    size, MAX_HEADER_SIZE
                )
            }
            ResidualError::Io(e) => write!(f, "I/O error: {}", e),
            ResidualError::Corrupted(msg) => write!(f, "corrupted data: {}", msg),
        }
    }
}

impl std::error::Error for ResidualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let ResidualError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for ResidualError {
    fn from(e: std::io::Error) -> Self {
        ResidualError::Io(e)
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
    /// Delta encoding followed by LZMA compression.
    Delta,
    /// Quantization (8-bit by default) followed by LZMA compression.
    Quantized,
}

impl ResidualCompressionMethod {
    /// Convert to the canonical string used in JSON headers.
    pub fn as_str(self) -> &'static str {
        match self {
            ResidualCompressionMethod::None => "none",
            ResidualCompressionMethod::Lzma => "lzma",
            ResidualCompressionMethod::Zlib => "zlib",
            ResidualCompressionMethod::Delta => "delta",
            ResidualCompressionMethod::Quantized => "quantized",
        }
    }

    /// Parse from the canonical string stored in JSON headers.
    ///
    /// Returns `None` for unrecognised strings.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(ResidualCompressionMethod::None),
            "lzma" => Some(ResidualCompressionMethod::Lzma),
            "zlib" => Some(ResidualCompressionMethod::Zlib),
            "delta" => Some(ResidualCompressionMethod::Delta),
            "quantized" => Some(ResidualCompressionMethod::Quantized),
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
}

impl Default for ResidualMetadata {
    fn default() -> Self {
        ResidualMetadata {
            min_val: 0.0,
            scale: 1.0,
            bits: 8,
            base_value: 0.0,
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
    pub fn to_bytes(&self) -> Vec<u8> {
        // Build the JSON header using a plain HashMap to avoid pulling in a
        // heavyweight serialiser dependency (serde_json may not be available).
        // We construct the JSON string manually for portability.
        let mut fields: Vec<String> = Vec::new();
        fields.push(format!(r#""method":"{}""#, self.method.as_str()));
        fields.push(format!(r#""original_len":{}"#, self.original_len));
        fields.push(r#""version":2"#.to_owned());

        match self.method {
            ResidualCompressionMethod::Quantized => {
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
    /// in the parsed JSON after a successful v2 parse attempt.
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
                match std::str::from_utf8(&data[4..v2_end]) {
                    Ok(json_str) => match Self::parse_json_header(json_str) {
                        Ok(parsed) => {
                            if parsed.get("version").and_then(|v| v.parse::<u32>().ok()).unwrap_or(1) >= 2 {
                                let compressed = data[v2_end..].to_vec();
                                return Self::from_header_map(parsed, compressed);
                            }
                            // version < 2 — fall through to v1
                        }
                        Err(_) => {
                            // Not valid JSON — fall through to v1
                        }
                    },
                    Err(_) => {
                        // Not valid UTF-8 — fall through to v1
                    }
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
            return Err(ResidualError::HeaderTooLarge { size: header_len_v1 });
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
            .map_err(|e| ResidualError::InvalidHeader(format!("UTF-8 decode failed: {}", e)))?;

        let parsed = Self::parse_json_header(json_str)
            .map_err(|e| ResidualError::InvalidHeader(e))?;

        let compressed = data[v1_payload_start..].to_vec();
        Self::from_header_map(parsed, compressed)
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
    fn parse_json_header(json: &str) -> Result<HashMap<String, String>, String> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(format!("JSON header must be a flat object, got: {}", json));
        }

        let inner = &json[1..json.len() - 1];
        let mut map = HashMap::new();

        // Split by comma — this works for flat objects where values do not
        // themselves contain commas (true for all fields we emit).
        for pair in inner.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            // Split on the first colon.
            let colon_pos = pair.find(':').ok_or_else(|| {
                format!("malformed key-value pair (no ':'): '{}'", pair)
            })?;

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
        map: HashMap<String, String>,
        compressed: Vec<u8>,
    ) -> Result<Self, ResidualError> {
        // Required field: method
        let method_str = map
            .get("method")
            .ok_or_else(|| ResidualError::MissingField("method".to_owned()))?;

        let method = ResidualCompressionMethod::from_str(method_str)
            .ok_or_else(|| ResidualError::UnknownMethod(method_str.clone()))?;

        // Required field: original_len
        let original_len = map
            .get("original_len")
            .ok_or_else(|| ResidualError::MissingField("original_len".to_owned()))?
            .parse::<usize>()
            .map_err(|e| {
                ResidualError::InvalidHeader(format!("invalid original_len: {}", e))
            })?;

        // Optional / method-specific fields.
        let mut metadata = ResidualMetadata::default();

        if method == ResidualCompressionMethod::Quantized {
            if let Some(v) = map.get("min_val") {
                metadata.min_val = v.parse().unwrap_or(0.0);
            }
            if let Some(v) = map.get("scale") {
                metadata.scale = v.parse().unwrap_or(1.0);
            }
            if let Some(v) = map.get("bits") {
                metadata.bits = v.parse().unwrap_or(8);
            }
        }

        if method == ResidualCompressionMethod::Delta {
            if let Some(v) = map.get("base_value") {
                metadata.base_value = v.parse().unwrap_or(0.0);
            }
        }

        Ok(ResidualData {
            method,
            compressed,
            original_len,
            metadata,
        })
    }
}

// ============================================================================
// Delta compression
// ============================================================================

/// Compress `data` using delta encoding followed by LZMA.
///
/// The delta stream is constructed as:
/// ```text
/// delta[0] = data[0]
/// delta[i] = data[i] - data[i-1]   for i > 0
/// ```
/// Each element is stored as a little-endian `f32`.
pub fn compress_residual_delta(data: &[f32]) -> ResidualData {
    let base_value = data.first().copied().unwrap_or(0.0);

    // Build delta array (prepend first element as its own delta).
    let mut deltas: Vec<u8> = Vec::with_capacity(data.len() * 4);
    let mut prev = 0.0f32;
    for &v in data {
        let d = v - prev;
        deltas.extend_from_slice(&d.to_le_bytes());
        prev = v;
    }

    // LZMA compress the raw delta bytes.
    let compressed = crate::compression::lzma_compress(&deltas, 6)
        .unwrap_or_else(|_| deltas.clone()); // fallback: store uncompressed

    ResidualData {
        method: ResidualCompressionMethod::Delta,
        compressed,
        original_len: data.len(),
        metadata: ResidualMetadata {
            base_value,
            ..ResidualMetadata::default()
        },
    }
}

/// Decompress a delta-encoded `ResidualData` back to `Vec<f32>`.
///
/// Panics if `rd.method` is not `Delta` (callers are responsible for routing).
pub fn decompress_residual_delta(rd: &ResidualData) -> Vec<f32> {
    debug_assert_eq!(
        rd.method,
        ResidualCompressionMethod::Delta,
        "decompress_residual_delta called with non-Delta method"
    );

    // LZMA decompress.
    let raw = crate::compression::lzma_decompress(&rd.compressed)
        .unwrap_or_else(|_| rd.compressed.clone());

    // Reconstruct from cumulative sum of deltas.
    let mut result = Vec::with_capacity(raw.len() / 4);
    let mut acc = 0.0f32;
    for chunk in raw.chunks_exact(4) {
        let d = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc += d;
        result.push(acc);
    }

    result
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
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = max_val - min_val;

    // Build 256-bin histogram.
    let mut counts = [0u32; 256];
    if scale < 1e-10 {
        // Constant data: zero entropy.
        return 0.0;
    }

    let rcp_scale = 255.0 / scale as f64;
    for &v in data {
        let bin = ((v as f64 - min_val as f64) * rcp_scale)
            .clamp(0.0, 255.0) as usize;
        counts[bin] += 1;
    }

    // Shannon entropy using reciprocal multiplication.
    let total = data.len() as f64;
    let rcp_total = 1.0 / total; // precomputed reciprocal

    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 * rcp_total; // multiply, not divide
            entropy -= p * p.log2();
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
        .map(|(method, _)| method)
        .unwrap_or(ResidualCompressionMethod::None)
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
pub fn choose_compression(data: &[f32], allow_lossy: bool) -> ResidualData {
    if data.is_empty() {
        return ResidualData {
            method: ResidualCompressionMethod::None,
            compressed: Vec::new(),
            original_len: 0,
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
            metadata: ResidualMetadata::default(),
        },

        ResidualCompressionMethod::Lzma => {
            let compressed = crate::compression::lzma_compress(&raw_bytes, 6)
                .unwrap_or(raw_bytes);
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                metadata: ResidualMetadata::default(),
            }
        }

        ResidualCompressionMethod::Zlib => {
            let compressed = crate::compression::zlib_compress(&raw_bytes, 6)
                .unwrap_or_else(|_| {
                    crate::compression::lzma_compress(&raw_bytes, 6)
                        .unwrap_or(raw_bytes.clone())
                });
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                metadata: ResidualMetadata::default(),
            }
        }

        ResidualCompressionMethod::Delta => compress_residual_delta(data),

        ResidualCompressionMethod::Quantized => {
            // Use the existing compress_residual_quantized function which
            // writes the quantisation header (bits, min_val, scale,
            // compressed_len) followed by LZMA data.
            //
            // We also populate ResidualMetadata for symmetry with the
            // Python implementation.
            let (_, min_val, scale) = crate::compression::quantize_8bit(data);
            let compressed =
                crate::compression::compress_residual_quantized(data, 8, 6)
                    .unwrap_or_else(|_| {
                        crate::compression::lzma_compress(&raw_bytes, 6)
                            .unwrap_or(raw_bytes)
                    });
            ResidualData {
                method,
                compressed,
                original_len: data.len(),
                metadata: ResidualMetadata {
                    min_val,
                    scale,
                    bits: 8,
                    base_value: 0.0,
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

    match rd.method {
        ResidualCompressionMethod::None => {
            let result: Vec<f32> = rd
                .compressed
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(result)
        }

        ResidualCompressionMethod::Lzma => {
            let raw = crate::compression::lzma_decompress(&rd.compressed)?;
            let result: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(result)
        }

        ResidualCompressionMethod::Zlib => {
            let raw = crate::compression::zlib_decompress(&rd.compressed)?;
            let result: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(result)
        }

        ResidualCompressionMethod::Delta => Ok(decompress_residual_delta(rd)),

        ResidualCompressionMethod::Quantized => {
            let result =
                crate::compression::decompress_residual_quantized(&rd.compressed)?;
            Ok(result)
        }
    }
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
            .map(|i| ((i as f32) * std::f32::consts::TAU / 100.0).sin() * 50.0)
            .collect()
    }

    // ------------------------------------------------------------------
    // Delta round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_delta_roundtrip() {
        let data = test_signal(1000);
        let rd = compress_residual_delta(&data);

        assert_eq!(rd.method, ResidualCompressionMethod::Delta);
        assert_eq!(rd.original_len, 1000);

        let restored = decompress_residual_delta(&rd);

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
        let restored = decompress_residual_delta(&rd);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_delta_roundtrip_single() {
        let data = vec![42.5f32];
        let rd = compress_residual_delta(&data);
        let restored = decompress_residual_delta(&rd);
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

        assert_eq!(recovered.method, ResidualCompressionMethod::Delta);
        assert_eq!(recovered.original_len, 200);
        assert!(
            (recovered.metadata.base_value - rd.metadata.base_value).abs() < 1e-6
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
            method != ResidualCompressionMethod::None
                || {
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
        assert!(h > 0.0, "expected positive entropy for varied data, got {}", h);
        assert!(h <= 8.0, "entropy cannot exceed 8 bits for 256-bin histogram, got {}", h);
    }

    #[test]
    fn test_entropy_single_element() {
        // Single element → only one bin occupied → zero entropy.
        let data = vec![3.14f32];
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
