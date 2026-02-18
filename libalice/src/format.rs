//! ALICE-Zip File Format — Header definitions and binary serialization
//!
//! Ports the Python `AliceFileHeader` from `alice_zip/core.py` to Rust.
//!
//! # Binary Layout
//!
//! All integers are **little-endian**.
//!
//! ## V1 format (65 bytes) — version 1.0
//! ```text
//! Offset  Size  Field
//!      0     9  magic          = b"ALICE_ZIP"
//!      9     1  version_major
//!     10     1  version_minor
//!     11     1  file_type      (AliceFileType discriminant)
//!     12     1  engine_index   (reserved, always 0)
//!     13     8  original_size  (u64 LE)
//!     21     8  compressed_size (u64 LE)
//!     29    32  original_hash  ([u8; 32], SHA-256 or zeroed)
//!     61     4  flags/checksum (u32 LE)
//! ```
//!
//! ## V2 format (66 bytes) — version >= 1.1
//! ```text
//! Offset  Size  Field
//!      0     9  magic          = b"ALICE_ZIP"
//!      9     1  version_major
//!     10     1  version_minor
//!     11     1  file_type      (AliceFileType discriminant)
//!     12     1  engine_index   (reserved, always 0)
//!     13     1  payload_type   (AlicePayloadType discriminant)  ← new in v2
//!     14     8  original_size  (u64 LE)
//!     22     8  compressed_size (u64 LE)
//!     30    32  original_hash  ([u8; 32], SHA-256 or zeroed)
//!     62     4  flags/checksum (u32 LE)
//! ```
//!
//! # Version Detection
//!
//! A header is v2 when `version_major > 1 || (version_major == 1 && version_minor >= 1)`.
//! Readers must accept both v1 and v2; unknown payload_type values fall back to
//! `AlicePayloadType::Procedural`.
//!
//! # Author
//! Moroya Sakamoto

use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes that appear at offset 0 of every .alice file.
pub const ALICE_MAGIC: &[u8; 9] = b"ALICE_ZIP";

/// Current default version written by `to_bytes()`.
pub const DEFAULT_VERSION_MAJOR: u8 = 1;
pub const DEFAULT_VERSION_MINOR: u8 = 1; // v2 format (>= 1.1)

/// Size of the v1 header (no payload_type field).
pub const HEADER_V1_SIZE: usize = 65;

/// Size of the v2 header (with payload_type field).
pub const HEADER_V2_SIZE: usize = 66;

// ============================================================================
// Enums
// ============================================================================

/// File types supported by ALICE_Zip.
///
/// Corresponds to `AliceFileType` in `alice_zip/core.py`.
///
/// Note: Python uses 0x01–0x05; we keep the same discriminants for
/// binary compatibility (see `From<u8>` / `to_u8` impls below).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AliceFileType {
    NumpyArray = 0x01,
    Image      = 0x02,
    Audio      = 0x03,
    Text       = 0x04,
    Binary     = 0x05,
}

impl AliceFileType {
    /// Encode to the single byte stored in the header.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode from the single byte stored in the header.
    ///
    /// Returns `Err(FormatError::InvalidFileType(v))` for unrecognised values.
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0x01 => Ok(AliceFileType::NumpyArray),
            0x02 => Ok(AliceFileType::Image),
            0x03 => Ok(AliceFileType::Audio),
            0x04 => Ok(AliceFileType::Text),
            0x05 => Ok(AliceFileType::Binary),
            _    => Err(FormatError::InvalidFileType(v)),
        }
    }
}

impl fmt::Display for AliceFileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AliceFileType::NumpyArray => write!(f, "NumpyArray"),
            AliceFileType::Image      => write!(f, "Image"),
            AliceFileType::Audio      => write!(f, "Audio"),
            AliceFileType::Text       => write!(f, "Text"),
            AliceFileType::Binary     => write!(f, "Binary"),
        }
    }
}

// ----------------------------------------------------------------------------

/// Payload type identifier for unified format handling.
///
/// Determines which decompressor should be used for the payload that follows
/// the header. Corresponds to `AlicePayloadType` in `alice_zip/core.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AlicePayloadType {
    /// Procedural parameters (original ALICE_ZIP format).
    Procedural  = 0x00,
    /// ALICE_IMG — JSON-based image payload.
    MediaImage  = 0x10,
    /// ALICE_AUD — JSON-based audio payload.
    MediaAudio  = 0x11,
    /// ALICE_VID — JSON-based video payload.
    MediaVideo  = 0x12,
    /// ALICE_TEX — JSON-based texture payload.
    Texture     = 0x20,
    /// LZMA-compressed fallback payload.
    LzmaFallback = 0x30,
}

impl AlicePayloadType {
    /// Encode to the single byte stored in the v2 header.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode from the single byte stored in the v2 header.
    ///
    /// Per the Python spec, unknown values fall back to `Procedural` instead
    /// of returning an error, matching the graceful-degradation policy.
    pub fn from_u8_lenient(v: u8) -> Self {
        match v {
            0x00 => AlicePayloadType::Procedural,
            0x10 => AlicePayloadType::MediaImage,
            0x11 => AlicePayloadType::MediaAudio,
            0x12 => AlicePayloadType::MediaVideo,
            0x20 => AlicePayloadType::Texture,
            0x30 => AlicePayloadType::LzmaFallback,
            _    => AlicePayloadType::Procedural, // graceful fallback
        }
    }

    /// Strict decode — returns `Err` for unrecognised values.
    pub fn from_u8_strict(v: u8) -> Result<Self, FormatError> {
        match v {
            0x00 => Ok(AlicePayloadType::Procedural),
            0x10 => Ok(AlicePayloadType::MediaImage),
            0x11 => Ok(AlicePayloadType::MediaAudio),
            0x12 => Ok(AlicePayloadType::MediaVideo),
            0x20 => Ok(AlicePayloadType::Texture),
            0x30 => Ok(AlicePayloadType::LzmaFallback),
            _    => Err(FormatError::InvalidPayloadType(v)),
        }
    }
}

impl fmt::Display for AlicePayloadType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlicePayloadType::Procedural  => write!(f, "Procedural"),
            AlicePayloadType::MediaImage  => write!(f, "MediaImage"),
            AlicePayloadType::MediaAudio  => write!(f, "MediaAudio"),
            AlicePayloadType::MediaVideo  => write!(f, "MediaVideo"),
            AlicePayloadType::Texture     => write!(f, "Texture"),
            AlicePayloadType::LzmaFallback => write!(f, "LzmaFallback"),
        }
    }
}

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur while parsing or constructing an [`AliceFileHeader`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// The magic bytes do not match `b"ALICE_ZIP"`.
    InvalidMagic,
    /// The data slice is too short to contain a complete header.
    TooShort { expected: usize, got: usize },
    /// An unrecognised `AliceFileType` discriminant was found.
    InvalidFileType(u8),
    /// An unrecognised `AlicePayloadType` discriminant was found (strict mode).
    InvalidPayloadType(u8),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::InvalidMagic =>
                write!(f, "invalid magic bytes (expected b\"ALICE_ZIP\")"),
            FormatError::TooShort { expected, got } =>
                write!(f, "data too short: expected at least {} bytes, got {}", expected, got),
            FormatError::InvalidFileType(v) =>
                write!(f, "invalid AliceFileType discriminant: 0x{:02X}", v),
            FormatError::InvalidPayloadType(v) =>
                write!(f, "invalid AlicePayloadType discriminant: 0x{:02X}", v),
        }
    }
}

impl std::error::Error for FormatError {}

// ============================================================================
// Header struct
// ============================================================================

/// Header for `.alice` files — binary format v1 (65 bytes) and v2 (66 bytes).
///
/// # Fields that map directly to the Python dataclass
///
/// | Rust field        | Python field      | Size (bytes) |
/// |-------------------|-------------------|--------------|
/// | `magic`           | `magic`           | 9            |
/// | `version_major`   | `version_major`   | 1            |
/// | `version_minor`   | `version_minor`   | 1            |
/// | `file_type`       | `file_type`       | 1            |
/// | `engine_index`    | `engine` (index)  | 1            |
/// | `payload_type`    | `payload_type`    | 1 (v2 only)  |
/// | `original_size`   | `original_size`   | 8            |
/// | `compressed_size` | `compressed_size` | 8            |
/// | `original_hash`   | `original_hash`   | 32           |
/// | `checksum`        | `flags`           | 4            |
///
/// # Additional metadata fields (in-memory only, not serialised)
///
/// These fields supplement the binary header for in-memory use and are
/// populated / defaulted to zero during `from_bytes`:
/// - `param_count`: number of generator parameters in the payload
/// - `original_dtype`: NumPy dtype code for the original array
/// - `shape_dims`: number of meaningful dimensions in `shape`
/// - `shape`: up to 4-dimensional array shape
///
/// # Version tracking
///
/// The private `header_version` field records whether the header was parsed
/// from a v1 or v2 binary; `to_bytes()` always writes v2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliceFileHeader {
    // --- Serialised fields (both v1 and v2) ---
    /// `b"ALICE_ZIP"` — 9 bytes.
    pub magic: [u8; 9],
    pub version_major: u8,
    pub version_minor: u8,
    pub file_type: AliceFileType,
    /// Engine index reserved field (always 0 in current implementations).
    pub engine_index: u8,
    /// V2 only; defaults to `Procedural` when reading a v1 file.
    pub payload_type: AlicePayloadType,
    /// Uncompressed data size in bytes.
    pub original_size: u64,
    /// Compressed data size in bytes.
    pub compressed_size: u64,
    /// SHA-256 hash of the original data, or 32 zero bytes if unused.
    pub original_hash: [u8; 32],
    /// Checksum / flags (maps to Python `flags` field, u32 LE).
    pub checksum: u32,

    // --- In-memory only supplemental fields (not serialised) ---
    /// Number of generator parameters packed after the header.
    pub param_count: u32,
    /// NumPy dtype code (e.g. `0x17` for float32). 0 = unset.
    pub original_dtype: u8,
    /// Number of valid dimensions in `shape` (0–4).
    pub shape_dims: u8,
    /// Array shape — up to 4 dimensions. Unused dims are 0.
    pub shape: [u32; 4],

    // --- Private bookkeeping ---
    /// Which version this struct was deserialised from (1 or 2).
    header_version: u8,
}

impl AliceFileHeader {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new header with sensible defaults.
    ///
    /// Writes `version 1.1` so that `to_bytes()` emits a 66-byte v2 header.
    pub fn new(
        file_type: AliceFileType,
        payload_type: AlicePayloadType,
        original_size: u64,
        compressed_size: u64,
    ) -> Self {
        Self {
            magic: *ALICE_MAGIC,
            version_major: DEFAULT_VERSION_MAJOR,
            version_minor: DEFAULT_VERSION_MINOR,
            file_type,
            engine_index: 0,
            payload_type,
            original_size,
            compressed_size,
            original_hash: [0u8; 32],
            checksum: 0,
            param_count: 0,
            original_dtype: 0,
            shape_dims: 0,
            shape: [0u32; 4],
            header_version: 2,
        }
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Serialise this header to the **v2 binary format** (66 bytes, LE).
    ///
    /// Layout (offsets in bytes):
    /// ```text
    ///  0.. 9  magic
    ///  9     version_major
    /// 10     version_minor
    /// 11     file_type
    /// 12     engine_index
    /// 13     payload_type
    /// 14..22 original_size   (u64 LE)
    /// 22..30 compressed_size (u64 LE)
    /// 30..62 original_hash   (32 bytes)
    /// 62..66 checksum        (u32 LE)
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_V2_SIZE);

        // magic (9)
        out.extend_from_slice(&self.magic);
        // version (2)
        out.push(self.version_major);
        out.push(self.version_minor);
        // file_type (1)
        out.push(self.file_type.to_u8());
        // engine_index (1)
        out.push(self.engine_index);
        // payload_type (1) — v2 field
        out.push(self.payload_type.to_u8());
        // original_size (8 LE)
        out.extend_from_slice(&self.original_size.to_le_bytes());
        // compressed_size (8 LE)
        out.extend_from_slice(&self.compressed_size.to_le_bytes());
        // original_hash (32)
        out.extend_from_slice(&self.original_hash);
        // checksum / flags (4 LE)
        out.extend_from_slice(&self.checksum.to_le_bytes());

        debug_assert_eq!(out.len(), HEADER_V2_SIZE,
            "to_bytes() produced {} bytes, expected {}", out.len(), HEADER_V2_SIZE);
        out
    }

    // -----------------------------------------------------------------------
    // Deserialisation
    // -----------------------------------------------------------------------

    /// Deserialise a header from raw bytes with **automatic v1/v2 detection**.
    ///
    /// * V1 (65 bytes) — `version 1.0`; `payload_type` is inferred as
    ///   `Procedural`.
    /// * V2 (66 bytes) — `version >= 1.1`; contains an explicit
    ///   `payload_type` byte; unknown values are treated as `Procedural`.
    ///
    /// Only the first `HEADER_V{1,2}_SIZE` bytes are consumed; any extra
    /// trailing bytes are ignored (forward-compatibility policy).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError`] if:
    /// - `data` is shorter than `HEADER_V1_SIZE` (65 bytes)
    /// - the magic bytes do not match `b"ALICE_ZIP"`
    /// - `file_type` contains an unrecognised discriminant
    pub fn from_bytes(data: &[u8]) -> Result<Self, FormatError> {
        // Minimum length check
        if data.len() < HEADER_V1_SIZE {
            return Err(FormatError::TooShort {
                expected: HEADER_V1_SIZE,
                got: data.len(),
            });
        }

        // Magic check (bytes 0..9)
        let magic_bytes: [u8; 9] = data[0..9].try_into().unwrap(); // length guaranteed
        if &magic_bytes != ALICE_MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        let version_major = data[9];
        let version_minor = data[10];

        // Determine format version.
        // v2: major > 1 OR (major == 1 AND minor >= 1)
        let is_v2 = (version_major > 1) || (version_major == 1 && version_minor >= 1);

        if is_v2 && data.len() >= HEADER_V2_SIZE {
            Self::parse_v2(data, version_major, version_minor, magic_bytes)
        } else {
            Self::parse_v1(data, version_major, version_minor, magic_bytes)
        }
    }

    /// Parse a v2 (66-byte) header.
    fn parse_v2(
        data: &[u8],
        version_major: u8,
        version_minor: u8,
        magic: [u8; 9],
    ) -> Result<Self, FormatError> {
        // Byte 11: file_type
        let file_type = AliceFileType::from_u8(data[11])?;
        // Byte 12: engine_index
        let engine_index = data[12];
        // Byte 13: payload_type (lenient — unknown → Procedural)
        let payload_type = AlicePayloadType::from_u8_lenient(data[13]);

        // Bytes 14..22: original_size (u64 LE)
        let original_size = u64::from_le_bytes(
            data[14..22].try_into().expect("slice length guaranteed"),
        );
        // Bytes 22..30: compressed_size (u64 LE)
        let compressed_size = u64::from_le_bytes(
            data[22..30].try_into().expect("slice length guaranteed"),
        );
        // Bytes 30..62: original_hash (32 bytes)
        let original_hash: [u8; 32] = data[30..62].try_into().expect("slice length guaranteed");
        // Bytes 62..66: checksum (u32 LE)
        let checksum = u32::from_le_bytes(
            data[62..66].try_into().expect("slice length guaranteed"),
        );

        Ok(Self {
            magic,
            version_major,
            version_minor,
            file_type,
            engine_index,
            payload_type,
            original_size,
            compressed_size,
            original_hash,
            checksum,
            param_count: 0,
            original_dtype: 0,
            shape_dims: 0,
            shape: [0u32; 4],
            header_version: 2,
        })
    }

    /// Parse a v1 (65-byte) header.
    fn parse_v1(
        data: &[u8],
        version_major: u8,
        version_minor: u8,
        magic: [u8; 9],
    ) -> Result<Self, FormatError> {
        // Byte 11: file_type
        let file_type = AliceFileType::from_u8(data[11])?;
        // Byte 12: engine_index
        let engine_index = data[12];
        // No payload_type byte in v1 — default to Procedural.

        // Bytes 13..21: original_size (u64 LE)
        let original_size = u64::from_le_bytes(
            data[13..21].try_into().expect("slice length guaranteed"),
        );
        // Bytes 21..29: compressed_size (u64 LE)
        let compressed_size = u64::from_le_bytes(
            data[21..29].try_into().expect("slice length guaranteed"),
        );
        // Bytes 29..61: original_hash (32 bytes)
        let original_hash: [u8; 32] = data[29..61].try_into().expect("slice length guaranteed");
        // Bytes 61..65: checksum (u32 LE)
        let checksum = u32::from_le_bytes(
            data[61..65].try_into().expect("slice length guaranteed"),
        );

        Ok(Self {
            magic,
            version_major,
            version_minor,
            file_type,
            engine_index,
            payload_type: AlicePayloadType::Procedural, // v1 default
            original_size,
            compressed_size,
            original_hash,
            checksum,
            param_count: 0,
            original_dtype: 0,
            shape_dims: 0,
            shape: [0u32; 4],
            header_version: 1,
        })
    }

    // -----------------------------------------------------------------------
    // Introspection helpers
    // -----------------------------------------------------------------------

    /// Return the **serialised** header size in bytes for this instance.
    ///
    /// - Returns `65` if the header was deserialised from a v1 file.
    /// - Returns `66` for all other cases (including freshly constructed headers).
    pub fn header_size(&self) -> usize {
        if self.header_version == 1 {
            HEADER_V1_SIZE
        } else {
            HEADER_V2_SIZE
        }
    }

    /// Return the header format version (1 or 2) recorded at parse time.
    #[inline]
    pub fn header_version(&self) -> u8 {
        self.header_version
    }

    /// Return `true` if the magic bytes are exactly `b"ALICE_ZIP"`.
    #[inline]
    pub fn has_valid_magic(&self) -> bool {
        &self.magic == ALICE_MAGIC
    }
}

impl fmt::Display for AliceFileHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AliceFileHeader(v{}.{}, type={}, payload={}, orig={}, comp={}, hdr_ver={})",
            self.version_major,
            self.version_minor,
            self.file_type,
            self.payload_type,
            self.original_size,
            self.compressed_size,
            self.header_version,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a minimal v2 byte buffer by hand, matching the Python struct
    /// format `'<9sBBBBB Q Q 32s I'` (66 bytes total).
    fn build_v2_bytes(
        version_major: u8,
        version_minor: u8,
        file_type: u8,
        engine_idx: u8,
        payload_type: u8,
        original_size: u64,
        compressed_size: u64,
        original_hash: [u8; 32],
        checksum: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_V2_SIZE);
        buf.extend_from_slice(ALICE_MAGIC);       //  0.. 9
        buf.push(version_major);                  //  9
        buf.push(version_minor);                  // 10
        buf.push(file_type);                      // 11
        buf.push(engine_idx);                     // 12
        buf.push(payload_type);                   // 13
        buf.extend_from_slice(&original_size.to_le_bytes());    // 14..22
        buf.extend_from_slice(&compressed_size.to_le_bytes());  // 22..30
        buf.extend_from_slice(&original_hash);    // 30..62
        buf.extend_from_slice(&checksum.to_le_bytes());         // 62..66
        assert_eq!(buf.len(), HEADER_V2_SIZE);
        buf
    }

    /// Build a minimal v1 byte buffer (65 bytes, no payload_type byte).
    fn build_v1_bytes(
        version_major: u8,
        version_minor: u8,
        file_type: u8,
        engine_idx: u8,
        original_size: u64,
        compressed_size: u64,
        original_hash: [u8; 32],
        checksum: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_V1_SIZE);
        buf.extend_from_slice(ALICE_MAGIC);       //  0.. 9
        buf.push(version_major);                  //  9
        buf.push(version_minor);                  // 10
        buf.push(file_type);                      // 11
        buf.push(engine_idx);                     // 12
        // no payload_type byte in v1
        buf.extend_from_slice(&original_size.to_le_bytes());    // 13..21
        buf.extend_from_slice(&compressed_size.to_le_bytes());  // 21..29
        buf.extend_from_slice(&original_hash);    // 29..61
        buf.extend_from_slice(&checksum.to_le_bytes());         // 61..65
        assert_eq!(buf.len(), HEADER_V1_SIZE);
        buf
    }

    // -----------------------------------------------------------------------
    // Core roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_roundtrip() {
        let original = AliceFileHeader::new(
            AliceFileType::NumpyArray,
            AlicePayloadType::Procedural,
            123_456,
            42_000,
        );

        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), HEADER_V2_SIZE, "to_bytes() must produce exactly 66 bytes");

        let parsed = AliceFileHeader::from_bytes(&bytes)
            .expect("roundtrip parse must succeed");

        assert_eq!(parsed.magic,           original.magic);
        assert_eq!(parsed.version_major,   original.version_major);
        assert_eq!(parsed.version_minor,   original.version_minor);
        assert_eq!(parsed.file_type,       original.file_type);
        assert_eq!(parsed.engine_index,    original.engine_index);
        assert_eq!(parsed.payload_type,    original.payload_type);
        assert_eq!(parsed.original_size,   original.original_size);
        assert_eq!(parsed.compressed_size, original.compressed_size);
        assert_eq!(parsed.original_hash,   original.original_hash);
        assert_eq!(parsed.checksum,        original.checksum);
        assert_eq!(parsed.header_version(), 2);
        assert_eq!(parsed.header_size(),    HEADER_V2_SIZE);
    }

    #[test]
    fn test_roundtrip_preserves_all_file_types() {
        let types = [
            AliceFileType::NumpyArray,
            AliceFileType::Image,
            AliceFileType::Audio,
            AliceFileType::Text,
            AliceFileType::Binary,
        ];
        for ft in types {
            let hdr = AliceFileHeader::new(ft, AlicePayloadType::Procedural, 0, 0);
            let bytes = hdr.to_bytes();
            let parsed = AliceFileHeader::from_bytes(&bytes).unwrap();
            assert_eq!(parsed.file_type, ft,
                "file_type {:?} must survive roundtrip", ft);
        }
    }

    #[test]
    fn test_roundtrip_preserves_all_payload_types() {
        let types = [
            AlicePayloadType::Procedural,
            AlicePayloadType::MediaImage,
            AlicePayloadType::MediaAudio,
            AlicePayloadType::MediaVideo,
            AlicePayloadType::Texture,
            AlicePayloadType::LzmaFallback,
        ];
        for pt in types {
            let hdr = AliceFileHeader::new(AliceFileType::Binary, pt, 0, 0);
            let bytes = hdr.to_bytes();
            let parsed = AliceFileHeader::from_bytes(&bytes).unwrap();
            assert_eq!(parsed.payload_type, pt,
                "payload_type {:?} must survive roundtrip", pt);
        }
    }

    // -----------------------------------------------------------------------
    // V1 compatibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_v1_compatibility() {
        // Construct a 65-byte v1 buffer by hand (version 1.0, no payload_type byte).
        let mut hash = [0u8; 32];
        hash[0] = 0xAB;
        hash[31] = 0xCD;

        let buf = build_v1_bytes(
            1, 0,                        // version 1.0  → triggers v1 path
            AliceFileType::Image.to_u8(),
            0,                           // engine_index
            9_999_999,                   // original_size
            1_234_567,                   // compressed_size
            hash,
            0xDEAD_BEEF,                 // checksum
        );
        assert_eq!(buf.len(), HEADER_V1_SIZE);

        let hdr = AliceFileHeader::from_bytes(&buf).expect("v1 parse must succeed");

        assert_eq!(hdr.header_version(), 1);
        assert_eq!(hdr.header_size(), HEADER_V1_SIZE);
        assert_eq!(hdr.version_major, 1);
        assert_eq!(hdr.version_minor, 0);
        assert_eq!(hdr.file_type, AliceFileType::Image);
        assert_eq!(hdr.payload_type, AlicePayloadType::Procedural,
            "v1 headers must default payload_type to Procedural");
        assert_eq!(hdr.original_size, 9_999_999);
        assert_eq!(hdr.compressed_size, 1_234_567);
        assert_eq!(hdr.original_hash, hash);
        assert_eq!(hdr.checksum, 0xDEAD_BEEF);
    }

    #[test]
    fn test_v1_with_trailing_bytes_ignored() {
        // Extra bytes after a v1 buffer must not cause an error (forward compat).
        let mut buf = build_v1_bytes(
            1, 0,
            AliceFileType::Binary.to_u8(),
            0,
            0, 0, [0u8; 32], 0,
        );
        buf.extend_from_slice(&[0xFFu8; 10]); // spurious trailing bytes
        let hdr = AliceFileHeader::from_bytes(&buf);
        assert!(hdr.is_ok(), "trailing bytes after a v1 header must be tolerated");
    }

    // -----------------------------------------------------------------------
    // V2 format
    // -----------------------------------------------------------------------

    #[test]
    fn test_v2_format() {
        let mut hash = [0u8; 32];
        for (i, b) in hash.iter_mut().enumerate() {
            *b = i as u8;
        }

        let buf = build_v2_bytes(
            1, 1,                                      // version 1.1 → v2
            AliceFileType::Audio.to_u8(),
            0,
            AlicePayloadType::MediaAudio.to_u8(),
            8_000_000,
            500_000,
            hash,
            0x1234_5678,
        );
        assert_eq!(buf.len(), HEADER_V2_SIZE);

        let hdr = AliceFileHeader::from_bytes(&buf).expect("v2 parse must succeed");

        assert_eq!(hdr.header_version(), 2);
        assert_eq!(hdr.header_size(), HEADER_V2_SIZE);
        assert_eq!(hdr.version_major, 1);
        assert_eq!(hdr.version_minor, 1);
        assert_eq!(hdr.file_type, AliceFileType::Audio);
        assert_eq!(hdr.payload_type, AlicePayloadType::MediaAudio);
        assert_eq!(hdr.original_size, 8_000_000);
        assert_eq!(hdr.compressed_size, 500_000);
        assert_eq!(hdr.original_hash, hash);
        assert_eq!(hdr.checksum, 0x1234_5678);
    }

    #[test]
    fn test_v2_with_trailing_bytes_ignored() {
        // Extra bytes after a v2 buffer must not cause an error.
        let mut buf = build_v2_bytes(
            1, 2,
            AliceFileType::Text.to_u8(),
            0,
            AlicePayloadType::LzmaFallback.to_u8(),
            0, 0, [0u8; 32], 0,
        );
        buf.extend_from_slice(&[0xAAu8; 32]);
        let hdr = AliceFileHeader::from_bytes(&buf);
        assert!(hdr.is_ok(), "trailing bytes after a v2 header must be tolerated");
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_magic() {
        let mut buf = build_v2_bytes(
            1, 1,
            AliceFileType::NumpyArray.to_u8(),
            0,
            AlicePayloadType::Procedural.to_u8(),
            0, 0, [0u8; 32], 0,
        );
        // Corrupt the magic bytes
        buf[0] = b'X';
        buf[1] = b'Y';
        buf[2] = b'Z';

        let result = AliceFileHeader::from_bytes(&buf);
        assert_eq!(result, Err(FormatError::InvalidMagic));
    }

    #[test]
    fn test_too_short() {
        // Empty slice
        let result = AliceFileHeader::from_bytes(&[]);
        assert_eq!(result, Err(FormatError::TooShort { expected: HEADER_V1_SIZE, got: 0 }));

        // 10 bytes — magic is fine but body is missing
        let mut buf = vec![0u8; 10];
        buf[..9].copy_from_slice(ALICE_MAGIC);
        let result = AliceFileHeader::from_bytes(&buf);
        assert_eq!(result, Err(FormatError::TooShort { expected: HEADER_V1_SIZE, got: 10 }));

        // 64 bytes — one byte short of v1
        let buf = vec![0u8; 64];
        let result = AliceFileHeader::from_bytes(&buf);
        assert_eq!(result, Err(FormatError::TooShort { expected: HEADER_V1_SIZE, got: 64 }));
    }

    #[test]
    fn test_invalid_file_type_discriminant() {
        let mut buf = build_v2_bytes(
            1, 1,
            0xFF, // invalid file_type
            0,
            AlicePayloadType::Procedural.to_u8(),
            0, 0, [0u8; 32], 0,
        );
        // Ensure magic is correct
        buf[..9].copy_from_slice(ALICE_MAGIC);

        let result = AliceFileHeader::from_bytes(&buf);
        assert_eq!(result, Err(FormatError::InvalidFileType(0xFF)));
    }

    // -----------------------------------------------------------------------
    // Individual enum roundtrips
    // -----------------------------------------------------------------------

    #[test]
    fn test_file_type_roundtrip() {
        let cases: &[(AliceFileType, u8)] = &[
            (AliceFileType::NumpyArray, 0x01),
            (AliceFileType::Image,      0x02),
            (AliceFileType::Audio,      0x03),
            (AliceFileType::Text,       0x04),
            (AliceFileType::Binary,     0x05),
        ];
        for &(ft, byte) in cases {
            assert_eq!(ft.to_u8(), byte, "{:?}.to_u8() should be 0x{:02X}", ft, byte);
            assert_eq!(AliceFileType::from_u8(byte).unwrap(), ft,
                "from_u8(0x{:02X}) should yield {:?}", byte, ft);
        }
        // Unknown value
        assert_eq!(
            AliceFileType::from_u8(0x00),
            Err(FormatError::InvalidFileType(0x00))
        );
        assert_eq!(
            AliceFileType::from_u8(0x06),
            Err(FormatError::InvalidFileType(0x06))
        );
    }

    #[test]
    fn test_payload_type_roundtrip() {
        let cases: &[(AlicePayloadType, u8)] = &[
            (AlicePayloadType::Procedural,   0x00),
            (AlicePayloadType::MediaImage,   0x10),
            (AlicePayloadType::MediaAudio,   0x11),
            (AlicePayloadType::MediaVideo,   0x12),
            (AlicePayloadType::Texture,      0x20),
            (AlicePayloadType::LzmaFallback, 0x30),
        ];
        for &(pt, byte) in cases {
            assert_eq!(pt.to_u8(), byte, "{:?}.to_u8() should be 0x{:02X}", pt, byte);
            assert_eq!(AlicePayloadType::from_u8_lenient(byte), pt,
                "from_u8_lenient(0x{:02X}) should yield {:?}", byte, pt);
            assert_eq!(AlicePayloadType::from_u8_strict(byte).unwrap(), pt,
                "from_u8_strict(0x{:02X}) should yield {:?}", byte, pt);
        }
        // Unknown value — lenient should fall back to Procedural, strict should error
        assert_eq!(
            AlicePayloadType::from_u8_lenient(0xFF),
            AlicePayloadType::Procedural
        );
        assert_eq!(
            AlicePayloadType::from_u8_strict(0xFF),
            Err(FormatError::InvalidPayloadType(0xFF))
        );
    }

    // -----------------------------------------------------------------------
    // Endianness / byte-order sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_little_endian_sizes() {
        // Verify that sizes are stored little-endian in the byte stream
        let hdr = AliceFileHeader::new(
            AliceFileType::Binary,
            AlicePayloadType::Procedural,
            0x0102_0304_0506_0708_u64,
            0xAABB_CCDD_EEFF_0011_u64,
        );
        let bytes = hdr.to_bytes();

        // In v2 format, original_size starts at byte 14
        let orig = u64::from_le_bytes(bytes[14..22].try_into().unwrap());
        assert_eq!(orig, 0x0102_0304_0506_0708_u64);

        // compressed_size starts at byte 22
        let comp = u64::from_le_bytes(bytes[22..30].try_into().unwrap());
        assert_eq!(comp, 0xAABB_CCDD_EEFF_0011_u64);
    }

    #[test]
    fn test_checksum_little_endian() {
        let mut hdr = AliceFileHeader::new(
            AliceFileType::Text,
            AlicePayloadType::Procedural,
            0, 0,
        );
        hdr.checksum = 0x1234_5678;
        let bytes = hdr.to_bytes();

        // checksum occupies bytes 62..66 in v2
        assert_eq!(bytes[62], 0x78); // LE: least significant byte first
        assert_eq!(bytes[63], 0x56);
        assert_eq!(bytes[64], 0x34);
        assert_eq!(bytes[65], 0x12);
    }

    // -----------------------------------------------------------------------
    // Display / Debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_impl() {
        let hdr = AliceFileHeader::new(
            AliceFileType::Image,
            AlicePayloadType::MediaImage,
            1024,
            512,
        );
        let s = hdr.to_string();
        assert!(s.contains("Image"),       "Display should mention file type");
        assert!(s.contains("MediaImage"),  "Display should mention payload type");
        assert!(s.contains("1024"),        "Display should mention original_size");
        assert!(s.contains("512"),         "Display should mention compressed_size");
    }

    #[test]
    fn test_format_error_display() {
        assert!(!FormatError::InvalidMagic.to_string().is_empty());
        assert!(!FormatError::TooShort { expected: 65, got: 10 }.to_string().is_empty());
        assert!(!FormatError::InvalidFileType(0xFF).to_string().is_empty());
        assert!(!FormatError::InvalidPayloadType(0x99).to_string().is_empty());
    }

    // -----------------------------------------------------------------------
    // has_valid_magic helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_has_valid_magic() {
        let good = AliceFileHeader::new(
            AliceFileType::Binary,
            AlicePayloadType::Procedural,
            0, 0,
        );
        assert!(good.has_valid_magic());

        let mut bad = good.clone();
        bad.magic[0] = b'X';
        assert!(!bad.has_valid_magic());
    }

    // -----------------------------------------------------------------------
    // header_size method
    // -----------------------------------------------------------------------

    #[test]
    fn test_header_size_v1_vs_v2() {
        // Fresh header is always v2
        let hdr_v2 = AliceFileHeader::new(
            AliceFileType::Binary,
            AlicePayloadType::Procedural,
            0, 0,
        );
        assert_eq!(hdr_v2.header_size(), HEADER_V2_SIZE);
        assert_eq!(hdr_v2.header_size(), 66);

        // Header parsed from v1 bytes
        let v1_buf = build_v1_bytes(
            1, 0,
            AliceFileType::Text.to_u8(),
            0, 0, 0, [0u8; 32], 0,
        );
        let hdr_v1 = AliceFileHeader::from_bytes(&v1_buf).unwrap();
        assert_eq!(hdr_v1.header_size(), HEADER_V1_SIZE);
        assert_eq!(hdr_v1.header_size(), 65);
    }
}
