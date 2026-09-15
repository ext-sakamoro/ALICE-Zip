//! Error type for ALICE-Zip

/// Errors returned by the `no_std` primitives (`lz77` / `dictionary`)
/// The zlib wrappers in [`crate::compression`] return `std::io::Error`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZipError {
    /// A token stream references bytes that do not exist yet
    InvalidData,
    /// Generic decompression failure
    DecompressFailed,
    /// [`crate::Dictionary`] cannot accept another phrase (capacity `0` or
    /// `u32` arena offset range exhausted)
    DictionaryFull,
    /// A generator was given a parameter outside its domain (non-positive
    /// `scale`, `octaves == 0`, texture size overflow)
    InvalidParameter,
}

impl core::fmt::Display for ZipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidData => write!(f, "invalid data"),
            Self::DecompressFailed => write!(f, "decompress failed"),
            Self::DictionaryFull => write!(f, "dictionary full"),
            Self::InvalidParameter => write!(f, "invalid generator parameter"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ZipError {}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
)]
mod tests {
    use super::*;

    /// Display出力の確認
    #[test]
    fn zip_error_display_invalid_data() {
        let e = ZipError::InvalidData;
        let s = alloc::format!("{e}");
        assert_eq!(s, "invalid data");
    }

    #[test]
    fn zip_error_display_decompress_failed() {
        let e = ZipError::DecompressFailed;
        let s = alloc::format!("{e}");
        assert_eq!(s, "decompress failed");
    }

    #[test]
    fn zip_error_display_dictionary_full() {
        let e = ZipError::DictionaryFull;
        let s = alloc::format!("{e}");
        assert_eq!(s, "dictionary full");
    }

    /// Copy
    #[test]
    fn zip_error_copy() {
        let e = ZipError::InvalidData;
        let e2 = e;
        assert_eq!(e, e2);
        assert_eq!(
            alloc::format!("{}", ZipError::InvalidParameter),
            "invalid generator parameter"
        );
    }

    /// PartialEq
    #[test]
    fn zip_error_eq() {
        assert_eq!(ZipError::InvalidData, ZipError::InvalidData);
        assert_ne!(ZipError::InvalidData, ZipError::DecompressFailed);
        assert_ne!(ZipError::DecompressFailed, ZipError::DictionaryFull);
    }

    /// Debug出力
    #[test]
    fn zip_error_debug() {
        let s = alloc::format!("{:?}", ZipError::InvalidData);
        assert!(s.contains("InvalidData"));
    }
}
