//! Error type for ALICE-Zip

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZipError {
    InvalidData,
    DecompressFailed,
    DictionaryFull,
}

impl core::fmt::Display for ZipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidData => write!(f, "invalid data"),
            Self::DecompressFailed => write!(f, "decompress failed"),
            Self::DictionaryFull => write!(f, "dictionary full"),
        }
    }
}

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

    /// Clone
    #[test]
    fn zip_error_clone() {
        let e = ZipError::InvalidData;
        let e2 = e.clone();
        assert_eq!(e, e2);
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
