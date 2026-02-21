//! ALICE-Zip Enterprise Edition
//!
//! **PROPRIETARY AND CONFIDENTIAL**
//!
//! Commercial features for ALICE-Zip compression:
//!
//! - **License Management**: License key validation and feature gating
//! - **Security**: Encrypted archive support (ChaCha20-Poly1305)
//! - **Access Control**: Permission-based archive access
//!
//! # License
//!
//! This software requires a valid Commercial License.
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_enterprise::{License, SecureArchive};
//!
//! // Validate license
//! let license = License::from_key("ALZ-PRO-3F-00000000-XXXX")?;
//!
//! // Create encrypted archive
//! let archive = SecureArchive::new(&license)?;
//! archive.compress_encrypted(&data, &password)?;
//! ```

#![warn(missing_docs)]

pub mod license;
pub mod security;

#[cfg(feature = "access-control")]
pub mod access;

// Re-export core library
pub use alice_core;

/// Enterprise library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Validate a license key
///
/// Returns `true` if the license is valid and not expired.
pub fn validate_license(key: &str) -> bool {
    license::LicenseKey::parse(key)
        .map(|k| !k.is_expired())
        .unwrap_or(false)
}

/// Activate trial license
///
/// Returns trial license key valid for specified days.
pub fn activate_trial(days: u16) -> String {
    license::LicenseKey::generate_trial(days)
}

/// Check if a feature is available
pub fn has_feature(feature: &str) -> bool {
    license::global::has_feature(feature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_core_reexport() {
        // Verify we can access core library
        assert!(true);
    }
}
