//! License Management Module
//!
//! License key validation and feature gating for ALICE-Zip Enterprise.
//!
//! # Key Format
//!
//! ```text
//! ALZ-{TYPE}-{FEATURES}-{EXPIRY}-{CHECKSUM}
//!
//! TYPE: TRL (Trial), STD (Standard), PRO (Professional), ENT (Enterprise)
//! FEATURES: Hex-encoded feature flags
//! EXPIRY: YYYYMMDD or 00000000 (perpetual)
//! CHECKSUM: SHA256-based 4-char checksum
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_enterprise::license::{LicenseKey, LicenseType, Feature};
//!
//! // Generate a license key
//! let key = LicenseKey::generate(LicenseType::Professional, None);
//!
//! // Parse and validate
//! let license = LicenseKey::parse(&key)?;
//! assert!(license.has_feature(Feature::Encryption));
//! ```

mod key;

pub use key::{
    LicenseKey,
    LicenseType,
    Feature,
    LicenseError,
};

/// Global license state management
pub mod global {
    use super::{LicenseKey, Feature};
    use std::sync::RwLock;

    static CURRENT_LICENSE: RwLock<Option<LicenseKey>> = RwLock::new(None);

    /// Initialize with a license key
    pub fn init(key: &str) -> Result<(), super::LicenseError> {
        let license = LicenseKey::parse(key)?;
        if license.is_expired() {
            return Err(super::LicenseError::Expired);
        }
        let mut guard = CURRENT_LICENSE.write().unwrap();
        *guard = Some(license);
        Ok(())
    }

    /// Activate trial license
    pub fn activate_trial(days: u16) {
        let key = LicenseKey::generate_trial(days);
        let _ = init(&key);
    }

    /// Check if a feature is available
    pub fn has_feature(feature: &str) -> bool {
        let guard = CURRENT_LICENSE.read().unwrap();
        guard.as_ref().map(|l| {
            let f = match feature.to_lowercase().as_str() {
                "encryption" | "security" => Feature::ENCRYPTION,
                "access_control" | "access" => Feature::ACCESS_CONTROL,
                "audit" => Feature::AUDIT,
                "parallel" => Feature::PARALLEL,
                "quantum" => Feature::QUANTUM,
                "api" => Feature::API,
                _ => Feature::empty(),
            };
            l.has_feature(f)
        }).unwrap_or(false)
    }

    /// Get current license
    pub fn get_license() -> Option<LicenseKey> {
        CURRENT_LICENSE.read().unwrap().clone()
    }

    /// Clear current license
    pub fn clear() {
        let mut guard = CURRENT_LICENSE.write().unwrap();
        *guard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_parse() {
        let key = LicenseKey::generate(LicenseType::Professional, None);
        let parsed = LicenseKey::parse(&key).unwrap();
        assert_eq!(parsed.license_type(), LicenseType::Professional);
        assert!(!parsed.is_expired());
    }

    #[test]
    fn test_trial_key() {
        let key = LicenseKey::generate_trial(14);
        let parsed = LicenseKey::parse(&key).unwrap();
        assert_eq!(parsed.license_type(), LicenseType::Trial);
        assert!(parsed.days_remaining().unwrap() <= 14);
    }

    #[test]
    fn test_global_api() {
        global::activate_trial(30);
        assert!(global::has_feature("encryption"));
        global::clear();
    }
}
