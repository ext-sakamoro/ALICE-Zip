//! License Key Generation and Parsing

use sha2::{Sha256, Digest};
use thiserror::Error;
use chrono::{NaiveDate, Utc};
use bitflags::bitflags;

/// License validation errors
#[derive(Error, Debug)]
pub enum LicenseError {
    /// Invalid key format
    #[error("invalid license key format")]
    InvalidFormat,

    /// Invalid checksum
    #[error("invalid license key checksum")]
    InvalidChecksum,

    /// License expired
    #[error("license has expired")]
    Expired,

    /// Invalid license type
    #[error("invalid license type: {0}")]
    InvalidType(String),

    /// Feature not available
    #[error("feature not available: {0}")]
    FeatureNotAvailable(String),
}

/// License types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenseType {
    /// 14-day trial
    Trial,
    /// Standard edition
    Standard,
    /// Professional edition
    Professional,
    /// Enterprise edition
    Enterprise,
}

impl LicenseType {
    /// Get type code for key generation
    pub fn code(&self) -> &'static str {
        match self {
            Self::Trial => "TRL",
            Self::Standard => "STD",
            Self::Professional => "PRO",
            Self::Enterprise => "ENT",
        }
    }

    /// Parse from code
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "TRL" => Some(Self::Trial),
            "STD" => Some(Self::Standard),
            "PRO" => Some(Self::Professional),
            "ENT" => Some(Self::Enterprise),
            _ => None,
        }
    }

    /// Get default features for this license type
    pub fn default_features(&self) -> Feature {
        match self {
            Self::Trial => Feature::trial_features(),
            Self::Standard => Feature::standard_features(),
            Self::Professional => Feature::professional_features(),
            Self::Enterprise => Feature::enterprise_features(),
        }
    }
}

bitflags! {
    /// Feature flags for license validation
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Feature: u16 {
        /// Encrypted archive support
        const ENCRYPTION = 0x01;
        /// Access control
        const ACCESS_CONTROL = 0x02;
        /// Audit logging
        const AUDIT = 0x04;
        /// Parallel compression
        const PARALLEL = 0x08;
        /// Advanced algorithms
        const ADVANCED = 0x10;
        /// Quantum-inspired optimization
        const QUANTUM = 0x20;
        /// API access
        const API = 0x40;
        /// Priority support
        const PRIORITY_SUPPORT = 0x80;
    }
}

impl Feature {
    /// Standard license features
    pub fn standard_features() -> Self {
        Self::ENCRYPTION | Self::PARALLEL
    }

    /// Professional license features
    pub fn professional_features() -> Self {
        Self::standard_features() | Self::ACCESS_CONTROL | Self::AUDIT | Self::ADVANCED
    }

    /// Enterprise license features
    pub fn enterprise_features() -> Self {
        Self::professional_features() | Self::QUANTUM | Self::API | Self::PRIORITY_SUPPORT
    }

    /// Trial features (same as Professional)
    pub fn trial_features() -> Self {
        Self::professional_features()
    }
}

/// Secret salt for checksum (in production, use environment variable)
const LICENSE_SALT: &str = "ALICE-ZIP-ENT-2026-SAKAMOTO";

/// Parsed license key
#[derive(Debug, Clone)]
pub struct LicenseKey {
    license_type: LicenseType,
    features: Feature,
    expiry_date: Option<NaiveDate>,
    raw_key: String,
}

impl LicenseKey {
    /// Generate a new license key
    pub fn generate(license_type: LicenseType, expiry: Option<NaiveDate>) -> String {
        let features = license_type.default_features();
        Self::generate_with_features(license_type, features, expiry)
    }

    /// Generate a new license key with custom features
    pub fn generate_with_features(
        license_type: LicenseType,
        features: Feature,
        expiry: Option<NaiveDate>,
    ) -> String {
        let type_code = license_type.code();
        let features_hex = format!("{:02X}", features.bits());
        let expiry_str = expiry
            .map(|d| d.format("%Y%m%d").to_string())
            .unwrap_or_else(|| "00000000".to_string());

        let checksum = Self::compute_checksum(type_code, &features_hex, &expiry_str);

        format!("ALZ-{}-{}-{}-{}", type_code, features_hex, expiry_str, checksum)
    }

    /// Generate trial license key
    pub fn generate_trial(days: u16) -> String {
        let expiry = Utc::now().date_naive() + chrono::Duration::days(days as i64);
        Self::generate(LicenseType::Trial, Some(expiry))
    }

    /// Parse a license key
    pub fn parse(key: &str) -> Result<Self, LicenseError> {
        let key = key.trim().to_uppercase();
        let parts: Vec<&str> = key.split('-').collect();

        if parts.len() != 5 {
            return Err(LicenseError::InvalidFormat);
        }

        let (prefix, type_code, features_hex, expiry_str, checksum) =
            (parts[0], parts[1], parts[2], parts[3], parts[4]);

        // Validate prefix
        if prefix != "ALZ" {
            return Err(LicenseError::InvalidFormat);
        }

        // Validate license type
        let license_type = LicenseType::from_code(type_code)
            .ok_or_else(|| LicenseError::InvalidType(type_code.to_string()))?;

        // Validate features
        let features_int = u16::from_str_radix(features_hex, 16)
            .map_err(|_| LicenseError::InvalidFormat)?;
        let features = Feature::from_bits(features_int)
            .ok_or(LicenseError::InvalidFormat)?;

        // Validate expiry
        let expiry_date = if expiry_str == "00000000" {
            None
        } else {
            Some(NaiveDate::parse_from_str(expiry_str, "%Y%m%d")
                .map_err(|_| LicenseError::InvalidFormat)?)
        };

        // Validate checksum
        let expected_checksum = Self::compute_checksum(type_code, features_hex, expiry_str);
        if checksum != expected_checksum {
            return Err(LicenseError::InvalidChecksum);
        }

        Ok(Self {
            license_type,
            features,
            expiry_date,
            raw_key: key,
        })
    }

    /// Compute checksum for license key
    fn compute_checksum(type_code: &str, features_hex: &str, expiry: &str) -> String {
        let data = format!("{}-{}-{}-{}", type_code, features_hex, expiry, LICENSE_SALT);
        let hash = Sha256::digest(data.as_bytes());
        // Take first 2 bytes as hex (4 chars)
        format!("{:02X}{:02X}", hash[0], hash[1])
    }

    /// Get license type
    pub fn license_type(&self) -> LicenseType {
        self.license_type
    }

    /// Get features
    pub fn features(&self) -> Feature {
        self.features
    }

    /// Check if license has a specific feature
    pub fn has_feature(&self, feature: Feature) -> bool {
        self.features.contains(feature)
    }

    /// Get expiry date
    pub fn expiry_date(&self) -> Option<NaiveDate> {
        self.expiry_date
    }

    /// Check if license is perpetual
    pub fn is_perpetual(&self) -> bool {
        self.expiry_date.is_none()
    }

    /// Check if license is expired
    pub fn is_expired(&self) -> bool {
        self.expiry_date
            .map(|d| Utc::now().date_naive() > d)
            .unwrap_or(false)
    }

    /// Get days remaining until expiry
    pub fn days_remaining(&self) -> Option<i64> {
        self.expiry_date.map(|d| {
            let today = Utc::now().date_naive();
            (d - today).num_days().max(0)
        })
    }

    /// Get raw key string
    pub fn raw_key(&self) -> &str {
        &self.raw_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_standard_key() {
        let key = LicenseKey::generate(LicenseType::Standard, None);
        assert!(key.starts_with("ALZ-STD-"));
        assert_eq!(key.split('-').count(), 5);
    }

    #[test]
    fn test_generate_professional_key() {
        let key = LicenseKey::generate(LicenseType::Professional, None);
        assert!(key.contains("PRO"));
    }

    #[test]
    fn test_generate_enterprise_key() {
        let key = LicenseKey::generate(LicenseType::Enterprise, None);
        assert!(key.contains("ENT"));
    }

    #[test]
    fn test_parse_valid_key() {
        let key = LicenseKey::generate(LicenseType::Professional, None);
        let parsed = LicenseKey::parse(&key).unwrap();

        assert_eq!(parsed.license_type(), LicenseType::Professional);
        assert!(parsed.is_perpetual());
        assert!(!parsed.is_expired());
    }

    #[test]
    fn test_parse_invalid_key() {
        assert!(LicenseKey::parse("").is_err());
        assert!(LicenseKey::parse("invalid").is_err());
        assert!(LicenseKey::parse("ALZ-XXX-00-00000000-0000").is_err());
    }

    #[test]
    fn test_corrupted_checksum() {
        let key = LicenseKey::generate(LicenseType::Professional, None);
        let corrupted = format!("{}XXXX", &key[..key.len()-4]);
        assert!(LicenseKey::parse(&corrupted).is_err());
    }

    #[test]
    fn test_trial_expiry() {
        let key = LicenseKey::generate_trial(14);
        let parsed = LicenseKey::parse(&key).unwrap();

        assert_eq!(parsed.license_type(), LicenseType::Trial);
        assert!(!parsed.is_perpetual());
        assert!(!parsed.is_expired());
        assert!(parsed.days_remaining().unwrap() <= 14);
    }

    #[test]
    fn test_feature_flags() {
        let key = LicenseKey::generate(LicenseType::Professional, None);
        let parsed = LicenseKey::parse(&key).unwrap();

        assert!(parsed.has_feature(Feature::ENCRYPTION));
        assert!(parsed.has_feature(Feature::ACCESS_CONTROL));
        assert!(!parsed.has_feature(Feature::QUANTUM));
    }

    #[test]
    fn test_enterprise_features() {
        let key = LicenseKey::generate(LicenseType::Enterprise, None);
        let parsed = LicenseKey::parse(&key).unwrap();

        assert!(parsed.has_feature(Feature::ENCRYPTION));
        assert!(parsed.has_feature(Feature::QUANTUM));
        assert!(parsed.has_feature(Feature::API));
    }
}
