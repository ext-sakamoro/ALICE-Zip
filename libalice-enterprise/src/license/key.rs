//! License Key Generation and Parsing

use alice_ephemeris::time::{CivilDate, Timestamp};
use alice_ephemeris::{civil_from_mjd, mjd_from_civil};
use bitflags::bitflags;
use sha2::{Digest, Sha256};
use thiserror::Error;

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

    /// A date outside what a key can carry (years 0..=9999) or outside the calendar
    #[error("date out of range for a license key")]
    DateOutOfRange,
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

/// The current date in UTC (the only place the license code reads the clock; every date
/// computation takes `today` as an argument)
#[allow(
    clippy::disallowed_methods,
    reason = "license expiry is a wall-clock date by definition; read once here and passed in everywhere else"
)]
fn today_utc() -> Result<CivilDate, LicenseError> {
    utc_date(Timestamp::from_system_time(std::time::SystemTime::now()))
}

/// The UTC calendar date of `t`
fn utc_date(t: Timestamp) -> Result<CivilDate, LicenseError> {
    t.civil_date_at_offset(0)
        .ok_or(LicenseError::DateOutOfRange)
}

/// The expiry date of a trial of `days` days started on `today`: `days` calendar days later.
/// [`LicenseError::DateOutOfRange`] when that date is outside the years a key carries (0..=9999)
fn trial_expiry(today: CivilDate, days: u16) -> Result<CivilDate, LicenseError> {
    let out = |_| LicenseError::DateOutOfRange;
    let start = mjd_from_civil(today.year(), today.month(), today.day()).map_err(out)?;
    let (year, month, day) = civil_from_mjd(start + i64::from(days)).map_err(out)?;
    if !(0..=9999).contains(&year) {
        return Err(LicenseError::DateOutOfRange);
    }
    CivilDate::from_ymd(year, month, day).map_err(out)
}

/// The `YYYYMMDD` text of a date in the key (years 0..=9999, the only ones a key can carry)
fn expiry_text(d: CivilDate) -> String {
    format!("{:04}{:02}{:02}", d.year(), d.month(), d.day())
}

/// The date of an 8-digit `YYYYMMDD` text; `None` for any other text or a day not in the calendar
fn parse_expiry(text: &str) -> Option<CivilDate> {
    if text.len() != 8 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let year = text[0..4].parse::<i64>().ok()?;
    let month = text[4..6].parse::<u8>().ok()?;
    let day = text[6..8].parse::<u8>().ok()?;
    CivilDate::from_ymd(year, month, day).ok()
}

/// Secret salt for checksum (in production, use environment variable)
const LICENSE_SALT: &str = "ALICE-ZIP-ENT-2026-SAKAMOTO";

/// Parsed license key
#[derive(Debug, Clone)]
pub struct LicenseKey {
    license_type: LicenseType,
    features: Feature,
    expiry_date: Option<CivilDate>,
    raw_key: String,
}

impl LicenseKey {
    /// Generate a new license key
    pub fn generate(license_type: LicenseType, expiry: Option<CivilDate>) -> String {
        let features = license_type.default_features();
        Self::generate_with_features(license_type, features, expiry)
    }

    /// Generate a new license key with custom features
    pub fn generate_with_features(
        license_type: LicenseType,
        features: Feature,
        expiry: Option<CivilDate>,
    ) -> String {
        let type_code = license_type.code();
        let features_hex = format!("{:02X}", features.bits());
        let expiry_str = expiry
            .map(expiry_text)
            .unwrap_or_else(|| "00000000".to_string());

        let checksum = Self::compute_checksum(type_code, &features_hex, &expiry_str);

        format!(
            "ALZ-{}-{}-{}-{}",
            type_code, features_hex, expiry_str, checksum
        )
    }

    /// Generate trial license key
    ///
    /// [`LicenseError::DateOutOfRange`] when the expiry falls outside the years a key carries
    pub fn generate_trial(days: u16) -> Result<String, LicenseError> {
        let expiry = trial_expiry(today_utc()?, days)?;
        Ok(Self::generate(LicenseType::Trial, Some(expiry)))
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
        let features_int =
            u16::from_str_radix(features_hex, 16).map_err(|_| LicenseError::InvalidFormat)?;
        let features = Feature::from_bits(features_int).ok_or(LicenseError::InvalidFormat)?;

        // Validate expiry
        let expiry_date = if expiry_str == "00000000" {
            None
        } else {
            Some(parse_expiry(expiry_str).ok_or(LicenseError::InvalidFormat)?)
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
    pub fn expiry_date(&self) -> Option<CivilDate> {
        self.expiry_date
    }

    /// Check if license is perpetual
    pub fn is_perpetual(&self) -> bool {
        self.expiry_date.is_none()
    }

    /// Check if license is expired (an error when today's date cannot be read)
    pub fn is_expired(&self) -> Result<bool, LicenseError> {
        Ok(self.is_expired_on(today_utc()?))
    }

    /// Whether the license is expired on `today` (the day after its expiry date and later)
    fn is_expired_on(&self, today: CivilDate) -> bool {
        self.expiry_date.map(|d| today > d).unwrap_or(false)
    }

    /// Get days remaining until expiry (an error when today's date cannot be read)
    pub fn days_remaining(&self) -> Result<Option<i64>, LicenseError> {
        Ok(self.days_remaining_on(today_utc()?))
    }

    /// Whole days from `today` to the expiry date, 0 on and after it
    fn days_remaining_on(&self, today: CivilDate) -> Option<i64> {
        self.expiry_date.map(|d| today.days_until(d).max(0))
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
        assert!(!parsed.is_expired().unwrap());
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
        let corrupted = format!("{}XXXX", &key[..key.len() - 4]);
        assert!(LicenseKey::parse(&corrupted).is_err());
    }

    #[test]
    fn test_trial_expiry() {
        let key = LicenseKey::generate_trial(14).unwrap();
        let parsed = LicenseKey::parse(&key).unwrap();

        assert_eq!(parsed.license_type(), LicenseType::Trial);
        assert!(!parsed.is_perpetual());
        assert!(!parsed.is_expired().unwrap());
        assert!(parsed.days_remaining().unwrap().unwrap() <= 14);
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

/// Goldens of the key text and the date arithmetic. The expected keys were computed outside this
/// crate (Python: SHA-256 of `TYPE-FEATURES-YYYYMMDD-salt`, first two bytes in upper-case hex) and
/// the expected dates with Python's `datetime`, so they hold whatever date type the code uses.
#[cfg(test)]
mod golden_tests {
    use super::*;

    fn ymd(y: i64, m: u8, d: u8) -> CivilDate {
        CivilDate::from_ymd(y, m, d).expect("a valid test date")
    }

    #[test]
    fn keys_for_fixed_dates() {
        let cases = [
            (
                LicenseType::Professional,
                0x05,
                Some(ymd(2026, 12, 31)),
                "ALZ-PRO-05-20261231-CB94",
            ),
            (
                LicenseType::Trial,
                0x01,
                Some(ymd(2024, 2, 29)),
                "ALZ-TRL-01-20240229-5560",
            ),
            (
                LicenseType::Enterprise,
                0xFF,
                None,
                "ALZ-ENT-FF-00000000-9B23",
            ),
            (
                LicenseType::Standard,
                0x03,
                Some(ymd(9999, 12, 31)),
                "ALZ-STD-03-99991231-8C1F",
            ),
        ];
        for (t, bits, expiry, want) in cases {
            let key =
                LicenseKey::generate_with_features(t, Feature::from_bits_truncate(bits), expiry);
            assert_eq!(key, want);
            let parsed = LicenseKey::parse(&key).expect("a generated key parses");
            assert_eq!(parsed.license_type(), t);
            assert_eq!(parsed.features().bits(), bits);
            assert_eq!(parsed.expiry_date(), expiry);
            assert_eq!(parsed.is_perpetual(), expiry.is_none());
        }
    }

    #[test]
    fn keys_whose_date_is_not_a_calendar_date_are_refused() {
        // each has the right checksum for its text, so only the date can refuse it
        for key in [
            "ALZ-PRO-05-20230229-880E",
            "ALZ-PRO-05-20261301-C730",
            "ALZ-PRO-05-20260000-5437",
            "ALZ-PRO-05-20260431-BD46",
        ] {
            assert!(
                matches!(LicenseKey::parse(key), Err(LicenseError::InvalidFormat)),
                "{key}"
            );
        }
    }

    #[test]
    fn expiry_texts_that_are_not_eight_digits_are_refused() {
        // checksums computed outside the crate for each text, so only the date form can refuse
        // it: 7 digits, 9 digits, a space, a sign
        for key in [
            "ALZ-PRO-05-2026123-DB87",
            "ALZ-PRO-05-202612311-2768",
            "ALZ-PRO-05-2026 231-18F1",
            "ALZ-PRO-05-+2026123-A2C6",
        ] {
            assert!(
                matches!(LicenseKey::parse(key), Err(LicenseError::InvalidFormat)),
                "{key}"
            );
        }
    }

    #[test]
    fn trial_expiry_adds_whole_days() {
        for (start, days, want) in [
            (ymd(2026, 11, 25), 14, ymd(2026, 12, 9)),
            (ymd(2024, 2, 20), 14, ymd(2024, 3, 5)),
            (ymd(2023, 12, 25), 10, ymd(2024, 1, 4)),
            (ymd(2026, 1, 1), 0, ymd(2026, 1, 1)),
            (ymd(2026, 1, 1), u16::MAX, ymd(2205, 6, 7)),
        ] {
            assert_eq!(trial_expiry(start, days).unwrap(), want, "{start} + {days}");
        }
    }

    #[test]
    fn trial_expiry_outside_the_years_a_key_carries_is_an_error() {
        for (start, days) in [
            (ymd(9999, 12, 31), 1),
            (ymd(9900, 1, 1), u16::MAX),
            (ymd(-1, 12, 1), 30),
            (ymd(1_000_000_000, 12, 31), 0),
        ] {
            assert!(
                matches!(trial_expiry(start, days), Err(LicenseError::DateOutOfRange)),
                "{start} + {days}"
            );
        }
        // the last day a key carries is still a date
        assert_eq!(
            trial_expiry(ymd(9999, 12, 30), 1).unwrap(),
            ymd(9999, 12, 31)
        );
        assert_eq!(trial_expiry(ymd(-1, 12, 31), 1).unwrap(), ymd(0, 1, 1));
    }

    #[test]
    fn the_utc_date_of_every_timestamp_is_read() {
        // the Timestamp range lies inside the calendar's, so the error branch is not reached from a
        // Timestamp; both ends of the range give a date
        assert_eq!(utc_date(Timestamp::UNIX_EPOCH).unwrap(), ymd(1970, 1, 1));
        assert_eq!(utc_date(Timestamp::MIN).unwrap(), ymd(-262_143, 1, 1));
        assert_eq!(utc_date(Timestamp::MAX).unwrap(), ymd(262_142, 12, 31));
    }

    #[test]
    fn expiry_and_days_remaining_on_a_given_day() {
        let key = LicenseKey::parse("ALZ-PRO-05-20261231-CB94").expect("golden key");
        for (today, expired, left) in [
            (ymd(2026, 1, 1), false, 364),
            (ymd(2026, 12, 30), false, 1),
            (ymd(2026, 12, 31), false, 0),
            (ymd(2027, 1, 1), true, 0),
            (ymd(2030, 6, 1), true, 0),
        ] {
            assert_eq!(key.is_expired_on(today), expired, "{today}");
            assert_eq!(key.days_remaining_on(today), Some(left), "{today}");
        }
        let perpetual = LicenseKey::parse("ALZ-ENT-FF-00000000-9B23").expect("golden key");
        assert!(!perpetual.is_expired_on(ymd(9999, 12, 31)));
        assert_eq!(perpetual.days_remaining_on(ymd(2026, 1, 1)), None);
    }
}
