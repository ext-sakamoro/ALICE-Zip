//! Key Derivation Functions

use hkdf::Hkdf;
use sha2::Sha256;
use thiserror::Error;
use zeroize::Zeroize;

/// KDF errors
#[derive(Error, Debug)]
pub enum KdfError {
    /// Invalid key length
    #[error("invalid key length: expected {expected}, got {got}")]
    InvalidLength { expected: usize, got: usize },

    /// Derivation failed
    #[error("key derivation failed")]
    DerivationFailed,
}

/// Derived key material
#[derive(Clone, Zeroize)]
#[zeroize(drop)]
pub struct DerivedKey {
    bytes: Vec<u8>,
}

impl DerivedKey {
    /// Create from bytes
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Get key bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get fixed-size array (32 bytes)
    pub fn as_array_32(&self) -> Result<[u8; 32], KdfError> {
        if self.bytes.len() < 32 {
            return Err(KdfError::InvalidLength {
                expected: 32,
                got: self.bytes.len(),
            });
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&self.bytes[..32]);
        Ok(arr)
    }
}

/// Key derivation with HKDF-SHA256
pub struct KeyDerivation {
    hkdf: Hkdf<Sha256>,
}

impl KeyDerivation {
    /// Create new KDF from input key material
    pub fn new(ikm: &[u8], salt: Option<&[u8]>) -> Self {
        let hkdf = Hkdf::<Sha256>::new(salt, ikm);
        Self { hkdf }
    }

    /// Derive a key with given info string
    pub fn derive(&self, info: &[u8], length: usize) -> Result<DerivedKey, KdfError> {
        let mut okm = vec![0u8; length];
        self.hkdf
            .expand(info, &mut okm)
            .map_err(|_| KdfError::DerivationFailed)?;
        Ok(DerivedKey::new(okm))
    }

    /// Derive encryption key (32 bytes)
    pub fn derive_encryption_key(&self) -> Result<DerivedKey, KdfError> {
        self.derive(b"alice-zip-encryption", 32)
    }

    /// Derive authentication key (32 bytes)
    pub fn derive_auth_key(&self) -> Result<DerivedKey, KdfError> {
        self.derive(b"alice-zip-authentication", 32)
    }

    /// Derive archive key (32 bytes)
    pub fn derive_archive_key(&self) -> Result<DerivedKey, KdfError> {
        self.derive(b"alice-zip-archive", 32)
    }

    /// Derive all keys at once
    pub fn derive_all_keys(&self) -> Result<DerivedKeys, KdfError> {
        Ok(DerivedKeys {
            encryption: self.derive_encryption_key()?,
            authentication: self.derive_auth_key()?,
            archive: self.derive_archive_key()?,
        })
    }
}

/// Collection of derived keys
pub struct DerivedKeys {
    /// Encryption key
    pub encryption: DerivedKey,
    /// Authentication key
    pub authentication: DerivedKey,
    /// Archive key
    pub archive: DerivedKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key() {
        let kdf = KeyDerivation::new(b"master-secret", Some(b"salt"));

        let key1 = kdf.derive(b"purpose-1", 32).unwrap();
        let key2 = kdf.derive(b"purpose-2", 32).unwrap();

        assert_eq!(key1.as_bytes().len(), 32);
        assert_eq!(key2.as_bytes().len(), 32);
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_deterministic_derivation() {
        let kdf1 = KeyDerivation::new(b"secret", Some(b"salt"));
        let kdf2 = KeyDerivation::new(b"secret", Some(b"salt"));

        let key1 = kdf1.derive(b"info", 32).unwrap();
        let key2 = kdf2.derive(b"info", 32).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_derive_all_keys() {
        let kdf = KeyDerivation::new(b"master", None);
        let keys = kdf.derive_all_keys().unwrap();

        assert_eq!(keys.encryption.as_bytes().len(), 32);
        assert_eq!(keys.authentication.as_bytes().len(), 32);
        assert_eq!(keys.archive.as_bytes().len(), 32);

        // All keys should be different
        assert_ne!(keys.encryption.as_bytes(), keys.authentication.as_bytes());
        assert_ne!(keys.encryption.as_bytes(), keys.archive.as_bytes());
    }

    #[test]
    fn test_as_array_32() {
        let kdf = KeyDerivation::new(b"secret", None);
        let key = kdf.derive(b"test", 32).unwrap();

        let arr = key.as_array_32().unwrap();
        assert_eq!(arr.len(), 32);
    }

    #[test]
    fn test_as_array_32_too_short() {
        let kdf = KeyDerivation::new(b"secret", None);
        let key = kdf.derive(b"test", 16).unwrap();

        assert!(key.as_array_32().is_err());
    }
}
