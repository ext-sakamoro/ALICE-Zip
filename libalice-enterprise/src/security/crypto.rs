//! Cryptographic primitives for archive encryption

use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce,
};
use rand_core::RngCore;
use thiserror::Error;
use zeroize::Zeroize;

/// Cryptographic errors
#[derive(Error, Debug)]
pub enum CryptoError {
    /// Encryption failed
    #[error("encryption failed")]
    EncryptionFailed,

    /// Decryption failed (wrong key or corrupted data)
    #[error("decryption failed: authentication tag mismatch")]
    DecryptionFailed,

    /// Invalid data format
    #[error("invalid encrypted data format")]
    InvalidFormat,

    /// Key derivation error
    #[error("key derivation error: {0}")]
    KeyDerivation(String),
}

/// Encryption key (256-bit)
#[derive(Clone, Zeroize)]
#[zeroize(drop)]
pub struct EncryptionKey {
    bytes: [u8; 32],
}

impl EncryptionKey {
    /// Create key from raw bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Derive key from password using HKDF
    pub fn from_password(password: &[u8], salt: &[u8]) -> Self {
        use hkdf::Hkdf;
        use sha2::Sha256;

        let hk = Hkdf::<Sha256>::new(Some(salt), password);
        let mut key_bytes = [0u8; 32];
        hk.expand(b"alice-zip-encryption-key", &mut key_bytes)
            .expect("32 bytes is valid output length");

        Self { bytes: key_bytes }
    }

    /// Generate random key
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        OsRng.fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Get key bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

/// Encrypted data with nonce
#[derive(Debug, Clone)]
pub struct EncryptedData {
    /// 12-byte nonce
    nonce: [u8; 12],
    /// Ciphertext with authentication tag
    ciphertext: Vec<u8>,
}

impl EncryptedData {
    /// Nonce size in bytes
    pub const NONCE_SIZE: usize = 12;
    /// Authentication tag size in bytes
    pub const TAG_SIZE: usize = 16;
    /// Minimum overhead (nonce + tag)
    pub const OVERHEAD: usize = Self::NONCE_SIZE + Self::TAG_SIZE;

    /// Create from components
    pub fn new(nonce: [u8; 12], ciphertext: Vec<u8>) -> Self {
        Self { nonce, ciphertext }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::NONCE_SIZE + self.ciphertext.len());
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&self.ciphertext);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() < Self::OVERHEAD {
            return Err(CryptoError::InvalidFormat);
        }

        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&bytes[..12]);
        let ciphertext = bytes[12..].to_vec();

        Ok(Self { nonce, ciphertext })
    }

    /// Get nonce
    pub fn nonce(&self) -> &[u8; 12] {
        &self.nonce
    }

    /// Get ciphertext
    pub fn ciphertext(&self) -> &[u8] {
        &self.ciphertext
    }
}

/// Archive encryptor using ChaCha20-Poly1305
pub struct ArchiveEncryptor {
    cipher: ChaCha20Poly1305,
}

impl ArchiveEncryptor {
    /// Create new encryptor with key
    pub fn new(key: &EncryptionKey) -> Self {
        let cipher = ChaCha20Poly1305::new_from_slice(key.as_bytes())
            .expect("32-byte key is valid");
        Self { cipher }
    }

    /// Encrypt plaintext
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedData, CryptoError> {
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = self.cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| CryptoError::EncryptionFailed)?;

        Ok(EncryptedData::new(nonce_bytes, ciphertext))
    }

    /// Encrypt with additional authenticated data (AAD)
    pub fn encrypt_with_aad(
        &self,
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<EncryptedData, CryptoError> {
        use chacha20poly1305::aead::Payload;

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let payload = Payload { msg: plaintext, aad };
        let ciphertext = self.cipher
            .encrypt(nonce, payload)
            .map_err(|_| CryptoError::EncryptionFailed)?;

        Ok(EncryptedData::new(nonce_bytes, ciphertext))
    }

    /// Decrypt ciphertext
    pub fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>, CryptoError> {
        let nonce = Nonce::from_slice(&encrypted.nonce);

        self.cipher
            .decrypt(nonce, encrypted.ciphertext.as_slice())
            .map_err(|_| CryptoError::DecryptionFailed)
    }

    /// Decrypt with additional authenticated data (AAD)
    pub fn decrypt_with_aad(
        &self,
        encrypted: &EncryptedData,
        aad: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        use chacha20poly1305::aead::Payload;

        let nonce = Nonce::from_slice(&encrypted.nonce);
        let payload = Payload {
            msg: &encrypted.ciphertext,
            aad,
        };

        self.cipher
            .decrypt(nonce, payload)
            .map_err(|_| CryptoError::DecryptionFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_from_password() {
        let key1 = EncryptionKey::from_password(b"password", b"salt");
        let key2 = EncryptionKey::from_password(b"password", b"salt");
        let key3 = EncryptionKey::from_password(b"different", b"salt");

        assert_eq!(key1.as_bytes(), key2.as_bytes());
        assert_ne!(key1.as_bytes(), key3.as_bytes());
    }

    #[test]
    fn test_key_generate() {
        let key1 = EncryptionKey::generate();
        let key2 = EncryptionKey::generate();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let key = EncryptionKey::generate();
        let encryptor = ArchiveEncryptor::new(&key);

        let plaintext = b"Test message for encryption";
        let encrypted = encryptor.encrypt(plaintext).unwrap();
        let decrypted = encryptor.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypted_data_serialization() {
        let key = EncryptionKey::generate();
        let encryptor = ArchiveEncryptor::new(&key);

        let plaintext = b"Serialize me";
        let encrypted = encryptor.encrypt(plaintext).unwrap();

        let bytes = encrypted.to_bytes();
        let restored = EncryptedData::from_bytes(&bytes).unwrap();

        let decrypted = encryptor.decrypt(&restored).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_encrypt_with_aad() {
        let key = EncryptionKey::generate();
        let encryptor = ArchiveEncryptor::new(&key);

        let plaintext = b"Secret data";
        let aad = b"archive-header-v1";

        let encrypted = encryptor.encrypt_with_aad(plaintext, aad).unwrap();

        // Correct AAD - should succeed
        let decrypted = encryptor.decrypt_with_aad(&encrypted, aad).unwrap();
        assert_eq!(plaintext.as_slice(), decrypted.as_slice());

        // Wrong AAD - should fail
        assert!(encryptor.decrypt_with_aad(&encrypted, b"wrong-aad").is_err());
    }

    #[test]
    fn test_tampered_ciphertext() {
        let key = EncryptionKey::generate();
        let encryptor = ArchiveEncryptor::new(&key);

        let plaintext = b"Don't tamper with me";
        let mut encrypted = encryptor.encrypt(plaintext).unwrap();

        // Tamper with ciphertext
        if !encrypted.ciphertext.is_empty() {
            encrypted.ciphertext[0] ^= 0xFF;
        }

        // Decryption should fail
        assert!(encryptor.decrypt(&encrypted).is_err());
    }
}
