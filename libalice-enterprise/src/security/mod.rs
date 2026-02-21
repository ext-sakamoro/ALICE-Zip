//! Security Module
//!
//! Encryption and authentication for ALICE-Zip archives.
//!
//! # Algorithms
//!
//! - **Encryption**: ChaCha20-Poly1305 AEAD
//! - **Key Derivation**: HKDF-SHA256
//! - **Authentication**: HMAC-SHA256
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_enterprise::security::{ArchiveEncryptor, EncryptionKey};
//!
//! // Derive key from password
//! let key = EncryptionKey::from_password(b"my-password", b"salt");
//!
//! // Encrypt data
//! let encryptor = ArchiveEncryptor::new(&key);
//! let encrypted = encryptor.encrypt(&plaintext)?;
//!
//! // Decrypt data
//! let decrypted = encryptor.decrypt(&encrypted)?;
//! ```

mod crypto;
mod kdf;

pub use crypto::{
    ArchiveEncryptor,
    EncryptedData,
    EncryptionKey,
    CryptoError,
};

pub use kdf::{
    KeyDerivation,
    DerivedKey,
    KdfError,
};

/// Security module version
pub const SECURITY_VERSION: u8 = 1;

/// Encrypted archive header magic bytes
pub const ENCRYPTED_MAGIC: [u8; 4] = [0xAE, 0x5A, 0x45, 0x01]; // AEZ + version

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::from_password(b"test-password", b"test-salt");
        let encryptor = ArchiveEncryptor::new(&key);

        let plaintext = b"Hello, ALICE-Zip Enterprise!";
        let encrypted = encryptor.encrypt(plaintext).unwrap();
        let decrypted = encryptor.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = EncryptionKey::from_password(b"password1", b"salt");
        let key2 = EncryptionKey::from_password(b"password2", b"salt");

        let encryptor1 = ArchiveEncryptor::new(&key1);
        let encryptor2 = ArchiveEncryptor::new(&key2);

        let plaintext = b"Secret data";
        let encrypted = encryptor1.encrypt(plaintext).unwrap();

        // Decryption with wrong key should fail
        assert!(encryptor2.decrypt(&encrypted).is_err());
    }
}
