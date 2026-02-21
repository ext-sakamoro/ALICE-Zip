# ALICE-Zip Enterprise Edition

> **Commercial features for ALICE-Zip compression**

## License

**PROPRIETARY AND CONFIDENTIAL**

This software requires a valid Commercial License.
For licensing inquiries, contact the author.

## Features

| Feature | Standard | Professional | Enterprise |
|---------|:--------:|:------------:|:----------:|
| Encrypted Archives | ✅ | ✅ | ✅ |
| Parallel Compression | ✅ | ✅ | ✅ |
| Access Control | - | ✅ | ✅ |
| Audit Logging | - | ✅ | ✅ |
| Advanced Algorithms | - | ✅ | ✅ |
| Quantum Optimization | - | - | ✅ |
| API Access | - | - | ✅ |
| Priority Support | - | - | ✅ |

## Installation

```toml
[dependencies]
alice-zip-enterprise = { path = "../libalice-enterprise" }
```

## Usage

### License Activation

```rust
use alice_enterprise::license::{LicenseKey, LicenseType};

// Generate license key
let key = LicenseKey::generate(LicenseType::Professional, None);

// Parse and validate
let license = LicenseKey::parse(&key)?;
assert!(!license.is_expired());

// Activate trial (14 days)
let trial_key = LicenseKey::generate_trial(14);
```

### Encrypted Archives

```rust
use alice_enterprise::security::{ArchiveEncryptor, EncryptionKey};

// Derive key from password
let key = EncryptionKey::from_password(b"my-password", b"salt");

// Encrypt data
let encryptor = ArchiveEncryptor::new(&key);
let encrypted = encryptor.encrypt(&plaintext)?;

// Decrypt data
let decrypted = encryptor.decrypt(&encrypted)?;
```

### Access Control

```rust
use alice_enterprise::access::{AccessControl, Permission, Role};

let mut acl = AccessControl::with_owner("alice");

// Grant access
acl.grant_role("bob", Role::Editor);
acl.grant_permission("charlie", Permission::READ);

// Check permission
if acl.check("bob", Permission::WRITE) {
    // Allow write access
}
```

## Security

| Component | Algorithm | Key Size |
|-----------|-----------|----------|
| Encryption | ChaCha20-Poly1305 | 256-bit |
| Key Derivation | HKDF-SHA256 | - |
| Checksum | SHA256 | - |

## License Key Format

```
ALZ-{TYPE}-{FEATURES}-{EXPIRY}-{CHECKSUM}

TYPE: TRL | STD | PRO | ENT
FEATURES: Hex-encoded feature flags
EXPIRY: YYYYMMDD or 00000000 (perpetual)
CHECKSUM: SHA256-based 4-char checksum

Example: ALZ-PRO-1F-00000000-A3B2
```

## Building

```bash
# Default features (security, license)
cargo build

# With access control
cargo build --features access-control

# All features
cargo build --features full
```

## Testing

```bash
cargo test --features full
```

---

Copyright (c) 2026 Moroya Sakamoto. All rights reserved.
