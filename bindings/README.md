# ALICE-Zip Language Bindings

This directory contains language bindings for ALICE-Zip, enabling use in various game engines and programming languages.

## Game Industry Exception

**Free with attribution for game development!** See [LICENSE](../LICENSE) for details.

Add this to your credits:
```
Powered by ALICE-Zip (https://github.com/ext-sakamoro/ALICE-Zip)
```

## Available Bindings

### C# / Unity

Location: `csharp/`

**Features:**
- P/Invoke bindings for all ALICE-Zip functions
- Unity Package Manager compatible
- Texture generation helpers for Unity
- Support for iOS, Android, Windows, Mac, Linux

**Quick Start (Unity):**
1. Copy `csharp/AliceZip.Unity/` to your `Packages/` folder
2. Build the native library for your platform
3. Place the library in `Plugins/` folder

```csharp
using AliceZip;
using AliceZip.Unity;

// Generate Perlin noise texture
Texture2D noise = AliceTextureGenerator.GeneratePerlinTexture(256, 256, seed: 42);

// Compress save data
byte[] compressed = Alice.LzmaCompress(saveData);
byte[] decompressed = Alice.LzmaDecompress(compressed);
```

### C++ / Unreal Engine 5

Location: `cpp/`

**Features:**
- Header-only C++ wrapper (`alice.hpp`)
- RAII memory management
- STL containers (std::vector)
- Exception-based error handling
- UE5 Blueprint support

**Quick Start (C++):**
```cpp
#include "alice.hpp"

// Generate Perlin noise
auto noise = alice::perlin_2d(256, 256, 42, 10.0f, 4);

// Compress data
auto compressed = alice::lzma_compress(data);
auto decompressed = alice::lzma_decompress(compressed);
```

**Quick Start (UE5):**
1. Copy `cpp/AliceZip.UE5/` to your `Plugins/` folder
2. Build the native library for your platform
3. Place library in `ThirdParty/lib/{Platform}/`

```cpp
// C++ in UE5
TArray<float> NoiseData;
UAliceZipLibrary::GeneratePerlin2D(256, 256, 42, 10.0f, 4, NoiseData);

// Blueprint
// Use "Generate Perlin 2D" node from ALICE-Zip category
```

## Building the Native Library

### Prerequisites
- Rust toolchain (rustup)
- Platform-specific C compiler

### Build Commands

```bash
# Navigate to libalice
cd libalice

# Build release library
cargo build --release

# Output location:
# - Windows: target/release/alice_core.dll
# - macOS: target/release/libalice_core.dylib
# - Linux: target/release/libalice_core.so
```

### Cross-compilation

```bash
# Windows (from macOS/Linux)
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu

# Linux (from macOS)
rustup target add x86_64-unknown-linux-gnu
cargo build --release --target x86_64-unknown-linux-gnu

# iOS
rustup target add aarch64-apple-ios
cargo build --release --target aarch64-apple-ios

# Android
rustup target add aarch64-linux-android
cargo build --release --target aarch64-linux-android
```

## API Reference

### Procedural Generation

| Function | Description |
|----------|-------------|
| `perlin_2d` | Generate 2D Perlin noise |
| `perlin_advanced` | Perlin noise with persistence/lacunarity |
| `sine_wave` | Generate sine wave |
| `fourier_generate` | Generate from Fourier coefficients |
| `polynomial_generate` | Generate polynomial curve |

### Compression

| Function | Description |
|----------|-------------|
| `lzma_compress` | LZMA compression (best ratio) |
| `lzma_decompress` | LZMA decompression |
| `zlib_compress` | zlib compression (faster) |
| `zlib_decompress` | zlib decompression |
| `residual_compress` | Lossy float compression |
| `residual_decompress` | Lossy float decompression |

## Platform Support

| Platform | C# | C++ | Status |
|----------|-----|------|--------|
| Windows x64 | ✅ | ✅ | Supported |
| macOS x64 | ✅ | ✅ | Supported |
| macOS ARM64 | ✅ | ✅ | Supported |
| Linux x64 | ✅ | ✅ | Supported |
| iOS | ✅ | ✅ | Supported |
| Android | ✅ | ✅ | Supported |
| WebGL | ⚠️ | ⚠️ | Planned |

## License

- **Core Library**: MIT License
- **Game Development**: Free with attribution (Game Industry Exception)
- **Other Commercial Use**: Contact for licensing

---

**Powered by ALICE-Zip** | [GitHub](https://github.com/ext-sakamoro/ALICE-Zip)
