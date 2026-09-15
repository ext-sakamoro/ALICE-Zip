<p align="center">
  <img src="assets/logo-on-light.png" alt="ALICE-Zip" width="400">
</p>

<h1 align="center">ALICE-Zip</h1>

<p align="center">
  <a href="https://crates.io/crates/alice-zip"><img src="https://img.shields.io/crates/v/alice-zip.svg" alt="crates.io"></a>
  <a href="https://docs.rs/alice-zip"><img src="https://docs.rs/alice-zip/badge.svg" alt="docs.rs"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/ci.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/security-audit.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/security-audit.yml/badge.svg" alt="Security"></a>
  <a href="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/fuzz.yml"><img src="https://github.com/ext-sakamoro/ALICE-Zip/actions/workflows/fuzz.yml/badge.svg" alt="Fuzz"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.9+-yellow.svg" alt="Python"></a>
</p>

> **Procedural Generation Compression Engine**
> *Store algorithms, not data.*

[日本語版 README](README_ja.md)

---

ALICE-Zip is a next-generation compression tool that stores **"how to generate the data"** instead of the data itself.

For patterns, waves, and mathematical data, it achieves compression ratios of **10x to 1000x**.
For everything else, it falls back to LZMA, ensuring it's **never worse** than standard tools.

## Features

- **Procedural Compression:** Sine waves, polynomials, and mathematical patterns
- **Adaptive Fallback:** Automatically selects LZMA when procedural methods don't help
- **Lossless:** Bit-perfect reconstruction
- **Cross-Platform:** Python, Rust, C#/Unity, C++/UE5

## Repository layout

| Path | What | Where it ships |
|------|------|----------------|
| `/` (`alice-zip`) | **Rust core crate** — compression primitives (LZ77, dictionary, BPE, entropy, quantisation, zlib / LZMA residual containers) + every generator law (polynomial / Fourier / Perlin), `no_std + alloc` | [crates.io](https://crates.io/crates/alice-zip) · [docs.rs](https://docs.rs/alice-zip) |
| `libalice/` (`alice-zip-cli`) | CLI `alice`, C FFI (`cdylib`), PyO3 native module; thin re-export of the core generators and compression | C# / UE5 bindings, pip `libalice` |
| `alice_zip/` | Python package (`ALICEZip` analyzer + `.alice` container) | pip `alice-zip` |
| `bindings/` | C++ / C# (Unity) / UE5 wrappers over `libalice/include/alice.h` | |

## Installation

```bash
# Python
pip install alice-zip

# Rust
cargo add alice-zip                       # std (default): + zlib wrappers
cargo add alice-zip --features fft,parallel,lzma   # rustfft analysis, rayon textures, LZMA residual containers
cargo add alice-zip --no-default-features     # no_std + alloc (libm float math)
```

### Rust crate

```rust
use alice_zip::prelude::*;

// Compression primitives (no_std)
let tokens = lz77_encode(b"abcabcabcabc", 256, 32);
assert_eq!(lz77_decode(&tokens)?, b"abcabcabcabc");
assert_eq!(shannon_entropy(&(0..=255u8).collect::<Vec<_>>()), 8.0);

// Generator laws: fit a series, keep the coefficients, regenerate
let series: Vec<f32> = (0..64).map(|i| 3.0 + 2.0 * i as f32).collect();
let (coeffs, degree, err) = fit_polynomial(&series, 4, 1e-6).unwrap();
assert_eq!((degree, generate_polynomial(64, &coeffs) == series), (1, true));

let (bins, dc) = analyze_signal(&generate_sine_wave(32, 1.0, 2.0, 0.0, 0.5), 4, 0.99);
let regenerated = generate_from_coefficients(32, &bins, dc);
# Ok::<(), alice_zip::ZipError>(())
```

| Feature | Adds | Default |
|---------|------|---------|
| `std` | `compression` (zlib via flate2), `std::error::Error` for `ZipError` | ✓ |
| `fft` | `generators::analyze_signal_fft` (rustfft, same contract as the naive DFT) | |
| `parallel` | rayon row parallelism for `generate_perlin_2d` / `_advanced` | |
| `lzma` | `compression::{lzma_compress, lzma_decompress}` + the `.alice` quantised / lossless residual containers (lzma-rs) | |
| *(none)* | `no_std + alloc`; float math via `libm`; CI builds the rlib for `thumbv7em-none-eabihf` | |

Two persisted coefficient conventions coexist under explicit names (they are
different laws and are never silently interchangeable): `fit_polynomial` /
`generate_polynomial` (`x = 0..n-1`, ascending — `alice-db` segments) and
`fit_polynomial_unit` / `generate_polynomial_unit` (`x ∈ [0, 1]`, descending —
the `.alice` container). Every law is checked against a closed-form answer in
[`tests/analytic_oracle.rs`](tests/analytic_oracle.rs) and fuzzed (7 targets,
including naive-DFT ↔ FFT parity). MSRV 1.87.

## Quick Start

### Command Line

```bash
# Compress
alice-zip compress data.bin -o data.alice

# Decompress
alice-zip decompress data.alice -o restored.bin

# Show file info
alice-zip info data.alice
```

### Python API

```python
from alice_zip import ALICEZip
import numpy as np

zipper = ALICEZip()

# Compress sine wave data
data = np.sin(np.linspace(0, 100*np.pi, 100000)).astype(np.float32)
compressed = zipper.compress(data)

print(f"Original: {data.nbytes:,} bytes")
print(f"Compressed: {len(compressed):,} bytes")
print(f"Ratio: {data.nbytes / len(compressed):.1f}x")

# Decompress
restored = zipper.decompress(compressed)
```

## How It Works

Traditional compression finds patterns in **bytes**. ALICE finds patterns in **mathematics**.

```
Original Data = Generated(parameters) + Residual

Where:
  - Generated()  = Mathematical function (polynomial, sine wave, etc.)
  - parameters   = Tiny description (~100 bytes)
  - Residual     = Compressed difference (often near-zero)
```

### Example

```
Input:  Sine wave, 100,000 samples (400 KB)
        ↓
Analysis: Detected as "Sine wave, freq=50Hz, amp=1.0"
        ↓
Output: Parameters only (~280 bytes)
        ↓
Result: 400 KB → 280 bytes = 1400x compression
```

## Benchmarks

| Data Type | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| Sine wave (100K samples) | 400 KB | ~280 bytes | **1400x** |
| Polynomial (degree 3) | 400 KB | ~285 bytes | **1400x** |
| Linear gradient | 400 KB | ~250 bytes | **1600x** |
| Random data | 400 KB | ~370 KB | 1.08x (LZMA fallback) |

## Ideal Use Cases

- **Scientific Data:** Simulation outputs, sensor readings, waveforms
- **Game Assets:** Procedural textures, terrain heightmaps
- **IoT/Edge:** Sensor logs on bandwidth-constrained devices
- **Time Series:** Telemetry, monitoring logs with patterns

## When NOT to Use

| Data Type | Reason |
|-----------|--------|
| JPEG/PNG/MP3 | Already compressed |
| Random/encrypted data | No pattern to exploit |
| Small files (<1KB) | Header overhead |

## Building from Source

```bash
git clone https://github.com/ext-sakamoro/ALICE-Zip
cd ALICE-Zip

# Install Python package
pip install -e .

# Run tests
pytest tests/ -v

# Rust core crate — every CI gate, locally (scripts/preflight.sh --quick ≈ 3 min)
scripts/preflight.sh

# CLI / FFI crate
cargo build --manifest-path libalice/Cargo.toml --release
```

## Related Projects

| Project | Description |
|---------|-------------|
| [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) | Model-based time-series database |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | Embedded/IoT model generator (no_std) |
| [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) | Ultra-low bandwidth video streaming |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | Complete Edge-to-Cloud pipeline demo |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## License

- Rust core crate `alice-zip` (`/`): [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your option
- CLI / FFI crate (`libalice/`), Python package (`alice_zip/`), bindings: [MIT](LICENSE-MIT)
- `libalice-enterprise/`: separate, unpublished proprietary crate (see its own [LICENSE](libalice-enterprise/LICENSE))

## Author

Created by **Moroya Sakamoto**

---

*"The best compression is not to store data, but to store the recipe for generating it."*
