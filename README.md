<p align="center">
  <img src="assets/logo-on-light.png" alt="ALICE-Zip" width="400">
</p>

<h1 align="center">ALICE-Zip</h1>

<p align="center">
  <a href="https://github.com/ext-sakamoro/ALICE-Zip"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
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

## Installation

```bash
pip install alice-zip
```

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
Output: Parameters only (100 bytes)
        ↓
Result: 400 KB → 100 bytes = 4000x compression
```

## Benchmarks

| Data Type | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| Sine wave (100K samples) | 400 KB | ~100 bytes | **4000x** |
| Polynomial (degree 3) | 400 KB | ~200 bytes | **2000x** |
| Linear gradient | 400 KB | ~150 bytes | **2600x** |
| Random data | 400 KB | ~380 KB | 1.05x (LZMA fallback) |

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

## Game Industry Exception

**Free for game development!** Just add this to your credits:

```
Powered by ALICE-Zip (https://github.com/ext-sakamoro/ALICE-Zip)
```

See [LICENSE](LICENSE) for details.

## Building from Source

```bash
git clone https://github.com/ext-sakamoro/ALICE-Zip
cd ALICE-Zip

# Install Python package
pip install -e .

# Run tests
pytest tests/ -v
```

## License

**Open Core License**

- **Personal / Educational**: Free under MIT License
- **Game Development**: Free with attribution only (excluding console firmware, game engines, etc.)
- **Commercial**: Core is free under MIT, Pro/Enterprise required for advanced features

See [LICENSE](LICENSE) for full details.

## Author

Created by **Moroya Sakamoto**

---

*"The best compression is not to store data, but to store the recipe for generating it."*
