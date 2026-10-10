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

For patterns, waves and mathematical data it reaches **75x to 806x with the
original samples recovered bit for bit**, and 160x to 1286x if a 1e-12 error is
acceptable. When no law fits, it falls back to byte compression and lands on the
same size as plain zlib rather than worse. Every figure on this page comes from
`cargo run --release --example compression_ratio --features lzma`.
<!-- claim-test: the_lossless_container_is_not_worse_than_this_crates_own_zlib_wrapper -->

## Features

- **Procedural Compression:** Sine waves, polynomials, and mathematical patterns
- **Adaptive Fallback:** compresses the bytes directly when no law fits, and is
  measured against plain zlib so "no worse than a standard tool" is a check, not
  a promise
- **Bit-exact, by construction:** the residual is stored as the bit-pattern xor
  against the model, so every finite sample — including signed zeros, denormals
  and values far smaller than their model — comes back unchanged. The older
  subtraction residual (`original - model` in `f32`) is *not* exact: on the sine
  below it loses the 99 samples nearest the zero crossings
  <!-- claim-test: the_xor_container_is_bit_exact_on_every_law_family_the_readme_quotes -->
- **Cross-Platform:** Python, Rust, C#/Unity, C++/UE5

## Repository layout

| Path | What | Where it ships |
|------|------|----------------|
| `/` (`alice-zip`) | **Rust core crate** — compression primitives (LZ77, dictionary, BPE, entropy, quantisation, zlib / LZMA residual containers) + every generator law (polynomial / Fourier / Perlin), `no_std + alloc` | [crates.io](https://crates.io/crates/alice-zip) · [docs.rs](https://docs.rs/alice-zip) |
| `libalice/` (`alice-zip-cli`) | CLI `alice`, C FFI (`cdylib`), PyO3 native module; thin re-export of the core generators and compression | C# / UE5 bindings, Python module built with maturin from `libalice/` |
| `alice_zip/` | Python package (`ALICEZip` analyzer + `.alice` container) | from this repository (not on PyPI) |
| `bindings/` | C++ / C# (Unity) / UE5 wrappers over `libalice/include/alice.h` | |

## Installation

```bash
# Python (not on PyPI: install from the repository)
pip install git+https://github.com/ext-sakamoro/ALICE-Zip

# Rust
cargo add alice-zip                       # std (default): + zlib wrappers
cargo add alice-zip --features fft,parallel,lzma   # rustfft analysis, rayon textures, LZMA residual containers
cargo add alice-zip --no-default-features     # no_std + alloc (sqrt / floor / round via libm)
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
| *(none)* | `no_std + alloc`; `sqrt` / `floor` / `round` via `libm` (the transcendentals go through `alice-det-math` in every build); CI builds the rlib for `thumbv7em-none-eabihf` | |

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

### Lossless mode (Python)

```python
from alice_zip import ProceduralCompressionDesigner

designer = ProceduralCompressionDesigner()
result = designer.compress(data, enable_lossless=True)
restored = designer.decompress(result)   # same dtype, same bytes as data
```

- `decompress` returns the input bit for bit, dtype included, for the eleven
  real dtypes (`float16/32/64`, `int8`–`int64`, `uint8`–`uint64`), NaN payloads
  and infinities included. <!-- claim-test: test_lossless_gives_back_the_input_bit_for_bit -->
- The residual is written from the original and kept in its precision
  (`float64` for float64 / int32 / uint32 / int64 / uint64 originals, `float32`
  otherwise). A value that the generated value plus the residual cannot
  rebuild exactly (NaN, an infinity, or a difference that is not exact) is
  stored as it is. The residual is a `ResidualData` file, version 4 (version 2
  when the residual is `float32` and nothing is stored as it is).
- When the parameters plus the residual are not smaller than LZMA of the
  input, the LZMA result is returned instead; so is a dtype the residual
  cannot carry (complex, …). <!-- claim-test: test_a_lossless_result_is_not_larger_than_lzma_of_the_input -->
- `quantize_residual=8` / `16` is lossy (`is_lossless` is `False`): each value
  is within half a quantisation step, plus the rounding of the residual stored
  as `float32` and of the output to its dtype (half a unit in the last place of
  each). <!-- claim-test: test_a_quantized_residual_is_off_by_at_most_half_a_step -->
- A result made by an earlier version (a `float32` residual, which was not
  lossless) is refused by `decompress`; compress the input again, or pass
  `allow_approximate=True` to read its approximate values. <!-- claim-test: test_a_result_of_the_earlier_residual_path_is_refused -->

## How It Works

Traditional compression finds patterns in **bytes**. ALICE finds patterns in **mathematics**.

```
Original Data = Generated(parameters) ^ Residual

Where:
  - Generated()  = Mathematical function (polynomial, sine wave, etc.)
  - parameters   = Tiny description (tens of bytes)
  - Residual     = Compressed bit-pattern xor against Generated()
                   (mostly zero bytes when the law is a good fit)
```

The xor is what makes the recovery exact. A difference (`original - model`)
computed in `f32` rounds away the original wherever it is much smaller than the
model, so `model + residual` does not return it; `bits ^ bits` has no such
case.

### Example

```
Input:  Sine wave, 100,000 samples (400,000 bytes of f32)
        ↓
Analysis: one Fourier coefficient — "freq bin 50, amp 1.0, phase 0"
        ↓
Output: 20 bytes of parameters + 5,306 bytes of compressed xor residual
        ↓
Result: 400,000 -> 5,326 bytes = 75x, and the samples come back bit for bit
        (plain zlib on the same bytes: 8,742 -> 45.8x)
```

Keeping only the 20 bytes of parameters is 20,000x, but that is an
**approximation** with a max absolute error of 1.8e-7 — not a reconstruction.
Both numbers are in the benchmark table below, in separate columns, because
quoting the first one next to the word "lossless" overstates the result by more
than two orders of magnitude.

### Laws with evidence (`law` module)

`law::SignalLaw` keeps a fitted polynomial together with the evidence it came
from, the measured residual, the `x` range the evidence covers, its provenance
and reference values it must reproduce. It is recomputed at new conditions but
never extrapolated, and new evidence is judged rather than appended:

```rust
use alice_zip::law::{IngestPolicy, Provenance, SignalLaw, Verdict};

let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 1.0 + 2.0 * i as f64)).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))?;
assert!((law.evaluate(2.5)? - 6.0).abs() < 1e-12); // a condition that was not measured
assert!(law.evaluate(9.0).is_err());               // outside the measured range

let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
assert!(matches!(law.ingest(&[(0.5, 2.0), (3.5, 8.0)], &policy), Verdict::Supports { .. }));
# Ok::<(), alice_zip::law::LawError>(())
```

| Verdict | When |
|---------|------|
| `Supports` | the new points agree with the law within the band |
| `ParameterUpdate` | the same form fits old and new points with other parameters (the refitted law is returned) |
| `ResidualGrew` | the deviation exceeds the band but not `break_factor` times it |
| `Breaks` | the new points are not described by this form |
| `OutOfRange` | some points lie outside the range the law was fitted on; nothing is judged |

The rules and their order are documented on the module and pinned by
[`tests/analytic_law.rs`](tests/analytic_law.rs).

#### Identifying a law by its content

`SignalLaw::law_id` returns a 32-byte identifier for a law, so a stored result
can name the law it came from:

```rust
use alice_zip::law::{Provenance, SignalLaw};

let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 1.0 + 2.0 * i as f64)).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))?;

// 32 bytes identifying the arithmetic used to evaluate the law. For a law
// evaluated through this crate that is `law::SEMANTICS_ID`, re-exported from
// `alice-det-math`; pass another value only if another arithmetic is used.
let id = law.law_id(&alice_zip::law::SEMANTICS_ID);
# Ok::<(), alice_zip::law::LawError>(())
```

The digest covers exactly what `evaluate` reads — the valid range and the
coefficients — plus `semantics_id`. The evidence, the residual, the provenance
and the oracle cases are deliberately left out: they record how the law was
obtained and justified, not what it computes, so the same law fitted from two
different measurement runs gets one identifier.

**Guaranteed:** equal identifier implies `evaluate` returns the same bits for
every `x`, on any target where the IEEE 754 basic operations hold.
<!-- claim-test: equal_law_id_implies_bit_identical_evaluation -->

That guarantee is why the crate takes its float transcendentals from
`alice-det-math` rather than from the platform: IEEE 754 does not require `sin`,
`cos`, `atan2` or `log2` to be correctly rounded, so the platform version may
differ between operating system, CPU and compiler. Measured on one machine by
changing nothing but the `std` feature, the 0.5 line produced different bits in
1 of 64 samples of `generate_multi_sine` and in 6 of 9 fields of
`analyze_signal`. `tests/determinism_golden.rs` records the bit patterns and CI
runs it on three operating systems and in the `no_std` build, and
`clippy.toml` refuses the platform methods so they cannot come back; `sqrt`,
`floor` and `round` stay on the platform because IEEE 754 specifies them
exactly.
<!-- claim-test: sinusoid_generators_are_the_recorded_bits -->

**Guaranteed:** every reconstruction law has one implementation. `sine_at`,
`multi_sine_at`, `fourier_at` and `polynomial_at` evaluate a law at one
position, which may be fractional, and the array generators are a `map` over
them — so reading a range and reading a single sample cannot disagree. A
consumer that answers point queries calls these rather than keeping its own
copy of the law. Before 0.7.0 a second copy existed outside this crate and the
two differed on 788 of 1024 samples for a sine; the law now accumulates in
`f64` and rounds once, which is the more accurate of the two forms (maximum
error against the `f64` closed form 5.95e-8 against 9.16e-7).
<!-- claim-test: array_generators_are_a_map_over_the_point_law -->

**Not guaranteed:** the converse. Two laws that evaluate identically can still
get different identifiers — appending a zero coefficient is the simplest case.
Reducing a law to a normal form first is a separate problem, so the identifier
is not a deduplication key.
<!-- claim-test: evaluation_equivalent_laws_may_still_differ_in_id -->

The byte layout is documented on `law_id` and pinned by a golden digest that CI
reproduces on every supported target, so an independent implementation can
compute the same identifier.
<!-- claim-test: law_id_golden -->

Floats enter the digest as raw bits, and `-0.0` is **not** folded into `+0.0`:
with a negative linear coefficient the sign of a zero constant term is visible
in the result at the lower end of the range, so folding the two would hand one
identifier to laws that compute different bits.
<!-- claim-test: negative_zero_is_observable_in_evaluation_so_it_changes_the_id -->

### Container with content hashes (`container` module)

`container::Container` holds several payloads in one file and identifies each
by its SHA-256. A section carries the bytes of an existing format unchanged,
so the inner format keeps its own version; the container fixes only the
header, the section table and the integrity of the file.

- **Identifier**: `Container::id` is SHA-256 over the header and the section
  table (which holds every payload's SHA-256), with domain separation and
  length prefixes in the same encoding as `law::SignalLaw::law_id`.
- **Integrity**: the last 32 bytes are the SHA-256 of everything before them.
  Readers always check it. A payload's own SHA-256 is checked when the payload
  is read, so `ContainerView` can return one section without hashing the
  others again.
- **Versions**: a major version other than 1 is refused; any minor version is
  read. A section with an unknown tag is refused if it is marked critical and
  kept unchanged otherwise, so writing back what was read gives the same bytes.
- **References**: an `SREF` section names other sections by SHA-256, so a
  payload used several times is stored once and a reference to a missing
  section is refused. An `LIDS` section lists law identifiers, which are
  compared with the semantics id in the header and, through
  `ContainerView::verify_law_ids`, with identifiers recomputed from the
  payloads.
- **Earlier files**: `container::read_any` also reads an `ALICE_ZIP` file of
  version 1.0 or 1.1 (header and payload as two sections) and refuses values
  that were never written.

<!-- claim-test: every_single_bit_flip_is_refused_by_the_expected_check -->
Changing any single bit of a container is refused (checked for every bit of a
four-section fixture), and the bytes and identifiers equal those of an
independent reference writer on Linux, macOS, Windows and `wasm32-wasip1`
(`tests/container_oracle.rs`, `tests/data/container/container_ref.py`).

```rust
use alice_zip::container::{read_any, Container, Tag};

let mut c = Container::new([0; 32]);
c.push(Tag::RAW, true, b"payload".to_vec());
let bytes = c.to_bytes();
assert_eq!(read_any(&bytes)?, c);
# Ok::<(), alice_zip::container::ContainerError>(())
```

## Benchmarks

100,000 `f32` samples (400,000 bytes), deflate level 9, reproduced by
`cargo run --release --example compression_ratio --features lzma`. The
`zlib alone` column is the same bytes through this crate's own zlib wrapper at
the same level — the floor the fallback has to match.

**Bit-exact** (law parameters + xor residual, samples recovered unchanged):

| Data type | Compressed | Ratio | Bit-exact | zlib alone |
|-----------|-----------|-------|-----------|------------|
| Sine wave | 5,326 B | **75x** | yes | 8,742 B (45.8x) |
| Polynomial (degree 3) | 3,916 B | **102x** | yes | 330,123 B (1.2x) |
| Linear gradient | 496 B | **806x** | yes | 293,616 B (1.4x) |
| Random data | 364,645 B | 1x | yes | 364,641 B (1.1x) |

<!-- claim-test: the_xor_container_is_bit_exact_on_every_law_family_the_readme_quotes -->

**Lossy**, for callers who can accept a bounded error:

| Data type | Law parameters only | Ratio | Max abs error | Quantised residual (16-bit) | Ratio | Max abs error |
|-----------|--------------------|-------|---------------|----------------------------|-------|---------------|
| Sine wave | 20 B | 20,000x | 1.8e-7 | 2,075 B | 193x | 1.4e-12 |
| Polynomial (degree 3) | 36 B | 11,111x | 6.0e-8 | 2,500 B | 160x | 9.1e-13 |
| Linear gradient | 20 B | 20,000x | 7.5e-9 | 311 B | 1,286x | 5.0e-14 |
| Random data | 4 B | 100,000x | 1.0e0 | 200,062 B | 2x | 1.5e-5 |

Reading the table:

- **Random data is the fallback working.** No law fits, so the whole signal goes
  through byte compression and the result matches plain zlib to four bytes. That
  is the point of the fallback — not to win, but not to lose.
- **The parameters-only column is not a reconstruction.** 20,000x is the size of
  the law; the error column is what you give up for it.
- **The xor container buys exactness, not size.** The subtraction residual the
  earlier releases used is still available (`compress_residual_lossless`) and is
  smaller on the polynomial (2,783 B against 3,916 B) and on short signals (at
  4,096 samples the sine goes the other way too), but it is not bit-exact.
  `compress_residual_xor` is the one to use when the samples have to come back
  unchanged; pick the other one when a bounded error is acceptable and size is
  what matters.

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

The same idea applied to other domains — this crate stores the recipe for a
*signal*; these store the recipe for a shape, a physical state, or a frame.

| Project | Description | Links |
|---------|-------------|-------|
| **ALICE-SDF** | 3D geometry as laws instead of polygons — GLSL / WGSL / HLSL transpile, SVO, marching cubes, cross-platform bit-exact evaluation | [crates.io](https://crates.io/crates/alice-sdf) · [docs.rs](https://docs.rs/alice-sdf) · [GitHub](https://github.com/ext-sakamoro/ALICE-SDF) |
| **ALICE-DetMath** | The bit-exact transcendental layer (`sin` / `exp` / `atan2` / …) that this crate's Fourier / Perlin generators and ALICE-SDF both evaluate, so the same law yields the same bits on every platform | [crates.io](https://crates.io/crates/alice-det-math) · [docs.rs](https://docs.rs/alice-det-math) · [GitHub](https://github.com/ext-sakamoro/ALICE-DetMath) |
| ALICE-DB | Model-based time-series database | [crates.io](https://crates.io/crates/alice-db) · [GitHub](https://github.com/ext-sakamoro/ALICE-DB) |
| ALICE-Edge | Embedded / IoT model generator (`no_std`) | [crates.io](https://crates.io/crates/alice-edge) · [GitHub](https://github.com/ext-sakamoro/ALICE-Edge) |
| ALICE-Streaming-Protocol | Ultra-low bandwidth video streaming | [crates.io](https://crates.io/crates/libasp) · [GitHub](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) |
| ALICE-Eco-System | Complete Edge-to-Cloud pipeline demo | [GitHub](https://github.com/ext-sakamoro/ALICE-Eco-System) |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## License

- Rust core crate `alice-zip` (`/`): [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your option
- CLI / FFI crate (`libalice/`), Python package (`alice_zip/`), bindings: [MIT](LICENSE-MIT)
- `libalice-enterprise/`: separate, unpublished proprietary crate (see its own [LICENSE](libalice-enterprise/LICENSE))

## Author

Created by **Moroya Sakamoto**

---

*"The best compression is not to store data, but to store the recipe for generating it."*
