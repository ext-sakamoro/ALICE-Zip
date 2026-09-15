# Changelog

All notable changes to the `alice-zip` Rust crate (repository root, crates.io)
are documented here. The CLI / FFI / Python native crate under `libalice/` has
its own [CHANGELOG](libalice/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-09-16

### Added
- `quantize` module (`no_std`): `quantize_8bit` / `quantize_16bit` and their inverses —
  the min / max quantisation law of the `.alice` residual payloads and `alice-edge`
  coefficient batches, moved here from `libalice` (single home, like `generators`)
- `lzma` feature (implies `std`, `lzma-rs`): `compression::{lzma_compress,
  lzma_decompress}` and the `.alice` residual containers
  `compress_residual_quantized` / `decompress_residual_quantized` /
  `compress_residual_lossless` / `decompress_residual_lossless` (byte-identical layout
  to `libalice` ≤ 2.3, documented on the module); `decompress_*` reject length overflow
- `tests/analytic_oracle.rs`: quantisation error ≤ half a step with exact endpoints
  (8 / 16 bit, n = 1..1000), residual container layout + round trip
- `tests/analytic_oracle.rs`: 6 laws that mutation testing (cargo-mutants, 8 shards,
  93% score) found unmeasured — sine DC term, energy-threshold edge values / inclusive
  cutoff, FFT empty-input guards, fit error = normalised MSE of the returned fit, 1D
  value-noise lattice / midpoint / octave-composition laws, and the alice-db persisted
  `generate_fbm_1d` values (bit-identical to 0.3.1)

- CI `libalice-python` job: `cargo check` / clippy of the `python` feature, `maturin
  develop`, and the Python test suite twice (native accelerator enabled / pure-Python
  fallback)
- `libalice/README.md` (the maturin / cargo `readme` used to point outside the crate,
  which current maturin rejects)

### Changed
- CI `quality-deep.yml`: mutants run as 8 shards over `--lib --test analytic_oracle`,
  `ulimit -v 6 GiB` so an infinite-loop mutant aborts instead of OOM-killing the runner
- docs.rs / CI feature set is now `std,fft,parallel,lzma`
- `libalice/pyproject.toml`: distribution name `alice-zip` → `libalice` (the import
  name; the root `pyproject.toml` is the `alice-zip` Python package), SPDX license
  string, author e-mail placeholder replaced in both pyproject files

### Fixed
- `alice_zip.native_accelerator`: `fourier_generate` / `multi_sine` /
  `polynomial_generate` raised `TypeError` on the native path when coefficients came
  as lists (as decoded from a `.alice` container) because PyO3 extracts tuples only;
  they are now normalised before the call (pure-Python fallback was unaffected)
- `alice_zip.native_accelerator.is_available()` reported `True` without the extension:
  the repository's `libalice/` directory imports as an empty namespace package and a
  legacy fallback imported the pure-Python `alice_zip` package itself as "native";
  availability now requires the extension entry points to exist

## [0.4.0] - 2026-09-15

### Added
- Real `no_std + alloc` support: `#![cfg_attr(not(feature = "std"), no_std)]`,
  float math through `libm` under `no_std`, CI builds the rlib for
  `thumbv7em-none-eabihf`. Until 0.3 the `no_std` claim was documentation only
  (`flate2` was unconditional and no attribute existed)
- Cargo features: `std` (default; zlib wrappers + `std::error::Error`),
  `fft` (`generators::analyze_signal_fft`, rustfft, implies `std`),
  `parallel` (rayon rows for the 2D Perlin textures, implies `std`)
- `generators::PerlinNoise` / `generate_perlin_2d` / `generate_perlin_advanced`
  (2D gradient Perlin, seeded permutation table, moved here from `libalice` so
  every generator law has exactly one home)
- `generators::fit_polynomial_unit` / `generate_polynomial_unit` — the `.alice`
  container convention (`x ∈ [0, 1]`, descending coefficients) beside the
  `alice-db` convention (`x = 0..n-1`, ascending)
- `generators::generate_fbm_1d` — the 1D value-noise law formerly reachable as
  `generate_perlin_advanced(n, _dimension, …)`
- `lz77::MAX_WINDOW` (65535); window / lookahead above it are clamped instead of
  truncated to `u16`
- `ZipError::InvalidParameter`; `ZipError` is `Copy`, `Hash`,
  `#[non_exhaustive]` and implements `std::error::Error` under `std`
- `tests/analytic_oracle.rs` — 22 closed-form / reference-implementation tests
  (entropy of `k` equiprobable symbols = `log2 k`, LZ77 round trip swept over
  window / lookahead, exact polynomial recovery, single-bin Fourier
  reconstruction including Nyquist, naive-DFT vs FFT parity, Perlin lattice
  zeros + `libalice` 2.2 reference, zlib level sweep)
- CI: 9-job `ci.yml` (3-OS tests, clippy `-D warnings`, bare-metal `no_std`,
  MSRV 1.87 real compile, feature powerset, rustdoc `-D warnings`, `libalice`),
  `security-audit.yml` (audit × 3 lockfiles, cargo-deny, machete, coverage,
  semver-checks, stub / FFI-guard grep), `fuzz.yml` (7 targets incl. Fourier
  parity fuzz), `quality-deep.yml` (cargo-mutants), `scripts/preflight.sh`

### Changed
- **Breaking**: `lz77_decode` returns `Result<Vec<u8>, ZipError>`
  (`InvalidData` for tokens that reference bytes before the output start)
- **Breaking**: `Dictionary::add` returns `Result<u32, ZipError>`
  (`DictionaryFull` for capacity `0` or `u32` arena overflow)
- **Breaking**: `generate_perlin_advanced(n, _dimension, seed, …)` (1D, unused
  `_dimension`) is now `generate_fbm_1d(n, seed, …) -> Result`; the name
  `generate_perlin_advanced(width, height, …) -> Result` is the 2D texture law
- `fit_polynomial` solves the least squares by Householder QR on `x/(n-1)`
  (normal equations lost 4-5 digits at degree 5); returned coefficients are
  unchanged in meaning
- `analyze_signal` removes the DC offset before the transform, drops numerical
  zeros (`< 1e-10`), and normalises the energy threshold by the **total** non-DC
  energy as its documentation always said (it used the truncated top-k energy)
- Naive DFT accumulates in `f64` (the naive path is the precision reference for
  the FFT path)
- `[package] resolver = "3"`; `rust-toolchain.toml` 1.98.1 + `thumbv7em` target;
  crates.io `exclude` covers the Python / bindings / fuzz directories

### Fixed
- `shannon_entropy`: a 20-term `log2` series saturated at ≈ 6.74 bit for a
  uniform byte distribution (correct: 8.0); `theoretical_min_size` inherited
  the error. `log2` is now exact (`f64::log2` / `libm::log2`)
- `lz77_encode` emitted a phantom trailing `0` literal when the last match
  reached the end of the input, so decode output was one byte longer than the
  input (unit tests sliced the extra byte away)
- `generate_from_coefficients` weighted the Nyquist bin (`k = n/2`) by 2 like a
  mirrored bin; it is self-conjugate and is now weighted by 1
- `lz77_decode` / `Dictionary::add` panicked on malformed input

## [0.3.1] - 2026-09-15
### Changed
- `rust-version = "1.87"` declared (const `Vec::len` in `Dictionary`), `homepage`
  / `documentation` metadata, rustfmt

## [0.3.0] - 2026-09-13
### Changed
- **Breaking**: `generate_perlin_advanced` takes `u64` seed / `u32` octaves
  (`alice-db` type compatibility)

## [0.2.1] - 2026-09-13
### Added
- `generators::generate_multi_sine`, `generators::generate_perlin_advanced`

## [0.2.0] - 2026-09-13
### Added
- `compression` (zlib wrappers over `flate2`) and `generators` (polynomial /
  Fourier / sinusoid) modules for `alice-db`
### Changed
- License: ALICE-Zip Open Core License → `MIT OR Apache-2.0`

## [0.1.0] - 2026-07-04
### Added
- Initial crates.io release: `lz77`, `dictionary`, `entropy`, `bpe`, `error`,
  `prelude` (split from a single `lib.rs`, 102 tests)

[Unreleased]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.3.0...v0.3.1
[0.3.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.1...alice-zip-v0.3.0
[0.2.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.0...alice-zip-v0.2.1
[0.2.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.1.0...alice-zip-v0.2.0
[0.1.0]: https://github.com/ext-sakamoro/ALICE-Zip/releases/tag/alice-zip-v0.1.0
