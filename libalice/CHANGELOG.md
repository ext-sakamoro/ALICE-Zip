# Changelog

All notable changes to ALICE-Zip (libalice) will be documented in this file.

## [Unreleased]

### Changed
- **Breaking:** `format::AliceFileHeader::from_bytes` refuses what the writer never
  produced, instead of reading it as something else. It now checks the header with the
  core crate's `container::parse_legacy_alice_zip_header`, so both crates have one
  reader for these files:
  - a version other than 1.0 or 1.1 → `FormatError::UnsupportedVersion` (a major
    version of 2 was read as 1.1, a minor version of 2 or more as 1.1)
  - an unknown `payload_type` → `FormatError::InvalidPayloadType` (was read as
    `Procedural`)
  - an engine index outside 0 to 3 → `FormatError::InvalidEngine`
  - a version 1.1 header shorter than 66 bytes → `FormatError::TooShort` (was parsed
    as a version 1.0 header)

  Files written by the Python package or by this crate (versions 1.0 and 1.1 with
  defined field values) read as before. Migration: a caller that matches on
  `FormatError` exhaustively adds the two new variants; a caller that relied on
  unknown `payload_type` values falling back to `Procedural` has to treat the error,
  since such a file was not written by any released writer.

## [2.7.0] - 2026-10-09

### Added
- Re-exports for the core crate's new residual API: `compress_residual_xor` /
  `decompress_residual_xor` (bit-exact by construction — it stores the bit-pattern
  xor against the model instead of the difference), `ResidualCodec`,
  `residual_codec_default`, `residual_container_codec` and the `*_with` variants of
  all three container constructors

### Changed
- Core crate raised to `alice-zip` 0.8, whose residual containers default to deflate
  instead of LZMA. `lzma-rs`'s encoder produced 316,528 bytes where zlib produced
  8,742 on the same input (36x worse), so every container this crate writes gets
  smaller; the `level` argument is now the deflate level and actually takes effect.
  Containers written by earlier releases still decode, but containers written from
  this release need 2.7.0 or later to read

## [2.6.0] - 2026-10-08

### Changed
- Core crate raised to `alice-zip` 0.7, whose reconstruction laws now accumulate in `f64`
  and round to `f32` once on return instead of accumulating in `f32`. The signal
  generators re-exported through this module therefore return different bits in the last
  places; analytic behaviour, the container formats and this crate's own API are unchanged,
  and the reconstruction is more accurate (maximum error against the `f64` closed form
  5.95e-8 against 9.16e-7)
- The core change exists because each law was implemented twice and the two copies
  disagreed on 788 of 1024 samples for a sine. This module hit the same class of defect
  between 2.2.0 and 2.3.0, when it carried its own copies of the polynomial / Fourier /
  Perlin laws and they diverged from the core crate. Point evaluators
  (`alice_zip::generators::{sine_at, multi_sine_at, fourier_at, polynomial_at}`) are the
  single implementation of each law now, and the array generators are a `map` over them

## [2.5.0] - 2026-10-08

### Changed
- Core crate raised to `alice-zip` 0.6, whose float transcendentals now come from
  `alice-det-math` instead of the platform `libm`. The signal generators and the entropy
  estimate re-exported through `alice_core` therefore return different bits in the last
  places; analytic behaviour, the container formats and this crate's own API are unchanged
  (see the core [CHANGELOG](../CHANGELOG.md) for why bit-identical transcendentals are a
  requirement rather than a preference)
- **Breaking:** this crate's own float transcendentals follow, so results change in the
  last places for `analyzer` (sine fitting, variance and mean squared error),
  `media::audio` (sinusoid synthesis), `media::image` (gradient direction and the radial
  distance normalisation) and `residual` (the entropy estimate). The repository
  `clippy.toml` now refuses the platform methods in this crate too, so a later change
  cannot reintroduce one unnoticed; `sqrt`, `floor`, `round` and `mul_add` are unaffected
  because IEEE 754 specifies each of them exactly

## [2.4.0] - 2026-09-16

### Changed
- `compression` is now a thin re-export of `alice_zip::compression` (`lzma` feature) and
  `alice_zip::quantize`: LZMA / zlib wrappers, 8 / 16-bit quantisation and the residual
  container codec live once in the core crate (byte-identical formats); the 17 contract
  tests stay here Direct `lzma-rs` / `flate2` dependencies removed

## [2.3.0] - 2026-09-15

### Changed
- `pyproject.toml`: distribution name `alice-zip` → `libalice` (matches `import libalice`;
  neither name was ever published on PyPI), `readme` / `license` inside the crate
  directory (maturin ≥ 1 refuses paths outside it), pyo3 / numpy 0.29
- `README.md` added (crate / FFI / Python build notes)
- Cargo package renamed `alice-zip` → `alice-zip-cli` (the crates.io `alice-zip` is the core
  crate at the repository root; two packages with one name broke `cargo semver-checks`).
  Library name `alice_core`, binary `alice` and the pip package `libalice` are unchanged
- `generators` is now a thin re-export of `alice_zip::generators` (path dependency on the
  core crate, 0.4). The three local copies (`fourier.rs` / `perlin.rs` / `polynomial.rs`)
  are removed; every generator law lives once in `../src/generators/`
- `generators::fit_polynomial` / `generate_polynomial` keep the `.alice` container
  convention (descending coefficients, `x ∈ [0, 1]`) via the core `*_unit` functions;
  `analyze_signal` is the core `analyze_signal_fft` (rustfft), bins `1..=n/2`
- `generate_perlin_2d` / `generate_perlin_advanced` return `Result` (`scale <= 0`,
  non-finite `scale`, `octaves == 0` and `width * height` overflow are errors instead
  of NaN textures); Python raises `ValueError`, FFI returns `ALICE_ERROR_INVALID_PARAM`
- Polynomial fitting uses Householder QR (core), replacing normal equations
- `profile.release` no longer sets `panic = "abort"` (required for FFI panic isolation)
- Removed direct dependencies `rustfft` / `rayon` (pulled in through the core crate)

### Fixed
- `alice_polynomial_generate` (C FFI) now evaluates `y = c0 + c1·x + …` at `x = 0..n-1`
  exactly as `alice.h` / C# / C++ documented; 2.2.0 evaluated the `.alice` convention
  (descending coefficients on `[0, 1]`) under that documentation
- `alice_get_last_error` returned a pointer to a non-NUL-terminated `String`; the message
  is now a `CString`
- `alice_free_buffer` / `alice_free_float_buffer` reset the buffer after freeing (a second
  free is a no-op instead of a double free)

### Added
- **FFI panic isolation**: all 16 `extern "C"` functions run inside `catch_unwind`; a Rust
  panic is reported as `ALICE_ERROR_INTERNAL_PANIC = 7` (new enum value in `alice.h`,
  C#, C++ bindings) with the message in `alice_get_last_error()` instead of aborting the
  host process (Rust ≥ 1.81 behaviour)

## [2.2.0] - 2026-02-23

### Added
- `generators::polynomial` — Polynomial evaluation (Horner's method) and least-squares fitting
- `generators::fourier` — FFT-based Fourier signal reconstruction and analysis via `rustfft`
- `generators::perlin` — Vectorized 2D Perlin noise generation with Rayon parallelism
- `analyzer` — Model competition: polynomial, Fourier, Perlin, constant, linear fitting
- `compression` — LZMA/zlib compression, 8-bit quantization, dequantization
- `residual` — Residual storage for lossless reconstruction
- `format` — `.alz` file format (32-byte header, multi-mode)
- `media` — Media-specific compression helpers
- `ffi` — C FFI exports
- `codec_bridge` — (feature `codec`) ALICE-Codec wavelet bridge
- Python bindings (feature `python`) — PyO3 + NumPy for all generators and analyzers
- CLI binary `alice` — compress, decompress, info, benchmark commands
- 81 unit tests
