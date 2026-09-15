# Changelog

All notable changes to ALICE-Zip (libalice) will be documented in this file.

## [2.3.0] - 2026-09-15

### Changed
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
