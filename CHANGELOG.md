# Changelog

All notable changes to the `alice-zip` Rust crate (repository root, crates.io)
are documented here. The CLI / FFI / Python native crate under `libalice/` has
its own [CHANGELOG](libalice/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.0] - 2026-10-08

### Changed

- **Breaking:** the float transcendentals now come from `alice-det-math`
  instead of the platform `libm`, so `generators::{analyze_signal,
  generate_from_coefficients, generate_sine_wave, generate_multi_sine}` and
  `entropy::{shannon_entropy, theoretical_min_size}` return different bits in
  the last places. Analytic behaviour is unchanged (the analytic oracles pass
  untouched) and `law::SignalLaw::law_id` is unaffected, since evaluating a
  polynomial law uses only IEEE 754 basic operations.

  The reason is that `law_id` promises that two laws share an identifier only
  if evaluating them returns the same bits for every `x`, and IEEE 754 does not
  require the transcendentals to be correctly rounded: the platform
  implementation is free to differ between operating system, CPU and compiler.
  Measured on one machine by changing nothing but the `std` feature,
  `generate_multi_sine` differed in 1 sample of 64 and `analyze_signal` in 6
  fields of 9. After the move, all five recorded scenarios agree.

- `math::FloatExt`, the `no_std` shim, now carries only `sqrt`, `floor` and
  `round`. IEEE 754 specifies each of them exactly, so the shim and the
  platform agree bit for bit; the module documentation no longer claims that
  the transcendentals stay "within 1-2 ulp", which is not a statement about
  determinism.

### Added

- `law::SEMANTICS_ID`, re-exported from `alice-det-math`: the value to pass as
  `semantics_id` to `law::SignalLaw::law_id` for a law evaluated through this
  crate. Previously a caller had nothing to pass but an invented array, and an
  identifier whose arithmetic half is invented does not identify arithmetic.
- `tests/determinism_golden.rs` — records the bit patterns of the law
  identifier, the evaluation it stands for, the sinusoid generators, the
  spectrum, the reconstruction, the entropy and one path with no transcendental
  on it. Nothing in the crate pinned a single bit before: the 193 existing
  tests all stayed green while the outputs above changed, because they compare
  against closed forms with tolerances. Each scenario checks a minimum byte
  count first, so a scenario that stopped producing values cannot pass by
  comparing nothing against nothing. CI runs the file on three operating
  systems in the default build and again in the `no_std` build, and every run
  has to produce the same digests.
- `clippy.toml` — `disallowed-methods` for the 50 float transcendental methods
  of `f32` and `f64`, so a platform implementation cannot come back unnoticed.
  `sqrt`, `floor`, `round`, `trunc`, `ceil` and `mul_add` are deliberately
  absent (IEEE 754 specifies all of them exactly). The entropy tests keep the
  platform `log2` behind a scoped allow, because an oracle whose expected value
  comes from the implementation under test proves nothing.

  The two gates catch different things, measured by reverting each change one
  at a time: of 10 such reversions, the golden caught 8 and the lint 7, with 2
  caught only by the lint (where the platform and `alice-det-math` happen to
  agree at the sampled points) and 3 only by the golden (changes to the
  identifier's field order and to the evaluation itself, which no lint can
  see). None survived both.

## [0.5.2] - 2026-10-07

### Added

- `law::SignalLaw::law_id` — a 32-byte content identifier for a law, so a
  stored result can name the law it came from. SHA-256 over the valid range,
  the coefficients and a caller-supplied identifier for the numeric semantics;
  the evidence, residual, provenance and oracle cases are excluded, so the same
  law fitted from two measurement runs gets one identifier. Equal identifier
  implies bit-identical `evaluate`; the converse is not guaranteed and the
  identifier is not a deduplication key. The byte layout is documented on the
  method and pinned by a golden digest.
- `law::LAW_ID_DOMAIN` and `law::SIGNAL_LAW_KIND` — the two tags the encoding
  mixes in, published so an independent implementation can reproduce an
  identifier byte for byte.
- `tests/analytic_law_id.rs` — injectivity over each evaluation input, metadata
  independence, the observability of a zero's sign, rejection of the inputs the
  encoding cannot canonicalise, and a cross-platform golden.

### Changed

- New dependency `sha2` (`default-features = false`, so the `no_std` build is
  unaffected) for the digest above.

## [0.5.1] - 2026-10-07

### Added
- `law` module (`no_std`): `SignalLaw` keeps a fixed-degree polynomial fitted by
  least squares together with its evidence, measured `ResidualStats`, the closed
  `ValidRange` of the evidence, `Provenance` and `OracleCase`s `evaluate` refuses
  `x` outside the range (`LawError::OutOfRange`) instead of extrapolating,
  `check_oracles` reports pass / fail / out of range, and `ingest` judges new
  evidence as `Supports` / `ParameterUpdate` (with the refitted law) /
  `ResidualGrew` / `Breaks` / `OutOfRange` / `NoEvidence` by documented rules
  The fit reuses the crate's Householder QR solver on `x` normalised to `[0, 1]`
- `tests/analytic_law.rs`: closed-form recovery at unsampled conditions, residual
  measured against the evidence, range refusal, oracle outcomes, each verdict,
  and degenerate input (empty / too few points / single `x` / NaN / repeated `x`)
- `law::SignalLawParts` with `SignalLaw::to_parts` / `from_parts` / `coefficients`:
  every field public for storing a law in another format; `from_parts` validates
  the parts (coefficients, finite values, domain, evidence inside the domain) and
  measures the residual again from the stored evidence instead of trusting it

### Changed
- `scripts/preflight.sh`: `cargo audit` keeps its advisory database under the build
  directory

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

[Unreleased]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.3.0...v0.3.1
[0.3.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.1...alice-zip-v0.3.0
[0.2.1]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.2.0...alice-zip-v0.2.1
[0.2.0]: https://github.com/ext-sakamoro/ALICE-Zip/compare/alice-zip-v0.1.0...alice-zip-v0.2.0
[0.1.0]: https://github.com/ext-sakamoro/ALICE-Zip/releases/tag/alice-zip-v0.1.0
