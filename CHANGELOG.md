# Changelog

All notable changes to the `alice-zip` Rust crate (repository root, crates.io)
are documented here. The CLI / FFI / Python native crate under `libalice/` has
its own [CHANGELOG](libalice/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `container` — a container that holds several payloads, each identified by
  its SHA-256: a 56-byte header (8-byte magic starting with a non-ASCII byte,
  major / minor version, semantics id), a section table (tag, critical flag,
  offset, length, SHA-256) and a trailing SHA-256 of the whole file.
  `Container::id` hashes the header and the table with domain separation.
  Readers refuse another major version, an unknown critical section, any
  reserved bit, a gap or extra byte, and a trailer or payload that does not
  match; unknown non-critical sections are kept. `SREF` sections refer to
  sections by SHA-256 (a missing one is refused), `LIDS` sections list law
  identifiers checked against the header's semantics id and, through
  `ContainerView::verify_law_ids`, against recomputed identifiers.
  `read_any` also reads `ALICE_ZIP` files of version 1.0 / 1.1 and refuses
  field values that were never written. `parse_legacy_alice_zip_header` returns the
  header as `LegacyHeader`; `LegacyHeader::verify_original` checks data offered
  as the original against `original_size` and, when one is recorded (not all
  zeros), `original_hash`. `ContainerView` checks a payload's
  hash only when it is read.
- `tests/container_oracle.rs` (25 tests): bytes and identifiers equal those of
  the independent reference writer `tests/data/container/container_ref.py`;
  every single-bit change of a fixture is refused by the check its position
  belongs to; degenerate input returns the stated error.
- `examples/container_roundtrip.rs`.
- CI: the container oracles run in the `no_std` build on each OS and under
  `wasmtime` on `wasm32-wasip1`; the wasm job fails when the summary reports
  no passed test.

## [0.8.0] - 2026-10-09

### Added

- `compression::compress_residual_xor` / `decompress_residual_xor` — a residual
  container that stores `original.to_bits() ^ model.to_bits()` instead of
  `original - model`. Reversible by construction, so every finite sample comes
  back unchanged: signed zeros, denormals, and values far smaller than their
  model all survive. Measured on a 100,000-sample sine fitted by
  `analyze_signal`, the subtraction form loses the **99 samples nearest the zero
  crossings** (where `|original| << |model|`, `fl(original - model)` rounds the
  original away and `model + residual` does not return it); the xor form loses
  none. It is also smaller on that signal (5,306 B against 8,450 B) because the
  xor of two close values is mostly zero bytes — on the degree-3 polynomial it
  is larger (3,916 B against 2,783 B), so the subtraction container stays for
  callers who prefer size over exactness
- `compression::ResidualCodec` + `residual_codec_default` +
  `*_with` variants of all three container constructors — the payload codec is
  now chosen explicitly and recorded in the container, so a reader never guesses
- `compression::residual_container_codec` — reads the codec back out of any
  container this module produces
- `scripts/claim_check.py` + a CI job: every `<!-- claim-test: NAME -->` in
  `README.md` / `README_ja.md` must resolve to a real `fn NAME`, the two
  documents must carry the same set of markers, and **0 markers fails**. The
  convention existed since the law module landed but nothing checked it
- `examples/compression_ratio.rs` — prints the benchmark table the READMEs
  quote, so the numbers there are reproducible rather than asserted

### Changed

- **Residual containers default to deflate instead of LZMA.** `lzma-rs` is pure
  Rust but its encoder is weak: on the same 400,000 bytes it produced
  **316,528 B where this crate's own zlib wrapper produced 8,742 B** (36x worse),
  and on the sine residual 46,916 B against 8,430 B. The README's promise that
  the fallback is "never worse than standard tools" was therefore wrong by 84x.
  Deflate comes from `flate2`, already a dependency, so nothing new is pulled in
  and the pure-Rust / `no_std`-friendly story is unchanged
- **The `level` argument now does something.** It is the deflate level `0..=9`
  (values above 9 are clamped to 9, and **0 means store**, the same as
  `zlib_compress`). How much the level matters depends on the input and its
  length — measured on the sine residual
  (`generate_sine_wave(n, 50.0, 1.0, 0.0, 0.0)` fitted by
  `analyze_signal(.., 8, 0.999)`, one coefficient):

  | n | non-zero residual | max | level 6 | level 9 | ratio |
  |---|---|---|---|---|---|
  | 4,096 | 3,008 | 1.192e-7 | 1,139 B | 1,009 B | 1.13x |
  | 100,000 | 74,100 | 1.788e-7 | 16,687 B | 8,430 B | 1.98x |

  ⚠️ An earlier draft of this entry quoted the 1.98x figure without the input
  it belongs to, which reads as a property of the level rather than of that
  signal at that length.
  The LZMA path still has fixed settings and ignores it, as before
- Containers are available with the `std` feature instead of requiring `lzma`;
  `ResidualCodec::Lzma` (and reading a version 0 container) still needs `lzma`
  and returns a clear error without it
- `compress_residual_lossless`'s documentation now states that the container
  stores the residual array exactly but that a pipeline built on the subtraction
  form is not bit-exact, and points at `compress_residual_xor`
- CI and `scripts/preflight.sh` build the library on the host without `std`
  under `-D warnings`, once as is and once with `alice-det-math/std` enabled.

### Fixed

- **`level 0` no longer silently becomes `level 1` in the residual containers.**
  The deflate level was floored with `clamp(1, 9)`, so "store it, do not
  compress" was unreachable through the containers while `zlib_compress` in the
  same module has always honoured 0: measured on the same 400,000 bytes,
  `zlib_compress(.., 0)` produced 400,071 (a store) and the container produced
  31,607 — byte-identical to level 1. ⚠️ Two public functions in one module gave
  the same argument two different meanings. The floor had no reason behind it
  and is gone; `tests/residual_container_oracle.rs` now pins that level 0
  produces more bytes than the input and differs from level 1 (the assertion
  goes red if the floor comes back)
- Both READMEs quoted parameters-only sizes under the heading "Lossless:
  Bit-perfect reconstruction" — 1400x for a sine that was in fact reconstructed
  with a 6e-8 error. The benchmark tables now separate **bit-exact**
  (75x / 102x / 806x), **lossy** (160x–1286x with the error stated) and
  **parameters only** (11,111x–20,000x with the error stated), and carry a
  `zlib alone` baseline column so the comparison is visible. `README_ja.md` was
  corrected in the same commit
- A build without the `std` feature failed under `-D warnings` with
  `trait FloatExt is never used` whenever another crate in the dependency graph
  enabled `alice-det-math`'s default `std` feature (for example a downstream
  crate that also depends on `alice-det-math`). Linking std made the float
  method calls (`x.sqrt()`, `x.floor()`, `x.round()`) resolve to std's inherent
  methods, so the `libm` shim was never used. The shim is now four functions in
  `math` (`floor_f32`, `sqrt_f32`, `sqrt_f64`, `round_f64`) that the call sites
  name explicitly, so the `std` feature of this crate alone decides which
  implementation runs. All four operations are exactly specified by IEEE 754,
  so the results are bit-identical either way.

### Format

- Containers written by 0.8.0 carry a version marker and a codec byte
  (`0xFD` lossless, `0xFE` xor, `0xFC` quantised). **Containers written by
  earlier releases still decode**; containers written by 0.8.0 need 0.8.0 or
  later to read

## [0.7.0] - 2026-10-08

### Changed

- **Breaking:** the reconstruction laws accumulate in `f64` and round to `f32`
  once on return, instead of accumulating in `f32` and adding the DC term
  last. `generators::{generate_sine_wave, generate_multi_sine,
  generate_from_coefficients}` therefore return different bits in the last
  places. Analytic behaviour is unchanged and `law_id` is unaffected (the
  polynomial path already evaluated in `f64`).

  The reason is that each law was implemented twice: here over whole arrays,
  and again in a consumer that answered single-sample queries without
  materialising a segment. On the same input at integer positions with
  `n = 1024` the two disagreed on **788 / 1024** samples for a sine,
  **881 / 1024** for a multi-sine and **962 / 1024** for a Fourier
  reconstruction. Against the `f64` closed form the `f32` accumulation was off
  by at most 9.16e-7 and the `f64` one by 5.95e-8, so the more accurate form
  became the law. `tests/determinism_golden.rs` re-records the two sinusoid
  digests for this reason.

### Added

- `generators::{sine_at, multi_sine_at, fourier_at, polynomial_at}` evaluate a
  law at one position, which may be fractional. These are **the** laws, and the
  array generators are now a `map` over them, so an array read and a point read
  cannot drift apart. A consumer that answers point queries calls these instead
  of keeping its own copy of the law.
- `tests/law_single_source.rs` pins both halves of that: the error of each
  array generator against the `f64` closed form stays inside one `f32` step
  (1.19e-7), and the array output equals the point output bit for bit over
  seven lengths including `n = 1`, `2`, `3` and `257`, with a byte-count gate so
  a comparison of nothing cannot pass. A separate case pins that the point
  evaluator does not round its position, which an integer-only comparison would
  miss.


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

- `.gitignore` を Rust / Python のビルド生成物に絞り、CI と test の comment を
  外部参照でなく理由そのものを書く形に直した (`.gitignore` は公開 package に
  同梱されるので、開発環境固有の除外は clone ごとの `.git/info/exclude` に置く)

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
