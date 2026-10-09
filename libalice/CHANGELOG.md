# Changelog

All notable changes to ALICE-Zip (libalice) will be documented in this file.

## [Unreleased]

### Changed (破壊的変更)
- `format::AliceFileHeader::from_bytes` は core crate の `container::parse_legacy_alice_zip_header` で header を検査してから読む (読み手を 1 箇所にする) 書き手が出したことのない値は別の値として読まず拒否する
  - 版 1.0 / 1.1 以外 → `FormatError::UnsupportedVersion` (major 2 を 1.1 として読んでいた)
  - 未知の `payload_type` → `FormatError::InvalidPayloadType` (`Procedural` として読んでいた)
  - engine の index が 0〜3 以外 → `FormatError::InvalidEngine`
  - 66 byte 未満の 1.1 header → `FormatError::TooShort` (1.0 header として読んでいた)
- `residual::ResidualData::from_bytes` は JSON header の `"version"` が 3 以上なら `ResidualError::UnsupportedVersion` (version 2 として読んでいた) Python 版と同じ規則
- `alice decompress` / `alice info` は `.alz` header の版が 1 以外、または mode が未知の file をエラーにする (版を読み飛ばし、未知の mode を raw LZMA として読んでいた)
- 移行: `FormatError` / `ResidualError` を網羅的に match している呼び出し側は新しい variant (`UnsupportedVersion` / `InvalidEngine`) を足す 未知の `payload_type` が `Procedural` に落ちることに頼っていた呼び出し側はエラーとして扱う (そうした file はどの書き手も出していない) Python package と本 crate が書いた file (版 1.0 / 1.1、定義された値) はこれまでどおり読める

### Fixed
- `residual` の delta は新しい method `BitDelta` (`"bitdelta"`) で書く: 隣り合う f32 の bit パターンの差を wrapping な u32 で持ち (先頭は先頭の bit パターン)、xz で圧縮する どの値も bit 単位で戻る (ランダムな bit パターン 10,000 個と NaN / 非正規化数 / ±0 / 無限大の往復の試験) 差分の列は Python の書き手と byte 単位で一致する (`tests/data/residual/bitdelta_streams.txt`) 旧 `"delta"` は `base_value` があれば読み、無ければ `ResidualError::DeltaWithoutBase` (Python の旧 writer は先頭の値を失っていた) `decompress_residual_delta` は `Result` を返す (復元に失敗すると圧縮前の byte を値として読んでいた)
- 残差の LZMA は xz (Python の `lzma.compress` の既定) と LZMA alone (本 crate の旧 writer) を先頭の magic で判別して読む
- 残差の書き手は、圧縮に失敗して別の方式に落ちた時に実際に使った方式を記録する (zlib / lzma / quantized の失敗で、method と中身が食い違う file を書いていた) 全部失敗すれば `None` の生の float
- 元データの dtype が float32 以外、または shape が 1 次元でない residual (Python の書き手が記録する) は `ResidualError::UnsupportedLayout` で拒否する この読み手は 1 次元の float32 しか返せないため どちらの読み手がどの file を受け付けるかは `tests/data/residual/acceptance.txt` に表で持ち、Rust と Python の両方の試験が読む
- `residual::ResidualData::from_bytes` は `"version"` が JSON の整数でなければ (float `2.0`、文字列 `"2"`) `ResidualError::InvalidHeader` 文字列の `"2"` を版 2 として読んでいた Python 版と同じ規則
- `alice compress --bits 16` が 16 bit で量子化する これまでは mode 11 (16 bit と表示) を書きながら中身は常に 8 bit で量子化していた (実測: 最大誤差 0.01176 = 8 bit の半 step、file の大きさも `--bits 8` と同じ) 本物の 16 bit は新しい mode 12 で書く mode 11 の既存 file は中身どおり 8 bit として読み続ける (書くことはもう無い) header の版は変わらない mode 12 の file は、未知の mode を拒否する読み手 (この版以降) にしか読めず、以前の読み手は raw LZMA として誤読していたので、以前の CLI で開かないこと

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
