//! ALICE-Zip — Compression primitives + procedural signal generators
//!
//! LZ77 sliding window / dictionary coding / Shannon entropy / byte-pair
//! encoding, plus the generator laws (polynomial / Fourier / Perlin) that let
//! downstream crates store *how to regenerate* a signal instead of its samples
//! ("send the law, not the data").
//!
//! The crate is `no_std + alloc` by default-features-off; the `std` feature
//! (on by default) adds the zlib wrappers and `std::error::Error`.
//!
//! # Quick start
//!
//! ```
//! use alice_zip::prelude::*;
//!
//! // Compression primitives (no_std)
//! let tokens = lz77_encode(b"abcabcabcabc", 256, 32);
//! assert_eq!(lz77_decode(&tokens)?, b"abcabcabcabc");
//! assert_eq!(shannon_entropy(&(0..=255u8).collect::<Vec<_>>()), 8.0);
//!
//! // Generator laws: fit a series, keep the coefficients, regenerate
//! let series: Vec<f32> = (0..64).map(|i| 3.0 + 2.0 * i as f32).collect();
//! let (coeffs, degree, err) = fit_polynomial(&series, 4, 1e-6).unwrap();
//! assert_eq!((degree, generate_polynomial(64, &coeffs) == series), (1, true));
//!
//! let (bins, dc) = analyze_signal(&generate_sine_wave(32, 1.0, 2.0, 0.0, 0.5), 4, 0.99);
//! let regenerated = generate_from_coefficients(32, &bins, dc);
//! # assert_eq!(regenerated.len(), 32);
//! # Ok::<(), alice_zip::ZipError>(())
//! ```
//!
//! # Module 構成
//!
//! | Module | 内容 | feature |
//! |--------|------|---------|
//! | [`lz77`] | LZ77 sliding-window encode / decode + [`lz77::LzToken`] | — |
//! | [`dictionary`] | phrase → index 辞書 [`dictionary::Dictionary`] (FIFO eviction) | — |
//! | [`entropy`] | Shannon entropy + 理論最小サイズ | — |
//! | [`bpe`] | Byte-pair encoding (最頻ペア検出 + 置換) | — |
//! | [`generators`] | polynomial / Fourier / Perlin generator laws | `fft` / `parallel` で加速 path 追加 |
//! | [`quantize`] | 8 / 16-bit min-max quantisation of `f32` samples | — |
//! | [`compression`] | zlib wrappers ([`flate2`]); LZMA + `.alice` residual containers | `std`; `lzma` |
//! | [`error`] | 共通 [`error::ZipError`] | — |
//! | [`prelude`] | 主要 API 一括 re-export | — |
//!
//! # no_std
//!
//! `--no-default-features` で `core + alloc` のみに依存する float の超越関数は
//! [`libm`] に委譲するため、`std` build と最終 ulp が異なることがある (法則は同一)
//! 64-bit atomics 等の前提は無く、`thumbv7em-none-eabihf` で CI が rlib build を検証する
//!
//! # backward compatibility
//!
//! v0.1.0 まで crate ルート直下に定義していた項目は module 移動後も
//! ルートから `pub use` で公開している

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod bpe;
#[cfg(feature = "std")]
pub mod compression;
pub mod dictionary;
pub mod entropy;
pub mod error;
pub mod generators;
pub mod lz77;
pub(crate) mod math;
pub mod prelude;
pub mod quantize;

#[cfg(test)]
mod integration_tests;

// Backward-compatible re-exports (= v0.1.0 まで crate root で提供していた API)
pub use crate::bpe::{bpe_replace, find_most_frequent_pair};
pub use crate::dictionary::Dictionary;
pub use crate::entropy::{shannon_entropy, theoretical_min_size};
pub use crate::error::ZipError;
pub use crate::lz77::{lz77_decode, lz77_encode, LzToken};
