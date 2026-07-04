//! ALICE-Zip — Compression engine
//!
//! LZ77 スライディングウィンドウ / 辞書符号化 / エントロピー推定 / Byte-pair encoding
//! を提供する `no_std + alloc` の圧縮 primitives crate
//!
//! # Module 構成
//!
//! | Module | 内容 |
//! |--------|------|
//! | [`lz77`] | LZ77 sliding-window encode / decode + [`lz77::LzToken`] |
//! | [`dictionary`] | phrase → index 辞書 [`dictionary::Dictionary`] (LRU 風 eviction) |
//! | [`entropy`] | Shannon entropy + 理論最小サイズ |
//! | [`bpe`] | Byte-pair encoding (最頻ペア検出 + 置換) |
//! | [`error`] | 共通 [`error::ZipError`] |
//! | [`prelude`] | 主要 API 一括 re-export |
//!
//! # backward compatibility
//!
//! v0.1.0 まで crate ルート直下に定義していた項目は module 移動後も
//! ルートから `pub use` で公開しているため、既存 downstream (bindings /
//! libalice / examples) はそのまま動作する

#![no_std]
extern crate alloc;

pub mod bpe;
pub mod dictionary;
pub mod entropy;
pub mod error;
pub mod lz77;
pub mod prelude;

#[cfg(test)]
mod integration_tests;

// Backward-compatible re-exports (= v0.1.0 まで crate root で提供していた API)
pub use crate::bpe::{bpe_replace, find_most_frequent_pair};
pub use crate::dictionary::Dictionary;
pub use crate::entropy::{shannon_entropy, theoretical_min_size};
pub use crate::error::ZipError;
pub use crate::lz77::{lz77_decode, lz77_encode, LzToken};
