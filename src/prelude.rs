//! Convenience re-export (= `use alice_zip::prelude::*;` で主要 API 一括取得)
//!
//! `lz77` / `dictionary` / `entropy` / `bpe` / `error` の公開型 + 関数を
//! prelude 経由で 1 行 import できるようにする

pub use crate::bpe::{bpe_replace, find_most_frequent_pair};
pub use crate::dictionary::Dictionary;
pub use crate::entropy::{shannon_entropy, theoretical_min_size};
pub use crate::error::ZipError;
pub use crate::lz77::{lz77_decode, lz77_encode, LzToken};
