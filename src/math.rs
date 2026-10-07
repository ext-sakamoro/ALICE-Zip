//! `no_std` 用の float 数学関数 shim (IEEE 754 が厳密に規定する演算のみ)
//!
//! `std` あり: `f32` / `f64` の inherent method をそのまま使う (本 module は空)
//!
//! `std` なし (`--no-default-features`): core には無いので同名 method を
//! [`FloatExt`] trait で提供し、実装は [`libm`] (pure Rust、`no_std`) に委譲する
//! 各 module は `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で
//! 取り込む (`abs` / `clamp` / `is_finite` は core にあるので shim 不要)
//!
//! 本 trait が持つのは `sqrt` / `floor` / `round` の 3 つだけで、どれも
//! **IEEE 754 が結果を 1 通りに定めている** ので、`libm` と platform libm
//! (glibc / macOS / MSVC CRT) で bit 単位に一致する
//!
//! 超越関数 (`sin` / `cos` / `atan2` / `log2` 等) は本 trait に**置かない**
//! IEEE 754 はこれらに正確丸めを要求しないので実装ごとに最終 ulp が違い、
//! 「差は 1-2 ulp に留まる」は決定論の主張にならない (1 ulp 違えば別の bit で、
//! 法則の内容識別子 [`crate::law::SignalLaw::law_id`] が指す評価結果が変わる)
//! ⇒ `alice-det-math` 経由に統一し、`clippy.toml` の `disallowed-methods` で
//! platform 版への逆流を禁止している
//!
//! 実測 (2026-10-08、同一機で feature set のみを変えた比較): 超越関数を
//! platform libm に置いていた 0.5.2 では `generate_multi_sine` が 64 sample の
//! うち 1、`analyze_signal` が 9 field のうち 6 で bits が違った `alice-det-math`
//! へ移した後は 5 項目すべて一致する

// 本 trait が使われるのは `no_std` の lib build だけ test build は std が
// 繋がり inherent method が trait を隠すので、そこでは未使用になる
#[cfg(not(feature = "std"))]
#[cfg_attr(test, allow(dead_code))]
pub trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
}
