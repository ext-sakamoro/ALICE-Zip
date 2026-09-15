//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f32` / `f64` の inherent method (`sin` / `sqrt` / `log2` 等) を
//! そのまま使う (本 module は空)
//!
//! `std` なし (`--no-default-features`): core には float の超越関数が無いため、
//! 同名 method を [`FloatExt`] trait で提供し、実装は [`libm`] (pure Rust、
//! `no_std`) に委譲する 各 module は
//! `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//! (`abs` / `clamp` / `is_finite` は core にあるので shim 不要)
//!
//! 精度注意: `libm` と platform libm (glibc / macOS / MSVC CRT) は最終 ulp で
//! 異なりうる 法則 (式の形・operand 順) は両 build で同一なので、差は超越関数
//! 1 呼び出しあたり ≤ 1-2 ulp に留まる

#[cfg(not(feature = "std"))]
pub trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn floor(self) -> Self;
    fn log2(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
}
