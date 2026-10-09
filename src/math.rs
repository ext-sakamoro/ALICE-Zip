//! IEEE 754 が厳密に規定する float 演算 (`sqrt` / `floor` / `round`) の入口
//!
//! `std` あり: `f32` / `f64` の inherent method を呼ぶ
//!
//! `std` なし (`--no-default-features`): core には無いので [`libm`] (pure Rust、
//! `no_std`) に委譲する (`abs` / `clamp` / `is_finite` は core にあるので不要)
//!
//! 呼び出し側は method 構文 (`x.sqrt()`) でなく本 module の関数を呼ぶ method
//! 構文だと、本 crate が `std` なしで build されていても、依存 graph の別の
//! crate が `std` を link した時点で `std` の inherent method が選ばれ、どの実装を
//! 通るかが本 crate の feature でなく graph 全体で決まる (実測: `alice-det-math`
//! の `std` が feature 統合で有効になると、以前の trait 方式の shim が 1 度も
//! 使われず `dead_code` になった) 関数にすると、どちらを通るかは本 crate の
//! `std` feature だけで決まる
//!
//! 置くのは `sqrt` / `floor` / `round` だけで、どれも **IEEE 754 が結果を
//! 1 通りに定めている** ので、`libm` と platform libm (glibc / macOS / MSVC CRT)
//! で bit 単位に一致する 実際に使う型の組だけを置く (使わない関数は `std` の
//! 有無によらず `dead_code` になるため)
//!
//! 超越関数 (`sin` / `cos` / `atan2` / `log2` 等) は本 module に**置かない**
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

/// `f32` の床関数
#[inline]
pub(crate) fn floor_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floorf(x)
    }
}

/// `f32` の平方根
#[inline]
pub(crate) fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}

/// `f64` の平方根
#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

/// `f64` の最近接丸め (0.5 は 0 から遠い側、`f64::round` と同じ)
#[inline]
pub(crate) fn round_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::round(x)
    }
}

#[cfg(test)]
mod tests {
    use super::{floor_f32, round_f64, sqrt_f32, sqrt_f64};

    // std あり / なしの両方で同じ bit を返すことを、値と符号の境界で固定する
    // (期待値は IEEE 754 が一意に定める結果で、被検査関数からは作らない)
    #[test]
    fn exact_operations_return_the_ieee_result() {
        assert_eq!(floor_f32(-0.5).to_bits(), (-1.0_f32).to_bits());
        assert_eq!(floor_f32(2.999_999_8).to_bits(), 2.0_f32.to_bits());
        assert_eq!(floor_f32(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(sqrt_f32(2.25).to_bits(), 1.5_f32.to_bits());
        assert_eq!(sqrt_f32(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert!(sqrt_f32(-1.0).is_nan());
        assert_eq!(sqrt_f64(0.0625).to_bits(), 0.25_f64.to_bits());
        assert_eq!(sqrt_f64(2.0).to_bits(), 0x3FF6_A09E_667F_3BCD);
        assert_eq!(round_f64(2.5).to_bits(), 3.0_f64.to_bits());
        assert_eq!(round_f64(-2.5).to_bits(), (-3.0_f64).to_bits());
        assert_eq!(round_f64(-0.4).to_bits(), (-0.0_f64).to_bits());
    }
}
