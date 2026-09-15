//! Entropy estimation (= Shannon entropy + theoretical minimum size)
//!
//! Bits-per-byte entropy computed from the empirical byte distribution,
//! plus a lower-bound estimate of the compressed size (`bits / 8`, rounded up).
//!
//! Law: `H = -Σ p_i · log2(p_i)` over the 256 byte values with `p_i > 0`
//! `log2` is exact (`f64::log2` / `libm::log2`), so `k` equiprobable symbols
//! give `H = log2(k)` to within a few ulp (uniform 256 → exactly `8.0`)
//! Versions ≤ 0.3 used a 20-term series that saturated at `≈ 6.74` for the
//! uniform case; `tests/analytic_oracle.rs` pins the exact values

// (test builds link std, whose inherent methods shadow the trait → allow)
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::FloatExt;

/// シャノンエントロピー (bits per byte)
///
/// `data` が空なら `0.0` 出力は常に `0.0 ..= 8.0`
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut entropy = 0.0;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// 理論最小圧縮サイズ (bytes) = `ceil(H · len / 8)`
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn theoretical_min_size(data: &[u8]) -> usize {
    let entropy = shannon_entropy(data);
    let bits = entropy * data.len() as f64;
    let v = bits / 8.0;
    let i = v as usize;
    if (i as f64) < v {
        i + 1
    } else {
        i
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// k 値等確率の entropy は log2(k) に一致する (解析解)
    fn assert_equiprobable(k: u32, repeats: usize) {
        let mut data = Vec::with_capacity(k as usize * repeats);
        for _ in 0..repeats {
            for v in 0..k {
                data.push(v as u8);
            }
        }
        let e = shannon_entropy(&data);
        let expected = f64::from(k).log2();
        assert!(
            (e - expected).abs() < 1e-9,
            "{k} 値等確率: {e} vs log2({k}) = {expected}"
        );
    }

    #[test]
    fn entropy_uniform_256_is_exactly_8() {
        let data: Vec<u8> = (0..=255).collect();
        assert!((shannon_entropy(&data) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_single() {
        let data = alloc::vec![42u8; 1000];
        assert_eq!(shannon_entropy(&data), 0.0);
    }

    /// 空データのエントロピーは0
    #[test]
    fn entropy_empty() {
        assert_eq!(shannon_entropy(b""), 0.0);
    }

    /// 1バイトデータのエントロピーは0
    #[test]
    fn entropy_one_byte() {
        assert_eq!(shannon_entropy(&[77]), 0.0);
    }

    #[test]
    fn entropy_equiprobable_matches_log2_k() {
        for k in [2u32, 3, 4, 8, 16, 100, 256] {
            assert_equiprobable(k, 7);
        }
    }

    /// 2値データ（不等確率 0.9 / 0.1）= 0.469 bit (解析解)
    #[test]
    fn entropy_two_values_unequal() {
        let mut data = alloc::vec![0u8; 900];
        data.extend_from_slice(&alloc::vec![1u8; 100]);
        let e = shannon_entropy(&data);
        let expected = -(0.9_f64 * 0.9_f64.log2() + 0.1_f64 * 0.1_f64.log2());
        assert!((e - expected).abs() < 1e-9, "{e} vs {expected}");
    }

    /// 追加：エントロピーは非負かつ ≤ 8
    #[test]
    fn entropy_bounds() {
        for v in 0..=255u8 {
            let data = alloc::vec![v; 100];
            assert!(shannon_entropy(&data) >= 0.0);
        }
        // 39 full cycles of 0..=255 → exactly uniform → exactly 8 bit
        let data: Vec<u8> = (0..256 * 39).map(|i| (i % 256) as u8).collect();
        let e = shannon_entropy(&data);
        assert!((e - 8.0).abs() < 1e-12, "9984 sample uniform: {e}");
        // 10000 samples: 16 symbols occur 40×, 240 occur 39× → strictly < 8
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let e = shannon_entropy(&data);
        assert!(e < 8.0 && e > 7.999, "near-uniform: {e}");
    }

    #[test]
    fn theoretical_min() {
        let data = alloc::vec![42u8; 1000];
        assert_eq!(theoretical_min_size(&data), 0);
    }

    /// 空データの理論最小サイズは0
    #[test]
    fn theoretical_min_empty() {
        assert_eq!(theoretical_min_size(b""), 0);
    }

    /// 1バイトの理論最小サイズは0（エントロピー=0）
    #[test]
    fn theoretical_min_single() {
        assert_eq!(theoretical_min_size(&[99]), 0);
    }

    /// 一様分布 256 byte → 8 bit × 256 / 8 = 256 byte ちょうど
    #[test]
    fn theoretical_min_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        assert_eq!(theoretical_min_size(&data), 256);
    }

    /// 2値等確率 8000 byte → 1 bit × 8000 / 8 = 1000 byte ちょうど
    #[test]
    fn theoretical_min_two_values() {
        let mut data = alloc::vec![];
        for _ in 0..4000 {
            data.push(0u8);
            data.push(1u8);
        }
        assert_eq!(theoretical_min_size(&data), 1000);
    }

    /// 端数は切り上げ: 3 値等確率 3 byte → log2(3) × 3 / 8 = 0.594 → 1
    #[test]
    fn theoretical_min_rounds_up() {
        assert_eq!(theoretical_min_size(&[1, 2, 3]), 1);
    }
}
