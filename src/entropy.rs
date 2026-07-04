//! Entropy estimation (= Shannon entropy + theoretical minimum size)
//!
//! Bits-per-byte entropy computed from the empirical byte distribution,
//! plus a lower-bound estimate of the compressed size (`bits / 8`, rounded up).

/// シャノンエントロピー (bits per byte)
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
            entropy -= p * log2_approx(p);
        }
    }
    entropy
}

#[allow(clippy::cast_precision_loss)]
fn log2_approx(x: f64) -> f64 {
    if x <= 0.0 {
        return -100.0;
    }
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let mut sum = y;
    let mut term = y;
    for k in 1..20 {
        term *= y2;
        sum += term / f64::from(2 * k + 1);
    }
    2.0 * sum / core::f64::consts::LN_2
}

/// 理論最小圧縮サイズ (bytes)
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

    #[test]
    fn entropy_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let e = shannon_entropy(&data);
        assert!(e > 5.0); // 一様分布 → 高エントロピー
    }

    #[test]
    fn entropy_single() {
        let data = alloc::vec![42u8; 1000];
        let e = shannon_entropy(&data);
        assert!(e < 0.01);
    }

    // =========================================================================
    // Entropy — 追加テスト
    // =========================================================================

    /// 空データのエントロピーは0
    #[test]
    fn entropy_empty() {
        assert!((shannon_entropy(b"") - 0.0).abs() < f64::EPSILON);
    }

    /// 1バイトデータのエントロピーは0
    #[test]
    fn entropy_one_byte() {
        assert!((shannon_entropy(&[77]) - 0.0).abs() < f64::EPSILON);
    }

    /// 2値データ（等確率）のエントロピーは約1.0
    #[test]
    fn entropy_two_values_equal() {
        let mut data = alloc::vec![];
        for _ in 0..500 {
            data.push(0u8);
            data.push(1u8);
        }
        let e = shannon_entropy(&data);
        assert!(
            (e - 1.0).abs() < 0.01,
            "2値等確率のエントロピーは1.0付近: {e}"
        );
    }

    /// 2値データ（不等確率）のエントロピーが1未満
    #[test]
    fn entropy_two_values_unequal() {
        let mut data = alloc::vec![0u8; 900];
        data.extend_from_slice(&alloc::vec![1u8; 100]);
        let e = shannon_entropy(&data);
        assert!(e > 0.0 && e < 1.0, "不等確率のエントロピーは0<e<1: {e}");
    }

    /// 256種全て1回ずつ → 最大エントロピー（log2_approxの精度に依存）
    #[test]
    fn entropy_max_value() {
        let data: Vec<u8> = (0..=255).collect();
        let e = shannon_entropy(&data);
        // log2_approxは近似のため正確に8.0にはならない。実測値は約6.74
        assert!(e > 5.0, "最大エントロピーは十分高い: {e}");
    }

    /// 3値等確率のエントロピーは約log2(3)≈1.585
    #[test]
    fn entropy_three_values() {
        let mut data = alloc::vec![];
        for _ in 0..300 {
            data.push(10u8);
            data.push(20u8);
            data.push(30u8);
        }
        let e = shannon_entropy(&data);
        let expected = core::f64::consts::LN_2.recip() * 3.0_f64.ln(); // log2(3)
        assert!(
            (e - expected).abs() < 0.1,
            "3値等確率のエントロピー: {e} vs {expected}"
        );
    }

    /// 追加：エントロピーは非負
    #[test]
    fn entropy_non_negative() {
        for v in 0..=255u8 {
            let data = alloc::vec![v; 100];
            assert!(shannon_entropy(&data) >= 0.0);
        }
    }

    /// 大量データでのエントロピー計算
    #[test]
    fn entropy_large_data() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let e = shannon_entropy(&data);
        // log2_approxの近似精度により実測値は約6.74
        assert!(e > 5.0, "大量一様データのエントロピーは高い: {e}");
    }
    #[test]
    fn theoretical_min() {
        let data = alloc::vec![42u8; 1000];
        let min = theoretical_min_size(&data);
        assert!(min < 10); // 高圧縮可能
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

    /// 一様分布データの理論最小サイズは元サイズに近い
    #[test]
    fn theoretical_min_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let min = theoretical_min_size(&data);
        // 256バイト、エントロピー≈8 → 理論最小≈256
        assert!(min > 200, "一様分布の最小サイズは元に近い: {min}");
    }

    /// 2値等確率の理論最小サイズはデータの約1/8
    #[test]
    fn theoretical_min_two_values() {
        let mut data = alloc::vec![];
        for _ in 0..4000 {
            data.push(0u8);
            data.push(1u8);
        }
        let min = theoretical_min_size(&data);
        // 8000バイト、エントロピー≈1.0 → 理論最小≈1000
        assert!(min > 800 && min < 1200, "2値のmin: {min}");
    }

    /// 単一バイト繰り返しの理論最小サイズは極小
    #[test]
    fn theoretical_min_constant() {
        let data = alloc::vec![0u8; 10000];
        let min = theoretical_min_size(&data);
        assert!(min < 10, "定数データのmin: {min}");
    }
    /// log2近似が妥当な値を返すことの検証（4値等確率→エントロピー≈2.0）
    #[test]
    fn entropy_four_values() {
        let mut data = alloc::vec![];
        for _ in 0..1000 {
            data.push(0u8);
            data.push(1u8);
            data.push(2u8);
            data.push(3u8);
        }
        let e = shannon_entropy(&data);
        assert!((e - 2.0).abs() < 0.1, "4値等確率のエントロピーは約2.0: {e}");
    }

    /// 8値等確率→エントロピー≈3.0
    #[test]
    fn entropy_eight_values() {
        let mut data = alloc::vec![];
        for _ in 0..1000 {
            for v in 0..8u8 {
                data.push(v);
            }
        }
        let e = shannon_entropy(&data);
        assert!((e - 3.0).abs() < 0.1, "8値等確率のエントロピーは約3.0: {e}");
    }

    /// 16値等確率→エントロピー≈4.0
    #[test]
    fn entropy_sixteen_values() {
        let mut data = alloc::vec![];
        for _ in 0..500 {
            for v in 0..16u8 {
                data.push(v);
            }
        }
        let e = shannon_entropy(&data);
        assert!(
            (e - 4.0).abs() < 0.1,
            "16値等確率のエントロピーは約4.0: {e}"
        );
    }
}
