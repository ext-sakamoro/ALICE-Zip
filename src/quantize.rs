//! Uniform min / max quantisation of `f32` samples to 8 or 16 bits (`no_std`)
//!
//! Law: `q = round((v − min) / scale · (2^bits − 1))`, `v' = q / (2^bits − 1) ·
//! scale + min` with `scale = max − min` (or `1.0` when the data is constant,
//! so every sample quantises to `0` and dequantises back to `min`) The
//! reconstruction error is at most `scale / (2 · (2^bits − 1))` per sample —
//! pinned by `tests/analytic_oracle.rs`
//!
//! 16-bit values are serialised **little-endian**, 2 bytes per sample, so the
//! byte streams are identical on every architecture These are the residual
//! payloads of the `.alice` container (`compression::compress_residual_quantized`)
//! and of `alice-edge` coefficient batches; the law moved here from `libalice`
//! in 0.5.0 (single home, see `generators`)

use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::FloatExt;

/// `(min, scale)` of the data with the constant-data guard applied
fn range(data: &[f32]) -> (f64, f64) {
    let min_val = f64::from(data.iter().copied().fold(f32::INFINITY, f32::min));
    let max_val = f64::from(data.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let scale = max_val - min_val;
    // Constant data: scale = 1.0 avoids the division by zero and quantises
    // every sample to 0 (= min), which is the exact reconstruction
    let scale = if scale < 1e-10 { 1.0 } else { scale };
    (min_val, scale)
}

/// Quantise `f32` samples to 8 bits
///
/// Returns `(quantised bytes, min, scale)`; an empty input returns
/// `(vec![], 0.0, 1.0)` NaN / infinite samples are outside the law's domain
/// (their quantised value is unspecified, everything else is unaffected)
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn quantize_8bit(data: &[f32]) -> (Vec<u8>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let (min_val, scale) = range(data);
    let quantized: Vec<u8> = data
        .iter()
        .map(|&v| {
            let normalized = ((f64::from(v) - min_val) / scale).clamp(0.0, 1.0);
            (normalized * 255.0).round() as u8
        })
        .collect();
    (quantized, min_val, scale)
}

/// Inverse of [`quantize_8bit`]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn dequantize_8bit(data: &[u8], min_val: f64, scale: f64) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            let normalized = f64::from(v) / 255.0;
            (normalized * scale + min_val) as f32
        })
        .collect()
}

/// Quantise `f32` samples to 16 bits (little-endian `u16` byte pairs)
///
/// Returns `(quantised bytes, min, scale)`; an empty input returns
/// `(vec![], 0.0, 1.0)`
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn quantize_16bit(data: &[f32]) -> (Vec<u8>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let (min_val, scale) = range(data);
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| {
            let normalized = ((f64::from(v) - min_val) / scale).clamp(0.0, 1.0);
            ((normalized * 65535.0).round() as u16).to_le_bytes()
        })
        .collect();
    (bytes, min_val, scale)
}

/// Inverse of [`quantize_16bit`] (a trailing odd byte is ignored)
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn dequantize_16bit(data: &[u8], min_val: f64, scale: f64) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let v = u16::from_le_bytes([chunk[0], chunk[1]]);
            let normalized = f64::from(v) / 65535.0;
            (normalized * scale + min_val) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn empty_and_constant() {
        assert_eq!(quantize_8bit(&[]), (Vec::new(), 0.0, 1.0));
        assert_eq!(quantize_16bit(&[]), (Vec::new(), 0.0, 1.0));
        let (q, min, scale) = quantize_8bit(&[2.5; 7]);
        assert_eq!((q, min, scale), (vec![0; 7], 2.5, 1.0));
        assert_eq!(dequantize_8bit(&[0; 7], 2.5, 1.0), vec![2.5; 7]);
        let (q, min, scale) = quantize_16bit(&[-1.0; 3]);
        assert_eq!((q, min, scale), (vec![0; 6], -1.0, 1.0));
    }

    #[test]
    fn endpoints_are_exact_and_error_is_bounded() {
        let data = [-3.0f32, 0.5, 1.25, 7.0];
        let (q, min, scale) = quantize_8bit(&data);
        assert_eq!((q[0], q[3]), (0, 255));
        for (a, b) in dequantize_8bit(&q, min, scale).iter().zip(&data) {
            assert!((a - b).abs() <= 10.0 / 255.0 / 2.0 + 1e-6);
        }
        let (q, min, scale) = quantize_16bit(&data);
        assert_eq!(q.len(), 8);
        assert_eq!((q[0], q[1], q[6], q[7]), (0, 0, 0xFF, 0xFF));
        for (a, b) in dequantize_16bit(&q, min, scale).iter().zip(&data) {
            assert!((a - b).abs() <= 10.0 / 65535.0 / 2.0 + 1e-6);
        }
    }

    #[test]
    fn sixteen_bit_is_little_endian() {
        // 0.0 → 0, 1.0 → 65535 = [0xFF, 0xFF], 0.5 → 32768 = [0x00, 0x80]
        let (q, _, _) = quantize_16bit(&[0.0, 0.5, 1.0]);
        assert_eq!(q, vec![0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF]);
        assert_eq!(dequantize_16bit(&[0x00, 0x80, 0xAA], 0.0, 1.0).len(), 1);
    }
}
