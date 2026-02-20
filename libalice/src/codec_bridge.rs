//! ALICE-Zip × ALICE-Codec Bridge
//!
//! Compress wavelet coefficients using ALICE-Zip's residual compression,
//! enabling efficient storage of Codec sub-band data.
//!
//! # Architecture
//!
//! ```text
//! ALICE-Codec Wavelet Coefficients (i32)
//!   → Convert to f32
//!   → ALICE-Zip quantized residual compression (LZMA)
//!   → Compact byte stream
//! ```
//!
//! LZMA preset is selected based on sub-band frequency characteristics:
//! - Low-frequency (LLL): High preset (better ratio, acceptable speed)
//! - High-frequency (HHH): Low preset (fast, noise-like data compresses poorly)

use alice_codec::SubBand3D;
use crate::compression;

/// Compressed sub-band data with metadata for reconstruction.
pub struct CompressedSubBand {
    /// LZMA-compressed coefficient data.
    pub data: Vec<u8>,
    /// Sub-band identifier.
    pub subband: SubBand3D,
    /// Number of coefficients stored.
    pub count: usize,
}

/// Select LZMA preset based on sub-band frequency characteristics.
///
/// Low-frequency sub-bands have smooth, correlated data that benefits from
/// higher compression effort. High-frequency sub-bands contain noise-like
/// data where extra effort yields diminishing returns.
fn select_preset(subband: SubBand3D) -> u32 {
    match subband.quant_strength() {
        1 => 6,     // LLL: DC, very compressible
        2 => 5,     // LLH/LHL/HLL: low-freq, good compression
        4 => 3,     // LHH/HLH/HHL: mid-freq, balanced
        8 => 1,     // HHH: high-freq noise, fast is fine
        _ => 4,     // Fallback
    }
}

/// Select quantization bits based on sub-band importance.
///
/// DC sub-band (LLL) uses 16-bit quantization to preserve fidelity.
/// Higher frequency sub-bands use 8-bit since quantization loss is
/// less perceptually significant.
fn select_bits(subband: SubBand3D) -> u8 {
    if subband.is_dc() {
        16
    } else if subband.quant_strength() <= 2 {
        16
    } else {
        8
    }
}

/// Compress wavelet sub-band coefficients using adaptive residual compression.
///
/// Converts integer wavelet coefficients to f32, then applies ALICE-Zip's
/// quantized LZMA compression with preset selected by sub-band frequency tier.
///
/// # Arguments
/// * `coefficients` - Integer wavelet coefficients from ALICE-Codec
/// * `subband` - Sub-band identifier (determines compression strategy)
///
/// # Returns
/// Compressed byte stream including quantization header
pub fn compress_subband(coefficients: &[i32], subband: SubBand3D) -> std::io::Result<CompressedSubBand> {
    let floats: Vec<f32> = coefficients.iter().map(|&v| v as f32).collect();
    let preset = select_preset(subband);
    let bits = select_bits(subband);

    let data = compression::compress_residual_quantized(&floats, bits, preset)?;

    Ok(CompressedSubBand {
        data,
        subband,
        count: coefficients.len(),
    })
}

/// Decompress wavelet sub-band coefficients.
///
/// Returns integer coefficients reconstructed from the compressed stream.
/// Note: lossy quantization means reconstructed values are approximate.
pub fn decompress_subband(compressed: &CompressedSubBand) -> std::io::Result<Vec<i32>> {
    let floats = compression::decompress_residual_quantized(&compressed.data)?;
    Ok(floats.iter().map(|&v| v.round() as i32).collect())
}

/// Compress a full set of 3D wavelet sub-bands.
///
/// Returns a Vec of compressed sub-band results. Each sub-band is compressed
/// independently with parameters tuned to its frequency characteristics.
pub fn compress_all_subbands(subbands: &[(SubBand3D, &[i32])]) -> std::io::Result<Vec<CompressedSubBand>> {
    subbands
        .iter()
        .map(|&(sb, coeffs)| compress_subband(coeffs, sb))
        .collect()
}

/// Decompress all sub-bands from compressed data.
pub fn decompress_all_subbands(compressed: &[CompressedSubBand]) -> std::io::Result<Vec<(SubBand3D, Vec<i32>)>> {
    compressed
        .iter()
        .map(|csb| {
            let coeffs = decompress_subband(csb)?;
            Ok((csb.subband, coeffs))
        })
        .collect()
}

/// Estimate compression ratio for a sub-band based on its frequency tier.
///
/// Low-frequency sub-bands (LLL) contain smooth, correlated data that
/// compresses very well. High-frequency sub-bands (HHH) contain noise-like
/// data that barely compresses.
pub fn estimate_ratio(subband: SubBand3D, data_len: usize) -> f64 {
    let base_ratio = match subband {
        SubBand3D::LLL => 8.0,
        SubBand3D::LLH | SubBand3D::LHL | SubBand3D::HLL => 4.0,
        SubBand3D::LHH | SubBand3D::HLH | SubBand3D::HHL => 2.0,
        SubBand3D::HHH => 1.2,
    };
    // Larger data generally compresses better due to pattern repetition
    let size_factor = (data_len as f64 / 1024.0).ln().max(1.0);
    base_ratio * size_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Sine-like correlated data (typical for low-frequency sub-band)
        let data: Vec<i32> = (0..256).map(|i| ((i as f32 * 0.1).sin() * 1000.0) as i32).collect();

        let compressed = compress_subband(&data, SubBand3D::LLL).unwrap();
        assert_eq!(compressed.count, 256);
        assert_eq!(compressed.subband, SubBand3D::LLL);

        let decompressed = decompress_subband(&compressed).unwrap();
        assert_eq!(data.len(), decompressed.len());

        // 16-bit quantization for LLL: error should be small relative to range
        let max_err: i32 = data.iter().zip(decompressed.iter())
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        assert!(max_err < 10, "Max error too large for LLL: {}", max_err);
    }

    #[test]
    fn test_hhh_roundtrip() {
        // Noise-like data (typical for high-frequency sub-band)
        let data: Vec<i32> = (0..128).map(|i| if i % 2 == 0 { 5 } else { -5 }).collect();

        let compressed = compress_subband(&data, SubBand3D::HHH).unwrap();
        let decompressed = decompress_subband(&compressed).unwrap();
        assert_eq!(data.len(), decompressed.len());

        // 8-bit quantization for HHH: larger error tolerance
        for (a, b) in data.iter().zip(decompressed.iter()) {
            assert!((a - b).abs() < 2, "HHH roundtrip error too large: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_all_subbands() {
        let data: Vec<i32> = (0..64).map(|i| i * 10).collect();
        let subbands = vec![
            (SubBand3D::LLL, data.as_slice()),
            (SubBand3D::HHH, data.as_slice()),
        ];

        let compressed = compress_all_subbands(&subbands).unwrap();
        assert_eq!(compressed.len(), 2);
        assert_eq!(compressed[0].subband, SubBand3D::LLL);
        assert_eq!(compressed[1].subband, SubBand3D::HHH);

        let decompressed = decompress_all_subbands(&compressed).unwrap();
        assert_eq!(decompressed.len(), 2);
        assert_eq!(decompressed[0].0, SubBand3D::LLL);
        assert_eq!(decompressed[1].0, SubBand3D::HHH);
    }

    #[test]
    fn test_estimate_ratio() {
        let lll_ratio = estimate_ratio(SubBand3D::LLL, 4096);
        let hhh_ratio = estimate_ratio(SubBand3D::HHH, 4096);
        assert!(
            lll_ratio > hhh_ratio,
            "LLL ratio {} should exceed HHH ratio {}",
            lll_ratio, hhh_ratio
        );

        // Mid-freq should be between
        let mid_ratio = estimate_ratio(SubBand3D::LHH, 4096);
        assert!(lll_ratio > mid_ratio);
        assert!(mid_ratio > hhh_ratio);
    }

    #[test]
    fn test_select_preset() {
        // DC sub-band should get highest preset
        assert!(select_preset(SubBand3D::LLL) > select_preset(SubBand3D::HHH));
    }

    #[test]
    fn test_select_bits() {
        // DC sub-band should get 16-bit precision
        assert_eq!(select_bits(SubBand3D::LLL), 16);
        // High-freq should get 8-bit
        assert_eq!(select_bits(SubBand3D::HHH), 8);
    }

    #[test]
    fn test_empty_subband() {
        let data: Vec<i32> = Vec::new();
        let compressed = compress_subband(&data, SubBand3D::LLL).unwrap();
        let decompressed = decompress_subband(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }
}
