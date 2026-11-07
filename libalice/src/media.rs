//! The media generators of the core crate, and writing audio to a file
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

pub use alice_zip::media::*;

use std::io::{self, Write};

/// Write the samples of `params` as a 32-bit IEEE-float WAV file.
///
/// The 44-byte RIFF/WAV header is constructed here and the f32 samples are serialised as little-endian bytes.
///
/// # Errors
///
/// What creating or writing the file returns
pub fn generate_to_wav(params: &AudioParams, path: &std::path::Path) -> io::Result<()> {
    let samples = AudioGenerator::new().generate(params);
    let file = std::fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    write_wav_f32(&mut writer, &samples, params.sample_rate, 1)
}

/// Write a minimal RIFF/WAV file with 32-bit IEEE-float PCM samples.
///
/// Header layout (44 bytes):
///
/// ```text
/// Offset  Size  Value
///  0       4    "RIFF"
///  4       4    file size - 8  (little-endian u32)
///  8       4    "WAVE"
/// 12       4    "fmt "
/// 16       4    16              (chunk size)
/// 20       2    3               (IEEE float PCM)
/// 22       2    num_channels
/// 24       4    sample_rate
/// 28       4    byte_rate       = sample_rate * num_channels * 4
/// 32       2    block_align     = num_channels * 4
/// 34       2    32              (bits per sample)
/// 36       4    "data"
/// 40       4    data size       = num_samples * num_channels * 4
/// 44+      …    f32 samples (little-endian)
/// ```
fn write_wav_f32<W: Write>(
    writer: &mut W,
    samples: &[f32],
    sample_rate: u32,
    num_channels: u16,
) -> io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * u32::from(num_channels) * 4; // f32 = 4 bytes
    let file_size = 36 + data_size; // everything after the initial 8-byte RIFF header

    let byte_rate = sample_rate * u32::from(num_channels) * 4;
    let block_align = num_channels * 4;

    // RIFF chunk descriptor
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt sub-chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?; // sub-chunk size
    writer.write_all(&3_u16.to_le_bytes())?; // audio format: IEEE float
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&32_u16.to_le_bytes())?; // bits per sample

    // data sub-chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // PCM samples
    for &s in samples {
        writer.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_output() {
        use std::io::Read;

        let params = AudioParams {
            pattern: AudioPattern::Sine,
            sample_rate: 44100,
            duration_secs: 0.1,
            frequency: 440.0,
            amplitude: 1.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.0,
            ..AudioParams::default()
        };

        // Write to a temp file
        let tmp_path = std::env::temp_dir().join("alice_test_audio.wav");
        generate_to_wav(&params, &tmp_path).expect("WAV write failed");

        // Read back and inspect the 44-byte header
        let mut file = std::fs::File::open(&tmp_path).expect("open failed");
        let mut header = [0u8; 44];
        file.read_exact(&mut header).expect("header read failed");

        // "RIFF" magic
        assert_eq!(&header[0..4], b"RIFF", "missing RIFF magic");
        // "WAVE" identifier
        assert_eq!(&header[8..12], b"WAVE", "missing WAVE identifier");
        // "fmt " sub-chunk
        assert_eq!(&header[12..16], b"fmt ", "missing fmt  chunk");
        // Audio format = 3 (IEEE float)
        let audio_fmt = u16::from_le_bytes([header[20], header[21]]);
        assert_eq!(audio_fmt, 3, "audio format must be 3 (IEEE float)");
        // Channels = 1
        let channels = u16::from_le_bytes([header[22], header[23]]);
        assert_eq!(channels, 1, "must be mono");
        // Sample rate
        let sr = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
        assert_eq!(sr, 44100, "sample rate mismatch");
        // Bits per sample = 32
        let bps = u16::from_le_bytes([header[34], header[35]]);
        assert_eq!(bps, 32, "must be 32-bit");
        // "data" sub-chunk
        assert_eq!(&header[36..40], b"data", "missing data chunk");

        // Clean up
        let _ = std::fs::remove_file(&tmp_path);
    }
}
