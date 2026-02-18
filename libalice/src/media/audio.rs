//! Audio Generator
//!
//! Rust port of the Python AudioGenerator in media_generators.py.
//! Generates procedural PCM audio from parametric descriptions and can
//! write the result directly to a 32-bit IEEE-float WAV file.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use std::f32::consts::PI;
use std::io::{self, Write};
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Enums
// ============================================================================

/// Detected / requested audio patterns (mirrors Python AudioPattern enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPattern {
    Silence,
    Sine,
    MultiSine,
    WhiteNoise,
}

// ============================================================================
// Params
// ============================================================================

/// Parameters for procedural audio generation
///
/// Mirrors Python `AudioParams` dataclass.
#[derive(Debug, Clone)]
pub struct AudioParams {
    /// Which waveform to generate
    pub pattern: AudioPattern,
    /// Samples per second (e.g. 44100)
    pub sample_rate: u32,
    /// Duration of the output in seconds
    pub duration_secs: f32,
    /// Frequency in Hz — used for `Sine`
    pub frequency: f32,
    /// Peak amplitude for `Sine` and `WhiteNoise` (0.0–1.0 range)
    pub amplitude: f32,
    /// Components for `MultiSine`: Vec of (frequency_hz, amplitude, phase_rad)
    pub components: Vec<(f32, f32, f32)>,
    /// RNG seed for `WhiteNoise`
    pub seed: u64,
    /// ADSR attack time in seconds
    pub attack: f32,
    /// ADSR decay time in seconds
    pub decay: f32,
    /// ADSR sustain level (0.0–1.0)
    pub sustain: f32,
    /// ADSR release time in seconds
    pub release: f32,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            pattern: AudioPattern::Silence,
            sample_rate: 44100,
            duration_secs: 1.0,
            frequency: 440.0,
            amplitude: 1.0,
            components: Vec::new(),
            seed: 42,
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.2,
        }
    }
}

// ============================================================================
// Generator
// ============================================================================

/// Procedural audio generator (stateless)
pub struct AudioGenerator;

impl AudioGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self
    }

    /// Generate a complete PCM f32 buffer for the given parameters.
    ///
    /// Returns normalised samples in the range [-1.0, 1.0].
    pub fn generate(&self, params: &AudioParams) -> Vec<f32> {
        let n = (params.duration_secs * params.sample_rate as f32) as usize;
        let sr = params.sample_rate;

        let mut samples = match params.pattern {
            AudioPattern::Silence => Self::generate_silence(n),
            AudioPattern::Sine => {
                Self::generate_sine(n, params.frequency, params.amplitude, 0.0, sr)
            }
            AudioPattern::MultiSine => {
                Self::generate_multi_sine(n, &params.components, sr)
            }
            AudioPattern::WhiteNoise => {
                Self::generate_white_noise(n, params.amplitude, params.seed)
            }
        };

        // Apply ADSR envelope to everything except silence
        if params.pattern != AudioPattern::Silence {
            samples = Self::apply_adsr_envelope(
                samples,
                params.attack,
                params.decay,
                params.sustain,
                params.release,
                sr,
            );
        }

        samples
    }

    /// Write PCM audio as a 32-bit IEEE-float WAV file.
    ///
    /// No external crate is required — the 44-byte RIFF/WAV header is
    /// constructed manually and the f32 samples are serialised as
    /// little-endian bytes.
    pub fn generate_to_wav(
        &self,
        params: &AudioParams,
        path: &std::path::Path,
    ) -> io::Result<()> {
        let samples = self.generate(params);
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        write_wav_f32(&mut writer, &samples, params.sample_rate, 1)
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    /// Return `n` zero samples
    fn generate_silence(n: usize) -> Vec<f32> {
        vec![0.0_f32; n]
    }

    /// Generate a single sine wave.
    ///
    /// Pre-computes `omega = 2π * freq / sr` to avoid division inside the loop.
    fn generate_sine(n: usize, freq: f32, amp: f32, phase: f32, sr: u32) -> Vec<f32> {
        let omega = 2.0 * PI * freq / sr as f32; // radians per sample
        (0..n)
            .map(|i| amp * (omega * i as f32 + phase).sin())
            .collect()
    }

    /// Generate a sum of sine waves from `(freq_hz, amplitude, phase_rad)` components.
    fn generate_multi_sine(n: usize, components: &[(f32, f32, f32)], sr: u32) -> Vec<f32> {
        // Pre-compute omega for each component
        let omegas: Vec<(f32, f32, f32)> = components
            .iter()
            .map(|&(f, a, p)| (2.0 * PI * f / sr as f32, a, p))
            .collect();

        (0..n)
            .map(|i| {
                omegas
                    .iter()
                    .map(|&(omega, amp, phase)| amp * (omega * i as f32 + phase).sin())
                    .sum::<f32>()
            })
            .collect()
    }

    /// Generate white noise using ChaCha8 CSPRNG for reproducibility.
    fn generate_white_noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                // Uniform [-amp, amp]
                let v: f32 = rng.gen_range(-1.0_f32..=1.0_f32);
                v * amp
            })
            .collect()
    }

    /// Apply an ADSR volume envelope in-place and return the modified samples.
    ///
    /// The envelope is applied per-sample according to the global sample position,
    /// matching the Python `_apply_envelope_chunk` logic.
    fn apply_adsr_envelope(
        mut samples: Vec<f32>,
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        sr: u32,
    ) -> Vec<f32> {
        let n = samples.len();
        let sr_f = sr as f32;

        let attack_end = (attack * sr_f) as usize;
        let decay_end = attack_end + (decay * sr_f) as usize;
        let release_start = n.saturating_sub((release * sr_f) as usize);

        // Pre-compute reciprocals to avoid repeated division
        let rcp_attack = if attack_end > 0 { 1.0 / attack_end as f32 } else { 1.0 };
        let decay_len = (decay * sr_f) as usize;
        let rcp_decay = if decay_len > 0 { 1.0 / decay_len as f32 } else { 1.0 };
        let release_len = n - release_start;
        let rcp_release = if release_len > 0 { 1.0 / release_len as f32 } else { 1.0 };

        for (i, s) in samples.iter_mut().enumerate() {
            let env = if i < attack_end {
                // Attack: 0 → 1
                i as f32 * rcp_attack
            } else if i < decay_end {
                // Decay: 1 → sustain
                let decay_pos = (i - attack_end) as f32;
                1.0 - (1.0 - sustain) * decay_pos * rcp_decay
            } else if i < release_start {
                // Sustain: flat
                sustain
            } else {
                // Release: sustain → 0
                let release_pos = (i - release_start) as f32;
                sustain * (1.0 - release_pos * rcp_release)
            };
            *s *= env;
        }
        samples
    }
}

impl Default for AudioGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WAV writing (no external crate)
// ============================================================================

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
    let data_size = num_samples * num_channels as u32 * 4; // f32 = 4 bytes
    let file_size = 36 + data_size; // everything after the initial 8-byte RIFF header

    let byte_rate = sample_rate * num_channels as u32 * 4;
    let block_align = num_channels * 4;

    // RIFF chunk descriptor
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt sub-chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?;       // sub-chunk size
    writer.write_all(&3_u16.to_le_bytes())?;         // audio format: IEEE float
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&32_u16.to_le_bytes())?;        // bits per sample

    // data sub-chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // PCM samples
    for &s in samples {
        writer.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44100;

    fn default_params(pattern: AudioPattern) -> AudioParams {
        AudioParams {
            pattern,
            sample_rate: SR,
            duration_secs: 1.0,
            frequency: 440.0,
            amplitude: 1.0,
            components: vec![(440.0, 0.5, 0.0), (880.0, 0.3, 0.0)],
            seed: 42,
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.2,
        }
    }

    #[test]
    fn test_silence() {
        let gen = AudioGenerator::new();
        let samples = gen.generate(&default_params(AudioPattern::Silence));
        assert_eq!(samples.len(), SR as usize);
        assert!(samples.iter().all(|&s| s == 0.0), "silence must be all zeros");
    }

    #[test]
    fn test_sine_period() {
        // A 440 Hz sine at 44100 Hz has a period of 44100/440 ≈ 100.23 samples.
        // After one full period the sample should be very close to the start value.
        let gen = AudioGenerator::new();
        let params = AudioParams {
            pattern: AudioPattern::Sine,
            // Disable envelope so we measure raw periodicity
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.0,
            ..default_params(AudioPattern::Sine)
        };
        let samples = gen.generate(&params);
        assert_eq!(samples.len(), SR as usize);

        // Period in samples for exactly 441 Hz at 44100 Hz = 100 samples (integer)
        // Use 441 Hz so we get an exact integer period
        let exact_params = AudioParams {
            frequency: 441.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.0,
            ..default_params(AudioPattern::Sine)
        };
        let s2 = gen.generate(&exact_params);
        let period = 100_usize; // 44100 / 441
        for i in 0..50 {
            let diff = (s2[i] - s2[i + period]).abs();
            assert!(
                diff < 1e-5,
                "not periodic at sample {i}: {} vs {}",
                s2[i],
                s2[i + period]
            );
        }
    }

    #[test]
    fn test_multi_sine() {
        let gen = AudioGenerator::new();
        let params = AudioParams {
            pattern: AudioPattern::MultiSine,
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.0,
            ..default_params(AudioPattern::MultiSine)
        };
        let samples = gen.generate(&params);
        assert_eq!(samples.len(), SR as usize);

        // RMS should be non-zero
        let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(rms > 0.0, "multi-sine RMS must be > 0");
    }

    #[test]
    fn test_adsr_envelope() {
        // Generate a sine with a long attack so we can verify the envelope slope
        let gen = AudioGenerator::new();
        let sr = 1000_u32; // low SR for easy arithmetic
        let params = AudioParams {
            pattern: AudioPattern::Sine,
            sample_rate: sr,
            duration_secs: 2.0,
            frequency: 10.0,
            amplitude: 1.0,
            components: vec![],
            seed: 0,
            attack: 0.5,  // 500 samples
            decay: 0.3,   // 300 samples
            sustain: 0.6,
            release: 0.2, // 200 samples
        };
        let samples = gen.generate(&params);
        let n = samples.len(); // 2000

        // At sample 0 the envelope is 0 → output should be ~0
        // (the sine might be 0 at 0 too, so check absolute amplitude is small)
        assert!(
            samples[0].abs() < 0.01,
            "first sample should be near zero (attack start): {}",
            samples[0]
        );

        // At the end of attack (sample 499), envelope = ~1.0, so |sample| ≤ 1.0
        // In sustain region (samples 800..n-200), envelope should be exactly `sustain`
        // We verify by checking that the amplitude matches the sustain level
        let sustain_idx = 900_usize;
        // The sine value at that index (before envelope) would be some value in [-1,1]
        // After envelope it must be within sustain * [-1,1]
        assert!(
            samples[sustain_idx].abs() <= params.sustain + 1e-4,
            "sustain region amplitude exceeded: {}",
            samples[sustain_idx].abs()
        );

        // Last sample: envelope → 0, so amplitude ≈ 0
        assert!(
            samples[n - 1].abs() < 0.05,
            "last sample should be near zero (release end): {}",
            samples[n - 1].abs()
        );
    }

    #[test]
    fn test_wav_output() {
        use std::io::Read;

        let gen = AudioGenerator::new();
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
            ..default_params(AudioPattern::Sine)
        };

        // Write to a temp file
        let tmp_path = std::env::temp_dir().join("alice_test_audio.wav");
        gen.generate_to_wav(&params, &tmp_path).expect("WAV write failed");

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
