//! Audio Generator
//!
//! Generates procedural PCM audio from parametric descriptions. The Python package calls this through the
//! extension (`alice_zip.native_accelerator.media_audio_stream`).
//!
//! License: LicenseRef-Proprietary
//! Author: Moroya Sakamoto

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Enums
// ============================================================================

/// Detected / requested audio patterns (mirrors Python `AudioPattern` enum)
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
    /// Components for `MultiSine`: Vec of (`frequency_hz`, amplitude, `phase_rad`)
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
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate a complete PCM f32 buffer for the given parameters.
    ///
    /// Returns normalised samples in the range [-1.0, 1.0].
    #[must_use]
    pub fn generate(&self, params: &AudioParams) -> Vec<f32> {
        let n = (params.duration_secs * params.sample_rate as f32) as usize;
        let mut samples = vec![0.0_f32; n];
        AudioStream::new(params, n).fill(&mut samples);
        samples
    }
}

/// The samples of [`AudioGenerator::generate`] in order, a block at a time: how the blocks are cut does not
/// change them
#[derive(Debug, Clone)]
pub struct AudioStream {
    pattern: AudioPattern,
    /// `(radians per sample, amplitude, phase)` of each sine
    sines: Vec<(f32, f32, f32)>,
    amplitude: f32,
    rng: ChaCha8Rng,
    position: usize,
    n: usize,
    attack_end: usize,
    decay_end: usize,
    release_start: usize,
    rcp_attack: f32,
    rcp_decay: f32,
    rcp_release: f32,
    sustain: f32,
}

impl AudioStream {
    /// The stream of `n` samples of `params` (`params.duration_secs` is not read)
    #[must_use]
    pub fn new(params: &AudioParams, n: usize) -> Self {
        let sr = params.sample_rate as f32;
        let rcp_sr = 1.0 / sr;
        let sines = match params.pattern {
            AudioPattern::Sine => {
                vec![(2.0 * PI * params.frequency * rcp_sr, params.amplitude, 0.0)]
            }
            _ => params
                .components
                .iter()
                .map(|&(f, a, p)| (2.0 * PI * f * rcp_sr, a, p))
                .collect(),
        };
        let attack_end = (params.attack * sr) as usize;
        let decay_len = (params.decay * sr) as usize;
        let decay_end = attack_end + decay_len;
        let release_start = n.saturating_sub((params.release * sr) as usize);
        let release_len = n - release_start;
        let reciprocal = |len: usize| if len > 0 { 1.0 / len as f32 } else { 1.0 };
        Self {
            pattern: params.pattern,
            sines,
            amplitude: params.amplitude,
            rng: ChaCha8Rng::seed_from_u64(params.seed),
            position: 0,
            n,
            attack_end,
            decay_end,
            release_start,
            rcp_attack: reciprocal(attack_end),
            rcp_decay: reciprocal(decay_len),
            rcp_release: reciprocal(release_len),
            sustain: params.sustain,
        }
    }

    /// The samples not yet given
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.n - self.position
    }

    /// Writes the next samples to the start of `out`, and returns how many: `out.len()`, or fewer at the end
    pub fn fill(&mut self, out: &mut [f32]) -> usize {
        let count = out.len().min(self.remaining());
        for slot in &mut out[..count] {
            *slot = self.next_sample();
        }
        count
    }

    fn next_sample(&mut self) -> f32 {
        let i = self.position;
        self.position += 1;
        let raw = match self.pattern {
            AudioPattern::Silence => return 0.0,
            AudioPattern::Sine => {
                let (omega, amp, phase) = self.sines[0];
                amp * alice_det_math::sin(omega * i as f32 + phase)
            }
            AudioPattern::MultiSine => self
                .sines
                .iter()
                .map(|&(omega, amp, phase)| amp * alice_det_math::sin(omega * i as f32 + phase))
                .sum::<f32>(),
            AudioPattern::WhiteNoise => {
                let v: f32 = self.rng.gen_range(-1.0_f32..=1.0_f32);
                v * self.amplitude
            }
        };
        raw * self.envelope(i)
    }

    /// The ADSR envelope at a sample
    fn envelope(&self, i: usize) -> f32 {
        if i < self.attack_end {
            i as f32 * self.rcp_attack
        } else if i < self.decay_end {
            let decay_pos = (i - self.attack_end) as f32;
            1.0 - (1.0 - self.sustain) * decay_pos * self.rcp_decay
        } else if i < self.release_start {
            self.sustain
        } else {
            let release_pos = (i - self.release_start) as f32;
            self.sustain * (1.0 - release_pos * self.rcp_release)
        }
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
        assert!(
            samples.iter().all(|&s| s == 0.0),
            "silence must be all zeros"
        );
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
            attack: 0.5, // 500 samples
            decay: 0.3,  // 300 samples
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
}
