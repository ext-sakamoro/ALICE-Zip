//! The stream gives the samples of `generate`, however the blocks are cut.

use alice_zip::media::{AudioGenerator, AudioParams, AudioPattern, AudioStream};

fn params(pattern: AudioPattern) -> AudioParams {
    AudioParams {
        pattern,
        sample_rate: 8000,
        duration_secs: 2.0,
        components: vec![(220.0, 0.5, 9000.0), (660.5, 0.25, 0.3)],
        seed: 7,
        ..AudioParams::default()
    }
}

const ALL: [AudioPattern; 4] = [
    AudioPattern::Silence,
    AudioPattern::Sine,
    AudioPattern::MultiSine,
    AudioPattern::WhiteNoise,
];

#[test]
fn any_cut_gives_the_samples_of_generate() {
    for pattern in ALL {
        let p = params(pattern);
        let whole = AudioGenerator::new().generate(&p);
        assert_eq!(whole.len(), 16000);
        for block in [1, 7, 4096, 16000, 20000] {
            let mut stream = AudioStream::new(&p, whole.len());
            let mut got = Vec::new();
            let mut buffer = vec![0.0_f32; block];
            loop {
                let n = stream.fill(&mut buffer);
                if n == 0 {
                    break;
                }
                got.extend(buffer[..n].iter().map(|s| s.to_bits()));
            }
            let want: Vec<u32> = whole.iter().map(|s| s.to_bits()).collect();
            assert_eq!(got, want, "{pattern:?} in blocks of {block}");
            assert_eq!(stream.remaining(), 0);
        }
    }
}

#[test]
fn the_seed_decides_the_noise() {
    let mut a = params(AudioPattern::WhiteNoise);
    let one = AudioGenerator::new().generate(&a);
    a.seed = 8;
    let two = AudioGenerator::new().generate(&a);
    assert_ne!(one, two);
}
