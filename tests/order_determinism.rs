//! The parallel 2D Perlin generation does not depend on the thread count
//! (ALICE-Law G13).
//!
//! oracle: the texture is a row-major function of (x, y) alone, so the output
//! computed in every pool must be the same bits as the serial composition of
//! `PerlinNoise::fbm` over the rows (the per-point formula of
//! `generate_perlin_advanced`, written out here); a combination that depended
//! on scheduling (an unordered collect, rows appended in completion order, a
//! reduction across rows) would differ from it.
//!
//! The tooth has to fire on every run, not on a lucky schedule: many short
//! rows (H rows of W points) give the scheduler thousands of chances to finish
//! rows out of order, and each pool (2 to 16 threads) is run several times.
#![cfg(feature = "parallel")]

use alice_zip::generators::{generate_perlin_advanced, PerlinNoise};

const W: usize = 4;
const H: usize = 4096;
const SEED: u64 = 0x5eed_1234;
const SCALE: f32 = 9.5;
const OCTAVES: u32 = 4;

fn in_pool(threads: usize) -> Vec<u32> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
        .install(|| generate_perlin_advanced(W, H, SEED, SCALE, OCTAVES, 0.5, 2.0))
        .expect("valid parameters")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn serial() -> Vec<u32> {
    let noise = PerlinNoise::new(SEED);
    let rcp = 1.0 / SCALE;
    (0..H)
        .flat_map(|y| {
            let noise = &noise;
            (0..W).map(move |x| {
                let v = (noise.fbm(x as f32 * rcp, y as f32 * rcp, OCTAVES, 0.5, 2.0) + 1.0) * 0.5;
                v.clamp(0.0, 1.0).to_bits()
            })
        })
        .collect()
}

#[test]
fn perlin_texture_is_the_same_bits_for_every_thread_count() {
    let expected = serial();
    assert_eq!(expected.len(), W * H);
    let mut runs = 0;
    for threads in [1usize, 2, 4, 8, 16] {
        for run in 0..4 {
            assert!(
                in_pool(threads) == expected,
                "{threads} threads (run {run}) differ from the serial composition"
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 20);
}
