//! The parallel 2D Perlin generation does not depend on the thread count
//! (ALICE-Law G13).
//!
//! oracle: the texture is a row-major function of (x, y) alone, so the output
//! computed with 1 thread, with 4 and with 7 must be the same bits, and must
//! equal the serial composition of `PerlinNoise::fbm` over the rows (the
//! per-point formula of `generate_perlin_advanced`, written out here); a
//! combination that depended on scheduling (an unordered collect, a reduction
//! across rows) would differ between the pools.
#![cfg(feature = "parallel")]

use alice_zip::generators::{generate_perlin_advanced, PerlinNoise};

const W: usize = 37;
const H: usize = 61;
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
    let one = in_pool(1);
    assert_eq!(one.len(), W * H);
    assert_eq!(in_pool(4), one, "4 threads differ from 1");
    assert_eq!(in_pool(7), one, "7 threads differ from 1");
    assert_eq!(
        serial(),
        one,
        "the parallel texture differs from the serial composition"
    );
}
