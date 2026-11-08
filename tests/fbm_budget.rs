//! The octave count is a loop bound, so it is bounded: points × octaves within `MAX_FBM_STEPS`,
//! and an octave whose frequency left the finite range is an error, not a `NaN`.

use alice_zip::generators::{
    generate_fbm_1d, generate_perlin_advanced, PerlinNoise, MAX_FBM_STEPS,
};
use alice_zip::ZipError;

#[test]
fn the_budget_is_the_stated_policy() {
    assert_eq!(MAX_FBM_STEPS, (1 << 30) / 24 * 64);
}

#[test]
fn a_call_over_the_budget_is_refused_before_it_runs() {
    // one point at u32::MAX octaves: the stored model e4 measured spinning for hours
    assert_eq!(
        generate_fbm_1d(1, 7, 1.0, u32::MAX, 0.5, 1.0),
        Err(ZipError::InvalidParameter)
    );
    let points = usize::try_from(MAX_FBM_STEPS / 64).unwrap();
    assert_eq!(
        generate_fbm_1d(points + 1, 7, 1.0, 64, 0.5, 1.0),
        Err(ZipError::InvalidParameter)
    );
    assert_eq!(
        generate_perlin_advanced(points + 1, 1, 7, 4.0, 64, 0.5, 1.0),
        Err(ZipError::InvalidParameter)
    );
    let noise = PerlinNoise::new(7);
    assert_eq!(
        noise.try_fbm(0.3, 0.4, u32::MAX, 0.5, 1.0),
        Err(ZipError::InvalidParameter)
    );
}

#[test]
fn the_budget_itself_is_accepted() {
    let budget_octaves = u32::try_from(MAX_FBM_STEPS).unwrap();
    // one point at the whole budget: lacunarity 1 keeps the frequency finite, persistence 0
    // makes every octave past the first add nothing, so the loop stops at once
    let one = generate_fbm_1d(1, 7, 1.0, budget_octaves, 0.0, 1.0).unwrap();
    assert_eq!(one, generate_fbm_1d(1, 7, 1.0, 1, 0.0, 1.0).unwrap());
}

#[test]
fn a_frequency_past_the_finite_range_is_an_error_not_nan() {
    // lacunarity 2 from scale 1 leaves f32 after 128 octaves
    assert_eq!(
        generate_fbm_1d(4, 7, 1.0, 200, 0.5, 2.0),
        Err(ZipError::InvalidParameter)
    );
    assert_eq!(
        generate_perlin_advanced(4, 4, 7, 4.0, 200, 0.5, 2.0),
        Err(ZipError::InvalidParameter)
    );
    let noise = PerlinNoise::new(7);
    assert_eq!(
        noise.try_fbm(0.3, 0.4, 200, 0.5, 2.0),
        Err(ZipError::InvalidParameter)
    );
    // the infallible form is documented to give 0.0 for what the fallible one refuses
    assert_eq!(
        noise.fbm(0.3, 0.4, 200, 0.5, 2.0).to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn an_amplitude_that_reached_zero_ends_the_sum() {
    // persistence 0.5 underflows to 0 after about 150 octaves; lacunarity 1 keeps the
    // frequency finite: every octave after the underflow adds nothing, and the result is the
    // number the octaves before it gave, not NaN
    let long = generate_fbm_1d(16, 7, 3.0, 100_000, 0.5, 1.0).unwrap();
    let short = generate_fbm_1d(16, 7, 3.0, 200, 0.5, 1.0).unwrap();
    assert!(long.iter().all(|v| v.is_finite()));
    assert_eq!(
        long.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        short.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
