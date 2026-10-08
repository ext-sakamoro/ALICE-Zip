//! 法則は 1 つ — 配列生成器は単点評価器の `map` であり、精度は f64 の閉形式に対して測る
//!
//! # なぜこの oracle が要るか
//!
//! 復元則は 2 箇所で実装されていた この crate の配列生成器 (`generate_sine_wave` /
//! `generate_multi_sine` / `generate_from_coefficients`) と、消費者側が単点問い合わせ
//! (`query_point` / `query_range`) のために持っていた写しである 同一入力・整数 index・
//! `n = 1024` で突合すると **sine 788 / multi-sine 881 / Fourier 962 の sample が
//! 1024 中で不一致**だった (2026-10-08 実測)
//!
//! 食い違いの原因は 3 つ重なっていた
//!
//! | | 配列生成器 (旧) | 写し |
//! |---|---|---|
//! | 内部精度 | `f32` (`core::f32::consts::PI`、`i as f32`) | `f64` |
//! | DC の積み方 | `f32` で総和 → 最後に `+ dc_offset` | `dc_offset` から積む |
//! | 超越関数 | `sin` / `cos` (`f32`) | platform libm (`f64`) |
//!
//! `f64` の閉形式を真値として最大誤差を測ると **配列 (f32) 9.16e-7 / 写し (f64) 5.95e-8**
//! で写しの方が 15 倍正確だった ⇒ **f64 内部 + `sin64` / `cos64` を正典とし、配列生成器は
//! 単点評価器の `map` として定義する** (2 実装が構造的に同一になり、食い違いが起きない)
//!
//! # 期待値の出所
//!
//! 下の許容差はどれも **閉形式 (解析解) を `f64` で評価したもの**で、この crate の出力は
//! 1 度も期待値に入っていない 許容差 `1e-7` は f32 の刻み (1.0 付近で 1.19e-7) の近傍に
//! 置いてあり、**f32 内部で積む実装では原理的に満たせない** (実測 9.16e-7 = 8 倍超過)

use alice_zip::generators;

/// `f32` の刻み (1.0 付近) これを下回る誤差は f32 の表現限界に達している
const F32_STEP_NEAR_ONE: f64 = 1.192_092_9e-7;

/// 閉形式に対する許容差 f32 の刻み以下 = 「f32 として最良」を要求する
const TOL: f64 = F32_STEP_NEAR_ONE;

fn max_abs_err(got: &[f32], truth: &[f64]) -> f64 {
    assert_eq!(got.len(), truth.len(), "長さが違えば比較が空振りする");
    assert!(!got.is_empty(), "比較 0 件の oracle は何も証明しない");
    got.iter()
        .zip(truth)
        .map(|(g, t)| (f64::from(*g) - t).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "the expected value has to come from somewhere independent of the implementation under test: the platform libm is that reference here, and the tolerance (one f32 step, 1.19e-7) is nine orders of magnitude looser than any disagreement between libm implementations"
)]
fn sine_matches_the_closed_form_to_f32_resolution() {
    let n = 1024;
    let (f, a, phase, dc) = (3.0_f32, 1.0_f32, 0.4_f32, 0.25_f32);

    // oracle: A·sin(2πf·i/n + φ) + dc を f64 で評価
    let truth: Vec<f64> = (0..n)
        .map(|i| {
            let theta =
                2.0 * core::f64::consts::PI * f64::from(f) * i as f64 / n as f64 + f64::from(phase);
            f64::from(a) * theta.sin() + f64::from(dc)
        })
        .collect();

    let err = max_abs_err(&generators::generate_sine_wave(n, f, a, phase, dc), &truth);
    assert!(
        err <= TOL,
        "sine の最大誤差 {err:.3e} が許容差 {TOL:.3e} (f32 の刻み) を超えた \
         f32 内部で積むと原理的に満たせない"
    );
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "the expected value has to come from somewhere independent of the implementation under test: the platform libm is that reference here, and the tolerance (one f32 step, 1.19e-7) is nine orders of magnitude looser than any disagreement between libm implementations"
)]
fn multi_sine_matches_the_closed_form_to_f32_resolution() {
    let n = 1024;
    let comps = [
        (1.0_f32, 1.0_f32, 0.0_f32),
        (3.0, 0.5, 0.7),
        (7.0, 0.25, -1.2),
    ];
    let dc = 0.125_f32;

    let truth: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = f64::from(dc);
            for &(freq, amp, phase) in &comps {
                let theta = 2.0 * core::f64::consts::PI * f64::from(freq) * i as f64 / n as f64
                    + f64::from(phase);
                sum += f64::from(amp) * theta.sin();
            }
            sum
        })
        .collect();

    let err = max_abs_err(&generators::generate_multi_sine(n, &comps, dc), &truth);
    assert!(
        err <= TOL,
        "multi-sine の最大誤差 {err:.3e} が許容差 {TOL:.3e} を超えた"
    );
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "the expected value has to come from somewhere independent of the implementation under test: the platform libm is that reference here, and the tolerance (one f32 step, 1.19e-7) is nine orders of magnitude looser than any disagreement between libm implementations"
)]
fn fourier_reconstruction_matches_the_closed_form_to_f32_resolution() {
    let n = 256;
    // 係数を直接与える (analyze_signal を通すと期待値が実装由来になる)
    let coefs = [(5_usize, 128.0_f32, 0.3_f32), (17, 64.0, -0.9)];
    let dc = 0.3_f32;

    let truth: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = f64::from(dc);
            for &(k, mag, phase) in &coefs {
                let weight = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
                let theta =
                    2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64 + f64::from(phase);
                sum += weight * f64::from(mag) / n as f64 * theta.cos();
            }
            sum
        })
        .collect();

    let err = max_abs_err(
        &generators::generate_from_coefficients(n, &coefs, dc),
        &truth,
    );
    assert!(
        err <= TOL,
        "Fourier 復元の最大誤差 {err:.3e} が許容差 {TOL:.3e} を超えた"
    );
}

/// 配列生成器は単点評価器の `map` — 1 bit でも違えば法則が 2 つある
///
/// ⚠️ 許容差でなく **bit 一致**を要求する 許容差で比べると、ここが本来捕まえるべき
/// 「2 実装が少しずつ違う」状態を全部通してしまう (旧実装の 788 / 1024 の不一致は
/// どれも 1e-6 以下なので、1e-5 の許容差なら気付けなかった)
#[test]
fn array_generators_are_a_map_over_the_point_law() {
    let mut compared = 0_usize;

    for &n in &[1_usize, 2, 3, 7, 64, 257, 1024] {
        // sine (位相 / 周波数 / DC を既定値から外す)
        for &(f, a, ph, dc) in &[
            (3.0_f32, 1.0_f32, 0.4_f32, 0.25_f32),
            (0.5, -2.0, -1.7, 0.0),
            (11.0, 0.125, 2.9, -3.5),
        ] {
            let arr = generators::generate_sine_wave(n, f, a, ph, dc);
            assert_eq!(arr.len(), n, "配列長が n でなければ比較が空振りする");
            for (i, got) in arr.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let want = generators::sine_at(n, f, a, ph, dc, i as f64);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "sine n={n} i={i}: 配列 {got} と単点 {want} の bit が違う"
                );
                compared += 1;
            }
        }

        // multi-sine (成分数 1 / 3、符号と位相を散らす)
        for comps in [
            &[(2.0_f32, 1.0_f32, 0.0_f32)][..],
            &[(1.0, 1.0, 0.3), (5.0, -0.5, 1.1), (13.0, 0.25, -2.2)][..],
        ] {
            let arr = generators::generate_multi_sine(n, comps, 0.125);
            assert_eq!(arr.len(), n);
            for (i, got) in arr.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let want = generators::multi_sine_at(n, comps, 0.125, i as f64);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "multi-sine n={n} i={i}: {got} vs {want}"
                );
                compared += 1;
            }
        }

        // Fourier (DC / Nyquist / 範囲外 k を含める = weight 分岐と skip を通す)
        let coefs = [
            (0_usize, 10.0_f32, 0.0_f32),
            (1, 128.0, 0.3),
            (n / 2, 64.0, -0.9),
            (n + 5, 99.0, 1.0),
        ];
        let arr = generators::generate_from_coefficients(n, &coefs, 0.3);
        assert_eq!(arr.len(), n);
        for (i, got) in arr.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let want = generators::fourier_at(n, &coefs, 0.3, i as f64);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "Fourier n={n} i={i}: {got} vs {want}"
            );
            compared += 1;
        }

        // 多項式 (次数 0 / 1 / 3)
        for coeffs in [
            &[2.5_f64][..],
            &[1.0, -0.5][..],
            &[0.25, 1.5, -0.125, 0.0625][..],
        ] {
            let arr = generators::generate_polynomial(n, coeffs);
            assert_eq!(arr.len(), n);
            for (i, got) in arr.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let want = generators::polynomial_at(coeffs, i as f64);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "多項式 n={n} i={i}: {got} vs {want}"
                );
                compared += 1;
            }
        }
    }

    // ⚠️ 比較 0 件で通る経路を塞ぐ (n の列が空 / 配列が空でも assert は全部通る)
    assert!(
        compared >= 10_000,
        "比較 {compared} 件 — 想定 (1 万件超) を下回った oracle が空振りしている"
    );
}

/// 単点評価器は非整数位置でも法則どおり — 整数しか見ない oracle では
/// 「`i` を丸めてから評価する」実装を通してしまう
#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "the expected value has to come from somewhere independent of the implementation under test: the platform libm is that reference here, and the tolerance (one f32 step, 1.19e-7) is nine orders of magnitude looser than any disagreement between libm implementations"
)]
fn point_law_is_continuous_between_samples() {
    let n = 64;
    let (f, a, ph, dc) = (1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32);

    // 半サンプル位置の閉形式
    for i in 0..n {
        let x = i as f64 + 0.5;
        let got = generators::sine_at(n, f, a, ph, dc, x);
        let theta = 2.0 * core::f64::consts::PI * f64::from(f) * x / n as f64 + f64::from(ph);
        let want = f64::from(a) * theta.sin() + f64::from(dc);
        let err = (f64::from(got) - want).abs();
        assert!(err <= TOL, "sine_at(i={x}) の誤差 {err:.3e} > {TOL:.3e}");
    }

    // ⚠️ 丸めていないことを直接 pin する (i と i+0.5 が同じ値なら丸めている)
    let at_3 = generators::sine_at(n, f, a, ph, dc, 3.0);
    let at_3_5 = generators::sine_at(n, f, a, ph, dc, 3.5);
    assert_ne!(
        at_3.to_bits(),
        at_3_5.to_bits(),
        "sine_at が位置を丸めている (i=3 と i=3.5 が同値)"
    );
}
