//! The discrete Fourier transform behind `analyze_signal_fft`, in `f64`, the same bits on
//! every target
//!
//! A power-of-two length runs an iterative radix-2 transform; any other length runs
//! Bluestein's algorithm (the transform written as a convolution with the chirp
//! `e^{-iπk²/n}`, evaluated with radix-2 transforms of the next power of two `≥ 2n − 1`).
//!
//! Why not a library: an FFT library picks its butterflies per target (SIMD kernels on
//! x86_64 and aarch64 differ, and some negate fused multiply-add results), and takes its
//! twiddles and `hypot` / `atan2` from the platform's libm, so its spectrum has no
//! cross-target bit guarantee. Here every product and sum is a separate IEEE operation (no
//! fused multiply-add; Rust does not contract), the twiddles come from
//! `alice_det_math::{cos64, sin64}` with the angle reduced exactly in integers, and the
//! order of every operation is fixed by the length alone.
//!
//! oracle: `tests/fft_oracle.rs` compares every bin with a compensated direct DFT for
//! lengths 1 to 4096 including primes, under a bound stated in that file.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

use alice_det_math::sin_cos64;

/// `(re, im)`
type C = (f64, f64);

/// `X[k] = Σ_i x[i]·e^{-2πi·k·i/n}` for `k ∈ 0..n` of a real sequence
#[allow(clippy::cast_precision_loss)]
pub(crate) fn dft_real(x: &[f64]) -> Vec<C> {
    let mut buf: Vec<C> = x.iter().map(|&v| (v, 0.0)).collect();
    dft(&mut buf);
    buf
}

/// Forward transform in place
fn dft(buf: &mut [C]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2(buf, &twiddles(n), false);
    } else {
        bluestein(buf);
    }
}

/// `(cos θ, sin θ)` for `θ = π·num/den`, `num` reduced exactly into `0..2·den` first
/// (`sin_cos64` is bit-identical to `cos64` and `sin64` called separately)
#[allow(clippy::cast_precision_loss)]
fn unit(num: usize, den: usize) -> C {
    let theta = PI * (num % (2 * den)) as f64 / den as f64;
    let (s, c) = sin_cos64(theta);
    (c, s)
}

/// `(cos, sin)` of `2π·j/n` for `j < n/2`, the table of a radix-2 transform of length `n`
/// and of every shorter power of two (read with a stride)
fn twiddles(n: usize) -> Vec<C> {
    (0..n / 2).map(|j| unit(2 * j, n)).collect()
}

/// Iterative radix-2 transform of a power-of-two length, forward (`e^{-iθ}`) or inverse
/// (`e^{+iθ}`, not scaled), with the table of [`twiddles`] for this length
fn radix2(buf: &mut [C], table: &[C], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two() && table.len() == n / 2);
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            buf.swap(i, j);
        }
    }
    // the sign of the exponent is applied in the butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let (c, s) = table[j * stride];
                let (ar, ai) = buf[start + j];
                let (br, bi) = buf[start + j + half];
                // t = w·b with w = c ∓ i·s
                let (tr, ti) = if inverse {
                    (c * br - s * bi, c * bi + s * br)
                } else {
                    (c * br + s * bi, c * bi - s * br)
                };
                buf[start + j] = (ar + tr, ai + ti);
                buf[start + j + half] = (ar - tr, ai - ti);
            }
        }
        len *= 2;
    }
}

/// Bluestein: `X[k] = w̄[k]·Σ_j (x[j]·w̄[j])·w[k−j]` with `w[j] = e^{iπj²/n}`, the sum a
/// cyclic convolution of length `m` (a power of two `≥ 2n − 1`)
#[allow(clippy::cast_precision_loss)]
fn bluestein(buf: &mut [C]) {
    let n = buf.len();
    let m = (2 * n - 1).next_power_of_two();
    // w[j] = (cos, sin) of π·j²/n, j² reduced modulo 2n exactly
    let w: Vec<C> = (0..n)
        .map(|j| unit(((j % (2 * n)) * (j % (2 * n))) % (2 * n), n))
        .collect();
    // a[j] = x[j]·w̄[j]
    let mut a = vec![(0.0, 0.0); m];
    for (j, (&(xr, xi), &(c, s))) in buf.iter().zip(&w).enumerate() {
        a[j] = (xr * c + xi * s, xi * c - xr * s);
    }
    // b[j] = w[j] at j and at m − j
    let mut b = vec![(0.0, 0.0); m];
    b[0] = w[0];
    for j in 1..n {
        b[j] = w[j];
        b[m - j] = w[j];
    }
    let table = twiddles(m);
    radix2(&mut a, &table, false);
    radix2(&mut b, &table, false);
    for (p, &(br, bi)) in a.iter_mut().zip(&b) {
        let (ar, ai) = *p;
        *p = (ar * br - ai * bi, ar * bi + ai * br);
    }
    radix2(&mut a, &table, true);
    // 1/m is a power of two: the scaling is exact
    let scale = 1.0 / m as f64;
    for (k, (out, &(c, s))) in buf.iter_mut().zip(&w).enumerate() {
        let (vr, vi) = (a[k].0 * scale, a[k].1 * scale);
        // X[k] = v·w̄[k]
        *out = (vr * c + vi * s, vi * c - vr * s);
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    /// Direct DFT with the angle reduced in integers and compensated sums
    fn direct(x: &[f64]) -> Vec<C> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = ((0.0, 0.0), (0.0, 0.0));
                for (i, &v) in x.iter().enumerate() {
                    let (c, s) = unit(2 * k * i, n);
                    add(&mut re, v * c);
                    add(&mut im, -(v * s));
                }
                (re.0 + re.1, im.0 + im.1)
            })
            .collect()
    }

    fn add(acc: &mut (f64, f64), v: f64) {
        let t = acc.0 + v;
        if acc.0.abs() >= v.abs() {
            acc.1 += (acc.0 - t) + v;
        } else {
            acc.1 += (v - t) + acc.0;
        }
        acc.0 = t;
    }

    /// Every bin, the DC and the negative frequencies included, within
    /// `16·log2(m)·ε64·Σ|x|` of the direct sum (the bound of `tests/fft_oracle.rs`)
    #[test]
    fn every_bin_matches_the_direct_sum() {
        let mut s = 7u64;
        for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 31, 97, 128, 1000] {
            let x: Vec<f64> = (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect();
            let got = dft_real(&x);
            let want = direct(&x);
            let m = if n.is_power_of_two() {
                n
            } else {
                (2 * n - 1).next_power_of_two()
            };
            let total: f64 = x.iter().map(|v| v.abs()).sum();
            let bound =
                16.0 * f64::from(if m > 1 { m.trailing_zeros() } else { 1 }) * f64::EPSILON * total;
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                let (dr, di) = ((g.0 - w.0).abs(), (g.1 - w.1).abs());
                let d = if dr >= di { dr } else { di };
                assert!(
                    d <= bound,
                    "n={n} k={k}: {g:?} vs {w:?} ({d:e} > {bound:e})"
                );
            }
        }
    }

    #[test]
    fn empty_and_single_values_are_their_own_transform() {
        assert!(dft_real(&[]).is_empty());
        assert_eq!(dft_real(&[-0.75]), vec![(-0.75, 0.0)]);
    }

    /// A unit impulse at 1 has the spectrum e^{-2πik/n}: on the unit circle in every bin
    #[test]
    fn a_shifted_impulse_has_unit_magnitude_in_every_bin() {
        for n in [5usize, 8, 12, 97] {
            let mut x = vec![0.0; n];
            x[1] = 1.0;
            for (k, (re, im)) in dft_real(&x).into_iter().enumerate() {
                let (c, s) = unit(2 * k, n);
                assert!(
                    (re - c).abs() < 1e-13 && (im + s).abs() < 1e-13,
                    "n={n} k={k}"
                );
            }
        }
    }
}
