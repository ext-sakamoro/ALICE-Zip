//! Oracles for the causal laws of `alice_zip::int2d` (`CausalLaw2`, `Prev`,
//! `encode_causal` / `decode_causal`).
//!
//! The expected values come from integer arithmetic written here: the previous-value deltas
//! and their zigzag / LEB128 bytes, and a reference decoder that reads the documented archive
//! layout itself and rebuilds the values in closed loop. Every test asserts that it compared
//! at least one value.
#![cfg(feature = "std")]

use alice_zip::int2d::{
    decode, decode_causal, encode, encode_causal, CausalLaw2, Codec, Domain, Int2dError, Mode,
    Prev, Zero,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn signed(&mut self, bound: i64) -> i64 {
        let span = u64::try_from(bound).unwrap() * 2 + 1;
        i64::try_from(self.next() % span).unwrap() - bound
    }
}

fn masked_lattice(ni: u32, nj: u32) -> Domain {
    let present: Vec<bool> = (0..ni * nj).map(|k| k % 3 != 1).collect();
    Domain::lattice(100, -50, ni, nj, Some(&present)).unwrap()
}

fn points(n: usize, seed: u64) -> Domain {
    let mut rng = Lcg(seed);
    let mut pts = Vec::new();
    while pts.len() < n {
        let p = (rng.signed(100_000), rng.signed(100_000));
        if !pts.contains(&p) {
            pts.push(p);
        }
    }
    Domain::points(pts).unwrap()
}

/// a random walk: each value is the previous one plus a step in [-step, step]
fn walk(n: usize, start: i64, step: i64, seed: u64) -> Vec<i64> {
    let mut rng = Lcg(seed);
    let mut v = start;
    (0..n)
        .map(|_| {
            v += rng.signed(step);
            v
        })
        .collect()
}

/// zigzag + LEB128 of each value, written from the definitions
fn varints(values: impl IntoIterator<Item = i64>) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        let mut u = if v >= 0 {
            (v as u64) << 1
        } else {
            ((!(v as u64)) << 1) | 1
        };
        loop {
            let byte = (u & 0x7f) as u8;
            u >>= 7;
            if u == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }
    out
}

/// the previous-value deltas v[k] - v[k-1] with v[-1] = 0 (the plain delta baseline)
fn deltas(values: &[i64]) -> Vec<i64> {
    values
        .iter()
        .enumerate()
        .map(|(k, &v)| v - if k == 0 { 0 } else { values[k - 1] })
        .collect()
}

const HEADER: usize = 4 + 2 + 32 + 32 + 1 + 8 + 8 + 8 + 32 + 32 + 1 + 8;

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
}

/// A reference decoder for a `Prev` archive, written from the documented layout: magic,
/// version, law id, domain digest, mode, q, count, max error, 2 digests, codec, payload
/// length, payload, trailer. It rebuilds in closed loop: value k = value k-1 (0 for k = 0)
/// + q * quotient k.
fn reference_decode(bytes: &[u8]) -> Vec<i64> {
    use std::io::Read;
    assert_eq!(&bytes[..4], b"ALC2");
    let q = i64::try_from(u64_at(bytes, 71)).unwrap();
    let count = usize::try_from(u64_at(bytes, 79)).unwrap();
    let codec = bytes[159];
    let plen = usize::try_from(u64_at(bytes, 160)).unwrap();
    let payload = &bytes[HEADER..HEADER + plen];
    let raw = if codec == 1 {
        payload.to_vec()
    } else {
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(payload)
            .read_to_end(&mut out)
            .unwrap();
        out
    };
    let mut out = Vec::with_capacity(count);
    let mut at = 0;
    for k in 0..count {
        let mut u = 0_u64;
        let mut shift = 0;
        loop {
            let b = raw[at];
            at += 1;
            u |= u64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let quot = ((u >> 1) as i64) ^ -((u & 1) as i64);
        let prev = if k == 0 { 0 } else { out[k - 1] };
        out.push(prev + q * quot);
    }
    assert_eq!(at, raw.len(), "reference decoder: trailing bytes");
    out
}

/// Lossless `Prev` stores exactly the plain previous-value delta stream (before the backend),
/// and both the codec's decoder and the reference decoder return the values
#[test]
fn lossless_prev_is_the_previous_value_delta_stream() {
    let mut compared = 0;
    for (domain, seed) in [(masked_lattice(64, 48), 1), (points(1500, 2), 3)] {
        let values = walk(domain.len(), 1_000_000, 500, seed);
        let enc = encode_causal(&domain, &values, &Prev, Mode::Lossless, Codec::Raw).unwrap();
        assert_eq!(enc.max_abs_error, 0);
        assert_eq!(
            &enc.bytes[HEADER..enc.bytes.len() - 32],
            varints(deltas(&values)).as_slice()
        );
        assert_eq!(decode_causal(&enc.bytes, &domain, &Prev).unwrap(), values);
        assert_eq!(reference_decode(&enc.bytes), values);
        let deflated =
            encode_causal(&domain, &values, &Prev, Mode::Lossless, Codec::Deflate).unwrap();
        assert_eq!(
            decode_causal(&deflated.bytes, &domain, &Prev).unwrap(),
            values
        );
        assert_eq!(reference_decode(&deflated.bytes), values);
        compared += values.len();
    }
    assert!(compared > 0);
}

/// Bounded is closed loop: over 10^6 values every |value - decoded| <= floor(q / 2), the
/// reported bound is the measured maximum, and the reference decoder agrees value by value
#[test]
fn bounded_prev_keeps_the_bound_on_a_million_values() {
    let domain = Domain::lattice(0, 0, 1000, 1000, None).unwrap();
    let values = walk(domain.len(), 0, 2_000, 9);
    let mut compared = 0;
    for q in [2_u64, 7, 1_000] {
        let enc = encode_causal(&domain, &values, &Prev, Mode::Bounded { q }, Codec::Raw).unwrap();
        let back = decode_causal(&enc.bytes, &domain, &Prev).unwrap();
        assert_eq!(
            back,
            reference_decode(&enc.bytes),
            "q = {q}: decoders differ"
        );
        let bound = i64::try_from(q / 2).unwrap();
        let mut worst = 0;
        for (v, b) in values.iter().zip(&back) {
            let e = (v - b).abs();
            assert!(e <= bound, "q = {q}: |{v} - {b}| = {e} > {bound}");
            worst = worst.max(e);
            compared += 1;
        }
        assert_eq!(enc.max_abs_error, u64::try_from(worst).unwrap(), "q = {q}");
    }
    assert!(compared >= 3_000_000);
}

#[test]
fn round_trips_on_adversarial_values() {
    let domain = masked_lattice(40, 30);
    let n = domain.len();
    let big = 1_i64 << 40;
    let mut rng = Lcg(5);
    let inputs = [
        vec![big; n],
        (0..n)
            .map(|k| if k % 2 == 0 { big } else { -big })
            .collect::<Vec<_>>(),
        (0..n).map(|_| rng.signed(big)).collect(),
        vec![0; n],
    ];
    let mut compared = 0;
    for values in &inputs {
        for mode in [Mode::Lossless, Mode::Bounded { q: 3 }] {
            let enc = encode_causal(&domain, values, &Prev, mode, Codec::Deflate).unwrap();
            let back = decode_causal(&enc.bytes, &domain, &Prev).unwrap();
            assert_eq!(back, reference_decode(&enc.bytes));
            if mode == Mode::Lossless {
                assert_eq!(&back, values);
            }
            compared += back.len();
        }
    }
    assert!(compared > 0);
}

/// A law that predicts 2 everywhere (to check that the identifier is compared)
struct Two;

impl CausalLaw2 for Two {
    fn predict(&self, _k: usize, _cell: (i64, i64), _decoded: &[i64]) -> Result<i64, Int2dError> {
        Ok(2)
    }
    fn law_id(&self) -> [u8; 32] {
        [2; 32]
    }
}

#[test]
fn a_mismatched_or_damaged_archive_is_refused() {
    let domain = masked_lattice(12, 10);
    let values = walk(domain.len(), 50, 20, 4);
    let enc = encode_causal(&domain, &values, &Prev, Mode::Lossless, Codec::Deflate).unwrap();
    assert!(matches!(
        decode_causal(&enc.bytes, &domain, &Two),
        Err(Int2dError::LawIdMismatch)
    ));
    let other = masked_lattice(10, 12);
    assert_eq!(other.len(), domain.len());
    assert!(matches!(
        decode_causal(&enc.bytes, &other, &Prev),
        Err(Int2dError::DomainMismatch)
    ));
    // the two archive kinds do not decode each other
    assert!(matches!(
        decode(&enc.bytes, &domain, &Zero),
        Err(Int2dError::BadHeader("magic"))
    ));
    let plain = encode(&domain, &values, &Zero, Mode::Lossless).unwrap();
    assert!(matches!(
        decode_causal(&plain.bytes, &domain, &Prev),
        Err(Int2dError::BadHeader("magic"))
    ));
    let mut flipped = 0;
    for byte in 0..enc.bytes.len() {
        for bit in 0..8 {
            let mut b = enc.bytes.clone();
            b[byte] ^= 1 << bit;
            assert!(
                decode_causal(&b, &domain, &Prev).is_err(),
                "flip at {byte}:{bit} accepted"
            );
            flipped += 1;
        }
    }
    assert!(flipped > 0);
    assert!(decode_causal(&enc.bytes[..enc.bytes.len() - 1], &domain, &Prev).is_err());
}

#[test]
fn overflow_is_an_error_not_a_wrap() {
    let domain = Domain::lattice(0, 0, 3, 1, None).unwrap();
    // MIN - MAX does not fit in i64
    let values = [i64::MAX, i64::MIN, 0];
    assert!(matches!(
        encode_causal(&domain, &values, &Prev, Mode::Lossless, Codec::Raw),
        Err(Int2dError::Overflow)
    ));
    assert!(matches!(
        encode_causal(&domain, &[0, 0], &Prev, Mode::Lossless, Codec::Raw),
        Err(Int2dError::LengthMismatch { .. })
    ));
    assert!(matches!(
        encode_causal(
            &domain,
            &[0, 0, 0],
            &Prev,
            Mode::Bounded { q: 0 },
            Codec::Raw
        ),
        Err(Int2dError::InvalidParameter(_))
    ));
}

#[test]
fn prev_has_its_own_identifier() {
    assert_ne!(Prev.law_id(), [0; 32]);
    assert_ne!(Prev.law_id(), Two.law_id());
    // the first cell is predicted from nothing: 0
    assert_eq!(Prev.predict(0, (5, 5), &[]).unwrap(), 0);
    assert_eq!(Prev.predict(3, (5, 5), &[7, 8, 9]).unwrap(), 9);
}

/// The archive bytes for fixed inputs, recorded once and checked on the CI's 3 OS
#[test]
fn causal_archive_bytes_are_the_recorded_ones() {
    use sha2::{Digest, Sha256};
    let domain = masked_lattice(48, 32);
    let values = walk(domain.len(), 10_000, 300, 42);
    let mut got = Vec::new();
    for (name, mode) in [
        ("lossless", Mode::Lossless),
        ("q=7", Mode::Bounded { q: 7 }),
    ] {
        let enc = encode_causal(&domain, &values, &Prev, mode, Codec::Deflate).unwrap();
        let hex: String = Sha256::digest(&enc.bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        got.push(format!("prev {name} {} {hex}", enc.bytes.len()));
    }
    if std::env::var("INT2D_PRINT_GOLDEN").as_deref() == Ok("1") {
        for l in &got {
            println!("{l}");
        }
    }
    assert_eq!(got.len(), GOLDEN.len());
    for (g, w) in got.iter().zip(GOLDEN) {
        assert_eq!(g, w);
    }
}

const GOLDEN: [&str; 2] = [
    "prev lossless 1724 7392d3d51e291ddafe1940a3f244a24eaaf3d1450d75da7ff0a0e639693451a5",
    "prev q=7 1062 09ad8104cfcba2f0ae18e0e95b7e2edf793f93a1847f4ff6df085c129aa723e7",
];
