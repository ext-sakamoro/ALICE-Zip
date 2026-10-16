//! Oracles for `alice_zip::int2d`, the integer 2-D residual codec.
//!
//! Every expected value here comes from integer arithmetic done in the test itself (the
//! input values, a plane written as a formula, `q / 2` by integer division), never from the
//! codec's own output. Every test also asserts that it compared at least one value, so a
//! case that silently compares nothing fails.
#![cfg(feature = "std")]

use alice_zip::int2d::{
    decode, encode, encode_with, residuals, Codec, Domain, Int2dError, IntLaw2, Mode, PlanePerTile,
    Zero, MAX_DENOMINATOR_BITS as MAX_K,
};

/// A deterministic sequence (64-bit LCG, Knuth's constants) for the test inputs
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// a value in [-bound, bound]
    fn signed(&mut self, bound: i64) -> i64 {
        let span = u64::try_from(bound).unwrap() * 2 + 1;
        i64::try_from(self.next() % span).unwrap() - bound
    }
}

fn full_lattice(ni: u32, nj: u32) -> Domain {
    Domain::lattice(-7, 11, ni, nj, None).expect("lattice")
}

fn masked_lattice(ni: u32, nj: u32) -> Domain {
    let present: Vec<bool> = (0..ni * nj).map(|k| k % 3 != 1).collect();
    Domain::lattice(100, -50, ni, nj, Some(&present)).expect("masked lattice")
}

fn points() -> Domain {
    let mut rng = Lcg(7);
    let mut pts = Vec::new();
    while pts.len() < 300 {
        let p = (rng.signed(1_000), rng.signed(1_000));
        if !pts.contains(&p) {
            pts.push(p);
        }
    }
    Domain::points(pts).expect("points")
}

/// The cells of a domain in the codec's order, computed here from the definition:
/// a lattice row by row (j outer, i inner) over the present cells, points sorted by (i, j)
fn cells(domain: &Domain) -> Vec<(i64, i64)> {
    domain.cells().collect()
}

fn values_for(domain: &Domain, mut f: impl FnMut(usize, i64, i64) -> i64) -> Vec<i64> {
    cells(domain)
        .into_iter()
        .enumerate()
        .map(|(k, (i, j))| f(k, i, j))
        .collect()
}

fn round_trip(domain: &Domain, values: &[i64], law: &dyn IntLaw2) -> usize {
    let enc = encode(domain, values, law, Mode::Lossless).expect("encode");
    assert_eq!(enc.max_abs_error, 0, "lossless reports an error bound");
    let back = decode(&enc.bytes, domain, law).expect("decode");
    assert_eq!(back, values, "lossless round trip changed the values");
    values.len()
}

#[test]
fn lossless_round_trip_on_every_domain_kind_and_law() {
    let mut compared = 0;
    for domain in [full_lattice(37, 23), masked_lattice(40, 30), points()] {
        let mut rng = Lcg(11);
        let smooth = values_for(&domain, |_, i, j| 1_000 * i - 300 * j + rng.signed(5));
        let zero = Zero;
        compared += round_trip(&domain, &smooth, &zero);
        for (tile, k) in [(8, 0), (16, 8), (64, 16)] {
            let law = PlanePerTile::fit(&domain, &smooth, tile, k).expect("fit");
            compared += round_trip(&domain, &smooth, &law);
        }
    }
    assert!(compared > 0, "compared nothing");
}

#[test]
fn lossless_round_trip_on_adversarial_values() {
    let domain = full_lattice(64, 64);
    let big = 1_i64 << 40;
    let mut rng = Lcg(3);
    let inputs = [
        values_for(&domain, |_, _, _| big),
        values_for(&domain, |_, _, _| -big),
        values_for(&domain, |k, _, _| if k % 2 == 0 { big } else { -big }),
        values_for(&domain, |_, _, _| rng.signed(big)),
        values_for(&domain, |k, _, _| {
            i64::try_from(k).unwrap() * 997 - 2_000_000
        }),
    ];
    let mut compared = 0;
    for values in &inputs {
        compared += round_trip(&domain, values, &Zero);
        let law = PlanePerTile::fit(&domain, values, 16, 8).expect("fit");
        compared += round_trip(&domain, values, &law);
    }
    assert!(compared > 0, "compared nothing");
}

#[test]
fn bounded_error_is_at_most_half_the_step() {
    let domain = masked_lattice(50, 40);
    let mut rng = Lcg(5);
    let values = values_for(&domain, |_, i, j| 40 * i + 7 * j + rng.signed(10_000));
    let law = PlanePerTile::fit(&domain, &values, 16, 8).expect("fit");
    let mut compared = 0;
    for q in [1_u64, 2, 3, 1_000] {
        let bound = i64::try_from(q / 2).unwrap();
        let enc = encode(&domain, &values, &law, Mode::Bounded { q }).expect("encode");
        let back = decode(&enc.bytes, &domain, &law).expect("decode");
        assert_eq!(back.len(), values.len());
        let mut worst = 0;
        for (v, b) in values.iter().zip(&back) {
            let e = (v - b).abs();
            assert!(e <= bound, "q = {q}: |{v} - {b}| = {e} > {bound}");
            worst = worst.max(e);
            compared += 1;
        }
        assert_eq!(
            enc.max_abs_error,
            u64::try_from(worst).unwrap(),
            "q = {q}: reported bound"
        );
        if q == 1 {
            assert_eq!(back, values, "q = 1 must be lossless");
        }
    }
    assert!(compared > 0, "compared nothing");
}

#[test]
fn a_plane_is_predicted_exactly() {
    // the scattered points get tiles large enough to hold at least 3 points not on a line;
    // a tile with fewer is fitted by a line or a constant and cannot reproduce a plane
    for (domain, tile) in [
        (full_lattice(64, 48), 16),
        (masked_lattice(33, 65), 16),
        (points(), 1024),
    ] {
        let values = values_for(&domain, |_, i, j| 3 * i - 5 * j + 7);
        let law = PlanePerTile::fit(&domain, &values, tile, 0).expect("fit");
        let r = residuals(&domain, &values, &law).expect("residuals");
        assert!(!r.is_empty(), "compared nothing");
        assert!(
            r.iter().all(|&x| x == 0),
            "a plane left a residual: {:?}",
            &r[..8]
        );
        // the same plane with coefficients in multiples of 2^-8 is still exact
        let law = PlanePerTile::fit(&domain, &values, tile, 8).expect("fit");
        let r = residuals(&domain, &values, &law).expect("residuals");
        assert!(r.iter().all(|&x| x == 0), "k = 8 left a residual");
        // a plane with rational slopes, rounded to integers: v = floor((2i + j + 2) / 4) is
        // within 1/2 of the plane (2i + j) / 4, so the fitted plane leaves residuals of at
        // most 1 in magnitude
        let values = values_for(&domain, |_, i, j| (2 * i + j + 2).div_euclid(4));
        let law = PlanePerTile::fit(&domain, &values, tile, 8).expect("fit");
        let r = residuals(&domain, &values, &law).expect("residuals");
        let worst = r.iter().map(|x| x.abs()).max().unwrap();
        assert!(worst <= 1, "rounded plane left a residual of {worst}");
    }
}

#[test]
fn the_zero_law_leaves_the_values_as_residuals() {
    let domain = masked_lattice(20, 20);
    let values = values_for(&domain, |k, _, _| i64::try_from(k).unwrap() - 100);
    let r = residuals(&domain, &values, &Zero).expect("residuals");
    assert!(!r.is_empty());
    assert_eq!(r, values);
}

#[test]
fn cells_follow_the_documented_order() {
    let present = [true, false, true, true, true, false];
    let d = Domain::lattice(10, 20, 3, 2, Some(&present)).unwrap();
    assert_eq!(cells(&d), vec![(10, 20), (12, 20), (10, 21), (11, 21)]);
    let d = Domain::points(vec![(5, 1), (-3, 9), (5, -2)]).unwrap();
    assert_eq!(cells(&d), vec![(-3, 9), (5, -2), (5, 1)]);
    assert_eq!(d.len(), 3);
}

#[test]
fn invalid_input_is_refused() {
    assert!(matches!(
        Domain::points(vec![]),
        Err(Int2dError::EmptyDomain)
    ));
    assert!(matches!(
        Domain::lattice(0, 0, 0, 5, None),
        Err(Int2dError::EmptyDomain)
    ));
    let none = [false; 6];
    assert!(matches!(
        Domain::lattice(0, 0, 3, 2, Some(&none)),
        Err(Int2dError::EmptyDomain)
    ));
    assert!(matches!(
        Domain::lattice(0, 0, 3, 2, Some(&[true; 5])),
        Err(Int2dError::InvalidParameter(_))
    ));
    assert!(matches!(
        Domain::points(vec![(1, 1), (1, 1)]),
        Err(Int2dError::InvalidParameter(_))
    ));
    let domain = full_lattice(4, 4);
    let values = vec![0_i64; 15];
    assert!(matches!(
        encode(&domain, &values, &Zero, Mode::Lossless),
        Err(Int2dError::LengthMismatch { .. })
    ));
    assert!(matches!(
        encode(&domain, &[0; 16], &Zero, Mode::Bounded { q: 0 }),
        Err(Int2dError::InvalidParameter(_))
    ));
    assert!(matches!(
        PlanePerTile::fit(&domain, &[0; 16], 0, 4),
        Err(Int2dError::InvalidParameter(_))
    ));
    assert!(
        PlanePerTile::fit(&domain, &[0; 16], u32::MAX, 4).is_ok(),
        "a large tile is allowed"
    );
    assert!(matches!(
        PlanePerTile::fit(&domain, &[0; 16], 8, 63),
        Err(Int2dError::InvalidParameter(_))
    ));
}

/// A law that predicts 1 everywhere: with the value `i64::MIN` the residual `MIN - 1` does
/// not fit in `i64`
struct One;

impl IntLaw2 for One {
    fn predict(&self, _i: i64, _j: i64) -> Result<i64, Int2dError> {
        Ok(1)
    }
    fn law_id(&self) -> [u8; 32] {
        [1; 32]
    }
}

#[test]
fn overflow_is_an_error_not_a_wrap() {
    let domain = full_lattice(4, 4);
    let mut values = vec![0_i64; 16];
    values[5] = i64::MIN;
    assert!(matches!(
        residuals(&domain, &values, &One),
        Err(Int2dError::Overflow)
    ));
    assert!(matches!(
        encode(&domain, &values, &One, Mode::Lossless),
        Err(Int2dError::Overflow)
    ));
    // values at both ends of i64 with the largest denominator: a sum, a determinant or a
    // coefficient does not fit, and the fit says so instead of wrapping
    values[5] = 0;
    values[15] = i64::MAX;
    values[0] = i64::MIN;
    assert!(matches!(
        PlanePerTile::fit(&domain, &values, 4, MAX_K).and_then(|law| encode(
            &domain,
            &values,
            &law,
            Mode::Lossless
        )),
        Err(Int2dError::Overflow)
    ));
}

#[test]
fn a_mismatched_or_damaged_archive_is_refused() {
    let domain = full_lattice(16, 16);
    let mut rng = Lcg(9);
    let values = values_for(&domain, |_, i, j| 5 * i + j + rng.signed(3));
    let law = PlanePerTile::fit(&domain, &values, 8, 4).expect("fit");
    let enc = encode(&domain, &values, &law, Mode::Lossless).expect("encode");

    // another law
    let other = PlanePerTile::fit(&domain, &values, 4, 4).expect("fit");
    assert_ne!(other.law_id(), law.law_id());
    assert!(matches!(
        decode(&enc.bytes, &domain, &other),
        Err(Int2dError::LawIdMismatch)
    ));
    assert!(matches!(
        decode(&enc.bytes, &domain, &Zero),
        Err(Int2dError::LawIdMismatch)
    ));
    // another domain with the same number of cells
    let shifted = Domain::lattice(-6, 11, 16, 16, None).unwrap();
    assert!(matches!(
        decode(&enc.bytes, &shifted, &law),
        Err(Int2dError::DomainMismatch)
    ));
    // truncated, bad magic, every single bit flipped in turn
    assert!(decode(&enc.bytes[..enc.bytes.len() - 1], &domain, &law).is_err());
    assert!(decode(&enc.bytes[..10], &domain, &law).is_err());
    let mut bad = enc.bytes.clone();
    bad[0] ^= 1;
    assert!(matches!(
        decode(&bad, &domain, &law),
        Err(Int2dError::BadHeader(_))
    ));
    let mut flipped = 0;
    for byte in 0..enc.bytes.len() {
        for bit in 0..8 {
            let mut b = enc.bytes.clone();
            b[byte] ^= 1 << bit;
            assert!(
                decode(&b, &domain, &law).map_or(true, |v| v != values),
                "a flip at byte {byte} bit {bit} decoded to the original"
            );
            assert!(
                decode(&b, &domain, &law).is_err(),
                "flip at {byte}:{bit} was accepted"
            );
            flipped += 1;
        }
    }
    assert!(flipped > 0);
}

#[test]
fn law_ids_name_the_law_and_its_parameters() {
    let domain = full_lattice(16, 16);
    let values = values_for(&domain, |_, i, j| i * j);
    let a = PlanePerTile::fit(&domain, &values, 8, 4).unwrap();
    let b = PlanePerTile::fit(&domain, &values, 8, 4).unwrap();
    assert_eq!(
        a.law_id(),
        b.law_id(),
        "the same fit gives the same identifier"
    );
    let c = PlanePerTile::fit(&domain, &values, 8, 5).unwrap();
    assert_ne!(
        a.law_id(),
        c.law_id(),
        "the denominator is part of the identifier"
    );
    let mut changed = values.clone();
    changed[0] += 1_000;
    let d = PlanePerTile::fit(&domain, &changed, 8, 4).unwrap();
    assert_ne!(
        a.law_id(),
        d.law_id(),
        "the coefficients are part of the identifier"
    );
    assert_ne!(Zero.law_id(), a.law_id());
}

#[test]
fn the_module_has_no_floating_point() {
    let src = include_str!("../src/int2d.rs");
    let words: Vec<&str> = src
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .collect();
    assert!(words.len() > 100, "read too little of the module");
    for banned in ["f32", "f64"] {
        assert!(!words.contains(&banned), "src/int2d.rs uses {banned}");
    }
}

/// The archive bytes for fixed inputs, recorded once: the codec is integer-only, so every
/// target must produce these exact bytes (CI runs this on x86_64 Linux / Windows and arm64
/// macOS). The digests also pin the deflate backend in Cargo.lock (flate2 / miniz_oxide); a
/// backend update that changes the compressed bytes shows up here, while decode is unaffected.
#[test]
fn archive_bytes_are_the_recorded_ones() {
    use sha2::{Digest, Sha256};
    let domain = masked_lattice(48, 32);
    let mut rng = Lcg(42);
    let values = values_for(&domain, |_, i, j| 25 * i - 12 * j + rng.signed(40));
    let plane = PlanePerTile::fit(&domain, &values, 16, 8).expect("fit");
    let cases: [(&str, &dyn IntLaw2, Mode); 3] = [
        ("plane lossless", &plane, Mode::Lossless),
        ("plane q=7", &plane, Mode::Bounded { q: 7 }),
        ("zero lossless", &Zero, Mode::Lossless),
    ];
    let mut got = Vec::new();
    for (name, law, mode) in cases {
        let enc = encode(&domain, &values, law, mode).expect("encode");
        let hex: String = Sha256::digest(&enc.bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        got.push(format!("{name} {} {hex}", enc.bytes.len()));
    }
    if std::env::var("INT2D_PRINT_GOLDEN").as_deref() == Ok("1") {
        for line in &got {
            println!("{line}");
        }
    }
    let want = GOLDEN_ARCHIVES;
    assert_eq!(
        got.len(),
        want.len(),
        "compared a different number of archives"
    );
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g, w);
    }
}

const GOLDEN_ARCHIVES: [&str; 3] = [
    "plane lossless 1065 642abe9806bd632ce6ade79710c19bf96437f3e9648a0e604e8664abd78e1712",
    "plane q=7 742 d23523533dcf5e27604280a683961ecc5e5b003d1238d40279a8c19c0ae64550",
    "zero lossless 1826 2ed498a851efe886a5d35413adf3df5dc8a5f370b8d3044b6208fd6e851a5022",
];

/// The tie direction of `Mode::Bounded`: `r' = q * round_half_up(r / q)`, so a residual of
/// exactly half a step goes up (r = 1, q = 2 -> 2; r = -1, q = 2 -> 0; r = 3, q = 6 -> 6)
#[test]
fn bounded_ties_round_half_up() {
    let domain = full_lattice(3, 1);
    let values = [1_i64, -1, 3];
    let expect = [[2_i64, 0, 4], [0, 0, 6]];
    let mut compared = 0;
    for (q, want) in [2_u64, 6].into_iter().zip(expect) {
        let enc = encode(&domain, &values, &Zero, Mode::Bounded { q }).expect("encode");
        let back = decode(&enc.bytes, &domain, &Zero).expect("decode");
        assert_eq!(back, want, "q = {q}");
        compared += back.len();
    }
    assert!(compared > 0);
}

/// An archive whose payload was changed and whose trailing SHA-256 was recomputed to match
/// (so it is internally consistent) is still refused: the header's digest of the values does
/// not match what the changed residuals decode to
#[test]
fn a_rewritten_payload_with_a_matching_trailer_is_refused() {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    const HEADER: usize = 4 + 2 + 32 + 32 + 1 + 8 + 8 + 8 + 32 + 32 + 1 + 8;
    let domain = full_lattice(8, 8);
    // 0..64 zigzag to 0..127: one varint byte each
    let values = values_for(&domain, |k, _, _| i64::try_from(k).unwrap());
    let enc = encode(&domain, &values, &Zero, Mode::Lossless).expect("encode");
    let body = &enc.bytes[..enc.bytes.len() - 32];
    let payload = &body[HEADER..];
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(payload)
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(
        raw.len(),
        values.len(),
        "every residual here is one varint byte"
    );
    raw[10] ^= 2; // a different residual, still a valid one-byte varint
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    z.write_all(&raw).unwrap();
    let new_payload = z.finish().unwrap();
    let mut forged = body[..HEADER - 8].to_vec();
    forged.extend_from_slice(&u64::try_from(new_payload.len()).unwrap().to_be_bytes());
    forged.extend_from_slice(&new_payload);
    let trailer: [u8; 32] = Sha256::digest(&forged).into();
    forged.extend_from_slice(&trailer);
    assert!(matches!(
        decode(&forged, &domain, &Zero),
        Err(Int2dError::HashMismatch)
    ));
}

/// The fields of an archive, split at the documented offsets, and an archive rebuilt from
/// them with a recomputed trailing SHA-256 (so a forged archive is internally consistent and
/// only the check under test can refuse it)
struct Fields {
    header: Vec<u8>,
    payload: Vec<u8>,
}

const HEADER: usize = 4 + 2 + 32 + 32 + 1 + 8 + 8 + 8 + 32 + 32 + 1 + 8;
const MODE_AT: usize = 70;
const MAX_ERR_AT: usize = 87;
const ORIGINAL_AT: usize = 95;
const CODEC_AT: usize = 159;

fn split(bytes: &[u8]) -> Fields {
    let body = &bytes[..bytes.len() - 32];
    Fields {
        header: body[..HEADER].to_vec(),
        payload: body[HEADER..].to_vec(),
    }
}

fn rebuild(f: &Fields) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut b = f.header.clone();
    b[HEADER - 8..].copy_from_slice(&u64::try_from(f.payload.len()).unwrap().to_be_bytes());
    b.extend_from_slice(&f.payload);
    let t: [u8; 32] = Sha256::digest(&b).into();
    b.extend_from_slice(&t);
    b
}

#[test]
fn the_forging_helper_reproduces_a_real_archive() {
    let domain = full_lattice(5, 5);
    let values = values_for(&domain, |k, _, _| i64::try_from(k).unwrap());
    let enc = encode(&domain, &values, &Zero, Mode::Bounded { q: 4 }).unwrap();
    assert_eq!(rebuild(&split(&enc.bytes)), enc.bytes);
    assert_eq!(enc.bytes[MODE_AT], 1, "bounded mode byte");
}

/// A forged error bound above q / 2 (with a matching trailer), and a lossless archive that
/// states a non-zero error, are refused by the header check
#[test]
fn a_forged_error_bound_is_refused() {
    let domain = full_lattice(6, 6);
    let values = values_for(&domain, |k, _, _| i64::try_from(k).unwrap() * 7);
    let enc = encode(&domain, &values, &Zero, Mode::Bounded { q: 10 }).unwrap();
    let mut f = split(&enc.bytes);
    f.header[MAX_ERR_AT..MAX_ERR_AT + 8].copy_from_slice(&6_u64.to_be_bytes()); // q / 2 + 1
    assert!(matches!(
        decode(&rebuild(&f), &domain, &Zero),
        Err(Int2dError::BadHeader("error bound"))
    ));
    f.header[MAX_ERR_AT..MAX_ERR_AT + 8].copy_from_slice(&5_u64.to_be_bytes()); // q / 2: allowed
    assert!(decode(&rebuild(&f), &domain, &Zero).is_ok());
    let enc = encode(&domain, &values, &Zero, Mode::Lossless).unwrap();
    let mut f = split(&enc.bytes);
    f.header[MAX_ERR_AT..MAX_ERR_AT + 8].copy_from_slice(&1_u64.to_be_bytes());
    assert!(matches!(
        decode(&rebuild(&f), &domain, &Zero),
        Err(Int2dError::BadHeader("error bound"))
    ));
}

/// A lossless archive whose two digests differ (with a matching trailer) is refused before
/// anything is decoded
#[test]
fn a_lossless_archive_with_different_digests_is_refused() {
    let domain = full_lattice(6, 6);
    let values = values_for(&domain, |k, _, _| i64::try_from(k).unwrap());
    let enc = encode(&domain, &values, &Zero, Mode::Lossless).unwrap();
    let mut f = split(&enc.bytes);
    f.header[ORIGINAL_AT] ^= 1;
    assert!(matches!(
        decode(&rebuild(&f), &domain, &Zero),
        Err(Int2dError::BadHeader("lossless digests differ"))
    ));
}

/// More varints than cells (with a matching trailer) are refused, in both codecs
#[test]
fn extra_residuals_are_refused() {
    let domain = full_lattice(4, 4);
    let values = vec![0_i64; 16];
    for codec in [Codec::Raw, Codec::Deflate] {
        let enc = encode_with(&domain, &values, &Zero, Mode::Lossless, codec).unwrap();
        let mut f = split(&enc.bytes);
        let mut raw = if codec == Codec::Raw {
            f.payload.clone()
        } else {
            inflate(&f.payload)
        };
        assert_eq!(raw.len(), 16);
        raw.push(0);
        f.payload = if codec == Codec::Raw {
            raw
        } else {
            deflate(&raw)
        };
        assert!(matches!(
            decode(&rebuild(&f), &domain, &Zero),
            Err(Int2dError::BadPayload("more residuals than cells"))
        ));
    }
}

/// A payload that expands past what the cell count allows (10 bytes per varint) is refused
/// with its own error, for a 1-cell archive whose payload inflates to 1 MiB
#[test]
fn a_payload_expanding_past_the_count_is_refused() {
    let domain = full_lattice(1, 1);
    let enc = encode(&domain, &[0], &Zero, Mode::Lossless).unwrap();
    let mut f = split(&enc.bytes);
    f.payload = deflate(&vec![0_u8; 1 << 20]);
    assert!(matches!(
        decode(&rebuild(&f), &domain, &Zero),
        Err(Int2dError::BadPayload("payload expands past the count"))
    ));
    // the same in the raw codec: 11 bytes for 1 cell
    let enc = encode_with(&domain, &[0], &Zero, Mode::Lossless, Codec::Raw).unwrap();
    let mut f = split(&enc.bytes);
    f.payload = vec![0_u8; 11];
    assert!(matches!(
        decode(&rebuild(&f), &domain, &Zero),
        Err(Int2dError::BadPayload("payload expands past the count"))
    ));
}

fn inflate(b: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(b)
        .read_to_end(&mut out)
        .unwrap();
    out
}

fn deflate(b: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    z.write_all(b).unwrap();
    z.finish().unwrap()
}

/// zigzag + LEB128, written here from the definitions (independent of the module)
fn varints(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    for &v in values {
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

/// `Codec::Raw` stores the zigzag + LEB128 residuals as they are; with the zero law the
/// payload is the values' varints exactly, and the archive decodes like the deflated one
#[test]
fn the_raw_codec_stores_the_varints_as_they_are() {
    let domain = masked_lattice(30, 20);
    let mut rng = Lcg(77);
    let values = values_for(&domain, |_, _, _| rng.signed(1 << 40));
    let enc = encode_with(&domain, &values, &Zero, Mode::Lossless, Codec::Raw).unwrap();
    assert_eq!(enc.bytes[CODEC_AT], 1, "raw codec byte");
    let f = split(&enc.bytes);
    assert_eq!(f.payload, varints(&values));
    assert_eq!(decode(&enc.bytes, &domain, &Zero).unwrap(), values);
    let deflated = encode(&domain, &values, &Zero, Mode::Lossless).unwrap();
    assert_eq!(deflated.bytes[CODEC_AT], 0, "deflate is the default");
    assert_eq!(inflate(&split(&deflated.bytes).payload), f.payload);
    // bounded with the raw codec keeps its bound
    let law = PlanePerTile::fit(&domain, &values, 16, 8).unwrap();
    let enc = encode_with(&domain, &values, &law, Mode::Bounded { q: 9 }, Codec::Raw).unwrap();
    let back = decode(&enc.bytes, &domain, &law).unwrap();
    assert!(values.iter().zip(&back).all(|(v, b)| (v - b).abs() <= 4));
    assert!(!back.is_empty());
}

/// A fitted law stored with `to_bytes` and read back with `from_bytes` (a fresh object, as a
/// decoder in another process would build) has the same identifier and decodes the archive;
/// `from_parts` gives the same law; every single bit flipped in the stored law is refused
#[test]
fn a_stored_law_decodes_the_archive() {
    let domain = masked_lattice(40, 28);
    let mut rng = Lcg(31);
    let values = values_for(&domain, |_, i, j| 9 * i - 4 * j + rng.signed(50));
    let law = PlanePerTile::fit(&domain, &values, 8, 6).unwrap();
    let enc = encode(&domain, &values, &law, Mode::Lossless).unwrap();
    let stored = law.to_bytes();
    let back = PlanePerTile::from_bytes(&stored).unwrap();
    assert_eq!(back, law);
    assert_eq!(back.law_id(), law.law_id());
    assert_eq!(decode(&enc.bytes, &domain, &back).unwrap(), values);
    let parts = PlanePerTile::from_parts(
        law.origin(),
        law.tile(),
        law.denominator_bits(),
        law.coefficients().clone(),
    )
    .unwrap();
    assert_eq!(parts.law_id(), law.law_id());
    let mut flipped = 0;
    for byte in 0..stored.len() {
        for bit in 0..8 {
            let mut b = stored.clone();
            b[byte] ^= 1 << bit;
            assert!(
                PlanePerTile::from_bytes(&b).is_err(),
                "flip at {byte}:{bit} was accepted"
            );
            flipped += 1;
        }
    }
    assert!(flipped > 0);
    assert!(PlanePerTile::from_bytes(&stored[..stored.len() - 1]).is_err());
    assert!(matches!(
        PlanePerTile::from_parts((0, 0), 0, 4, law.coefficients().clone()),
        Err(Int2dError::InvalidParameter(_))
    ));
    assert!(matches!(
        PlanePerTile::from_parts((0, 0), 8, 33, law.coefficients().clone()),
        Err(Int2dError::InvalidParameter(_))
    ));
    assert!(matches!(
        PlanePerTile::from_parts((0, 0), 8, 4, Default::default()),
        Err(Int2dError::InvalidParameter(_))
    ));
}

/// Tiles in the caller's units, far beyond the old 1024 limit: points at coordinates around
/// 10^9 (a 1e-7 degree grid), tiles of 10^4 and 10^5 units, values around 10^9 that lie on
/// a plane. The fit is exact, the archive round trips, and an input past the documented
/// bound is an overflow error, not a wrong plane.
#[test]
fn large_tiles_and_large_coordinates() {
    // a 41 x 41 grid of points 10^4 apart, at coordinates around 1.35e9 / 3.5e8: a tile of
    // 10^5 holds 10 x 10 points, and the last tiles hold one row or one column (fitted by a
    // line, which is exact for points on a plane)
    let mut pts = Vec::new();
    for a in 0..41_i64 {
        for b in 0..41_i64 {
            pts.push((1_349_800_000 + a * 10_000, 349_800_000 + b * 10_000));
        }
    }
    let domain = Domain::points(pts).unwrap();
    let values = values_for(&domain, |_, i, j| 3 * i - 5 * j + 7);
    let mut compared = 0;
    // 10^4: one point per tile (the mean, exact); 3 * 10^4: tiles of 3 x 3 points; 10^5:
    // tiles of 10 x 10 points, within the documented bound (12 n^3 T^4 D 2^k with n = 100,
    // T = 10^5, D < 10^6, k = 8: below 2^118)
    for tile in [10_000_u32, 30_000, 100_000] {
        let law = PlanePerTile::fit(&domain, &values, tile, 8).expect("fit");
        let r = residuals(&domain, &values, &law).unwrap();
        assert!(
            r.iter().all(|&x| x == 0),
            "tile {tile}: a plane left a residual"
        );
        let enc = encode(&domain, &values, &law, Mode::Lossless).unwrap();
        assert_eq!(decode(&enc.bytes, &domain, &law).unwrap(), values);
        compared += r.len();
    }
    assert!(compared > 0);
    // far past the bound (12 n^3 T^4 D 2^k with T = 2^32 - 1, D about 2^62, k = 32)
    let wild = values_for(&domain, |k, _, _| {
        if k % 2 == 0 {
            i64::MAX / 2
        } else {
            i64::MIN / 2
        }
    });
    assert!(matches!(
        PlanePerTile::fit(&domain, &wild, u32::MAX, 32),
        Err(Int2dError::Overflow)
    ));
}

/// A stored law with two tiles swapped (out of index order) and its SHA-256 recomputed is
/// refused: the order is part of the format, so one law has one byte string
#[test]
fn a_stored_law_out_of_tile_order_is_refused() {
    use sha2::{Digest, Sha256};
    let domain = full_lattice(32, 8);
    let values = values_for(&domain, |k, i, j| 2 * i + j + i64::try_from(k % 5).unwrap());
    let law = PlanePerTile::fit(&domain, &values, 8, 4).unwrap();
    assert!(law.coefficients().len() >= 2);
    let stored = law.to_bytes();
    const FIXED: usize = 4 + 2 + 8 + 8 + 4 + 1 + 8;
    let mut body = stored[..stored.len() - 32].to_vec();
    let (first, second) = (FIXED, FIXED + 40);
    let tile0 = body[first..first + 40].to_vec();
    let tile1 = body[second..second + 40].to_vec();
    body[first..first + 40].copy_from_slice(&tile1);
    body[second..second + 40].copy_from_slice(&tile0);
    let check: [u8; 32] = Sha256::digest(&body).into();
    body.extend_from_slice(&check);
    assert!(matches!(
        PlanePerTile::from_bytes(&body),
        Err(Int2dError::BadHeader("law tiles out of order"))
    ));
}
