//! Integer 2-D residual codec: values on an integer grid (or a set of integer points) stored
//! as the residual against an integer law
//!
//! Everything on the prediction and residual path is integer arithmetic: a law predicts an
//! `i64` for a cell `(i, j)`, the residual is `value - prediction`, and the residuals are
//! zigzag + LEB128 coded before the deflate backend. Nothing here uses a floating-point
//! type, so the same input gives the same bytes on every target, and this module does not
//! depend on [`crate::law::SEMANTICS_ID`] (the identifier of the floating-point arithmetic used
//! by the other laws of this crate); the law identifiers below therefore take no semantics
//! identifier. Overflow is an error ([`Int2dError::Overflow`]), never a wrap.
//!
//! - [`Domain`]: a lattice (origin, size and a presence mask) or a sorted set of points.
//!   The cells are visited row by row for a lattice (`j` outer, `i` inner, present cells
//!   only) and in `(i, j)` order for points.
//! - [`IntLaw2`]: an integer law, implemented here by [`Zero`] and [`PlanePerTile`] and by
//!   the caller for its own base surfaces.
//! - [`encode`] / [`decode`] (`std` feature): the archive, with the law identifier, the
//!   domain digest, the mode and the SHA-256 of the values in its header.
//!
//! A fitted [`PlanePerTile`] is stored apart from the archives it encodes
//! ([`PlanePerTile::to_bytes`] / [`PlanePerTile::from_bytes`]): the archive names its law
//! only by [`IntLaw2::law_id`], so the same law is not repeated in every archive, and a
//! caller's own law (which this module cannot serialize) works the same way.
//!
//! The README's example (the same code, run as a doctest):
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use alice_zip::int2d::{decode, encode, Domain, Mode, PlanePerTile};
//!
//! let domain = Domain::lattice(0, 0, 4, 4, None)?;
//! let values: Vec<i64> = (0..16).map(|k| 3 * k - 5).collect();
//! let law = PlanePerTile::fit(&domain, &values, 4, 8)?;
//! let enc = encode(&domain, &values, &law, Mode::Lossless)?;
//! assert_eq!(decode(&enc.bytes, &domain, &law)?, values);
//! # }
//! # Ok::<(), alice_zip::int2d::Int2dError>(())
//! ```
#![deny(
    clippy::float_arithmetic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use sha2::{Digest, Sha256};

use crate::law::LAW_ID_DOMAIN;

/// Kind tag of [`Zero`] in its law identifier
pub const ZERO_LAW_KIND: &[u8] = b"int-law-2d/zero/v1";
/// Kind tag of [`PlanePerTile`] in its law identifier
pub const PLANE_PER_TILE_KIND: &[u8] = b"int-law-2d/plane-per-tile/v1";
/// Domain separation of [`Domain::digest`]
pub const DOMAIN_DIGEST_DOMAIN: &[u8] = b"alice-zip/int2d/domain/v1";
/// Largest denominator exponent of [`PlanePerTile`] (coefficients are multiples of 2^-k)
pub const MAX_DENOMINATOR_BITS: u8 = 32;

/// Errors of this module
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Int2dError {
    /// The domain has no cell
    EmptyDomain,
    /// The number of values is not the number of cells of the domain
    LengthMismatch {
        /// cells in the domain
        cells: usize,
        /// values given
        values: usize,
    },
    /// An argument is outside its range (the message says which)
    InvalidParameter(&'static str),
    /// An integer operation would overflow
    Overflow,
    /// A cell the law has no prediction for
    OutsideLaw {
        /// the cell
        cell: (i64, i64),
    },
    /// The archive was written with another law
    LawIdMismatch,
    /// The archive was written for another domain
    DomainMismatch,
    /// The archive header is not one this module writes (the message says why)
    BadHeader(&'static str),
    /// The residual stream does not decode to the stated number of values
    BadPayload(&'static str),
    /// The decoded values do not match the SHA-256 in the header
    HashMismatch,
    /// The compression backend failed
    Backend(String),
}

impl fmt::Display for Int2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDomain => write!(f, "the domain has no cell"),
            Self::LengthMismatch { cells, values } => {
                write!(f, "{values} values for a domain of {cells} cells")
            }
            Self::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
            Self::Overflow => write!(f, "integer overflow"),
            Self::OutsideLaw { cell } => write!(f, "the law has no prediction for {cell:?}"),
            Self::LawIdMismatch => write!(f, "the archive was written with another law"),
            Self::DomainMismatch => write!(f, "the archive was written for another domain"),
            Self::BadHeader(why) => write!(f, "bad header: {why}"),
            Self::BadPayload(why) => write!(f, "bad payload: {why}"),
            Self::HashMismatch => write!(f, "the decoded values do not match the header digest"),
            Self::Backend(e) => write!(f, "compression backend: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Int2dError {}

/// `floor(num / den + 1/2)` for `den > 0`: rounding to nearest, halves toward +infinity
fn round_half_up(num: i128, den: i128) -> Result<i128, Int2dError> {
    debug_assert!(den > 0);
    let two_num = num.checked_mul(2).ok_or(Int2dError::Overflow)?;
    let two_den = den.checked_mul(2).ok_or(Int2dError::Overflow)?;
    let shifted = two_num.checked_add(den).ok_or(Int2dError::Overflow)?;
    Ok(shifted.div_euclid(two_den))
}

fn to_i64(x: i128) -> Result<i64, Int2dError> {
    i64::try_from(x).map_err(|_| Int2dError::Overflow)
}

fn len_u64(n: usize) -> u64 {
    // usize is at most 64 bits on every target this crate builds for
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn update_length_prefixed(h: &mut Sha256, bytes: &[u8]) {
    h.update(len_u64(bytes.len()).to_be_bytes());
    h.update(bytes);
}

// ------------------------------------------------------------------ domain

/// The cells the values live on
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Domain {
    /// `ni x nj` cells starting at `(i0, j0)`; `present` has one bit per cell, row by row
    Lattice {
        /// first column
        i0: i64,
        /// first row
        j0: i64,
        /// columns
        ni: u32,
        /// rows
        nj: u32,
        /// bit `k` of word `k / 64` set = cell `k` (row-major) is present
        present: Vec<u64>,
    },
    /// Distinct points, sorted by `(i, j)`
    Points(Vec<(i64, i64)>),
}

impl Domain {
    /// A lattice of `ni x nj` cells at `(i0, j0)`, every cell present unless `present` says
    /// otherwise (one flag per cell, row by row)
    ///
    /// # Errors
    ///
    /// [`Int2dError::EmptyDomain`] when no cell is present, [`Int2dError::InvalidParameter`]
    /// when `present` has the wrong length, [`Int2dError::Overflow`] when the last cell's
    /// coordinates do not fit in `i64`
    pub fn lattice(
        i0: i64,
        j0: i64,
        ni: u32,
        nj: u32,
        present: Option<&[bool]>,
    ) -> Result<Self, Int2dError> {
        let n = usize::try_from(u64::from(ni) * u64::from(nj))
            .map_err(|_| Int2dError::InvalidParameter("lattice too large"))?;
        if n == 0 {
            return Err(Int2dError::EmptyDomain);
        }
        i0.checked_add(i64::from(ni) - 1)
            .ok_or(Int2dError::Overflow)?;
        j0.checked_add(i64::from(nj) - 1)
            .ok_or(Int2dError::Overflow)?;
        let mut words = alloc::vec![0_u64; n.div_ceil(64)];
        match present {
            None => {
                for k in 0..n {
                    words[k / 64] |= 1 << (k % 64);
                }
            }
            Some(p) => {
                if p.len() != n {
                    return Err(Int2dError::InvalidParameter(
                        "present has one flag per cell",
                    ));
                }
                for (k, &on) in p.iter().enumerate() {
                    if on {
                        words[k / 64] |= 1 << (k % 64);
                    }
                }
            }
        }
        if words.iter().all(|&w| w == 0) {
            return Err(Int2dError::EmptyDomain);
        }
        Ok(Self::Lattice {
            i0,
            j0,
            ni,
            nj,
            present: words,
        })
    }

    /// A set of points, sorted here by `(i, j)`
    ///
    /// # Errors
    ///
    /// [`Int2dError::EmptyDomain`] for no point, [`Int2dError::InvalidParameter`] when a
    /// point is repeated
    pub fn points(mut points: Vec<(i64, i64)>) -> Result<Self, Int2dError> {
        if points.is_empty() {
            return Err(Int2dError::EmptyDomain);
        }
        points.sort_unstable();
        if points.windows(2).any(|w| w[0] == w[1]) {
            return Err(Int2dError::InvalidParameter("a point is repeated"));
        }
        Ok(Self::Points(points))
    }

    /// The number of cells
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Lattice { present, .. } => present.iter().map(|w| w.count_ones() as usize).sum(),
            Self::Points(p) => p.len(),
        }
    }

    /// Always false: a domain has at least one cell
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The cells in order (row by row for a lattice, `(i, j)` order for points)
    pub fn cells(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        let (lattice, points) = match self {
            Self::Lattice {
                i0,
                j0,
                ni,
                nj,
                present,
            } => {
                let n = usize::try_from(u64::from(*ni) * u64::from(*nj)).unwrap_or(0);
                let ni_us = *ni as usize;
                let it = (0..n)
                    .filter(move |k| present[k / 64] >> (k % 64) & 1 == 1)
                    .map(move |k| {
                        // k / ni < nj and k % ni < ni, both checked to fit in `lattice`
                        let di = i64::try_from(k % ni_us).unwrap_or(0);
                        let dj = i64::try_from(k / ni_us).unwrap_or(0);
                        (i0 + di, j0 + dj)
                    });
                (Some(it), None)
            }
            Self::Points(p) => (None, Some(p.iter().copied())),
        };
        lattice
            .into_iter()
            .flatten()
            .chain(points.into_iter().flatten())
    }

    /// SHA-256 of the canonical encoding of the domain (kind, parameters and mask or points,
    /// big-endian)
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        update_length_prefixed(&mut h, DOMAIN_DIGEST_DOMAIN);
        match self {
            Self::Lattice {
                i0,
                j0,
                ni,
                nj,
                present,
            } => {
                h.update([0_u8]);
                h.update(i0.to_be_bytes());
                h.update(j0.to_be_bytes());
                h.update(ni.to_be_bytes());
                h.update(nj.to_be_bytes());
                for w in present {
                    h.update(w.to_be_bytes());
                }
            }
            Self::Points(p) => {
                h.update([1_u8]);
                h.update(len_u64(p.len()).to_be_bytes());
                for (i, j) in p {
                    h.update(i.to_be_bytes());
                    h.update(j.to_be_bytes());
                }
            }
        }
        h.finalize().into()
    }
}

// ------------------------------------------------------------------ laws

/// An integer law over 2-D cells: an exact integer prediction for each cell it covers
pub trait IntLaw2 {
    /// The prediction for cell `(i, j)`
    ///
    /// # Errors
    ///
    /// [`Int2dError::Overflow`] when the prediction does not fit, [`Int2dError::OutsideLaw`]
    /// for a cell the law does not cover
    fn predict(&self, i: i64, j: i64) -> Result<i64, Int2dError>;

    /// A content identifier of the law (its kind and all its parameters): an archive written
    /// with one law decodes only with a law of the same identifier
    fn law_id(&self) -> [u8; 32];
}

/// The law that predicts 0 everywhere (the residuals are the values)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zero;

impl IntLaw2 for Zero {
    fn predict(&self, _i: i64, _j: i64) -> Result<i64, Int2dError> {
        Ok(0)
    }

    fn law_id(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        update_length_prefixed(&mut h, LAW_ID_DOMAIN);
        update_length_prefixed(&mut h, ZERO_LAW_KIND);
        h.finalize().into()
    }
}

/// One plane per square tile, with rational coefficients of the fixed denominator `2^k`
///
/// The tile of cell `(i, j)` is `((i - i0) div tile, (j - j0) div tile)` (floor division), and
/// within it `x = (i - i0) mod tile`, `y = (j - j0) mod tile`. The prediction is
/// `round_half_up((a x + b y + c) / 2^k)`, with `(a, b, c)` the integer coefficients of that
/// tile (so the plane's slopes and offset are multiples of `2^-k`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanePerTile {
    i0: i64,
    j0: i64,
    tile: u32,
    k: u8,
    coeffs: BTreeMap<(i64, i64), (i64, i64, i64)>,
}

impl PlanePerTile {
    /// Least-squares plane per tile for `values` on `domain`, tiles of `tile x tile` cells
    /// anchored at the domain's first cell (the lattice origin, or the smallest point), and
    /// coefficients rounded half up to multiples of `2^-k`.
    ///
    /// The fit is exact integer arithmetic: the normal equations of each tile are summed in
    /// `i128` and solved by Cramer's rule, so it gives the same coefficients everywhere. A
    /// tile whose points do not determine a plane (fewer than 3, or all on a line) gets the
    /// least-squares line in `y`, else in `x`, else the mean.
    ///
    /// `tile` is in the units of the coordinates and has no fixed upper limit. Each tile is
    /// fitted to its values minus the tile's first value `v0` (added back to `c`, which gives
    /// the same coefficients), so the sizes involved are the number of cells `n` of a tile,
    /// the tile edge `T` and the spread `D = max |v - v0|` within the tile. The fit succeeds
    /// when `12 n^3 T^4 max(D, 1) 2^k < 2^127` (the largest term of the determinants and the
    /// scaling); beyond that, or for coefficients that do not fit in `i64`, it returns
    /// [`Int2dError::Overflow`] rather than a wrong plane. For example `n = 10^3` cells,
    /// `T = 10^5` and `D = 10^4` with `k = 8` give about `2^121`, below the bound;
    /// `n = 10^4` with the same `T`, `D` and `k` does not.
    ///
    /// # Errors
    ///
    /// [`Int2dError::InvalidParameter`] for `tile = 0` or `k > MAX_DENOMINATOR_BITS`, [`Int2dError::LengthMismatch`], [`Int2dError::Overflow`]
    /// when a sum or determinant does not fit in `i128` or a coefficient in `i64`
    pub fn fit(domain: &Domain, values: &[i64], tile: u32, k: u8) -> Result<Self, Int2dError> {
        check_tile_and_k(tile, k)?;
        let n = domain.len();
        if n != values.len() {
            return Err(Int2dError::LengthMismatch {
                cells: n,
                values: values.len(),
            });
        }
        let (i0, j0) = match domain {
            Domain::Lattice { i0, j0, .. } => (*i0, *j0),
            Domain::Points(p) => p[0],
        };
        // per tile: the tile's first value v0, and n, Sx, Sy, Sxx, Sxy, Syy, Sv, Sxv, Syv of
        // the shifted values v - v0 (the fit of v - v0 plus v0 is the fit of v)
        let mut sums: BTreeMap<(i64, i64), (i128, [i128; 9])> = BTreeMap::new();
        let t = i64::from(tile);
        for ((i, j), &v) in domain.cells().zip(values) {
            let (key, x, y) = tile_of(i0, j0, t, i, j)?;
            let (v0, s) = sums.entry(key).or_insert((i128::from(v), [0; 9]));
            let (x, y, v) = (i128::from(x), i128::from(y), sub(i128::from(v), *v0)?);
            let terms = [
                1,
                x,
                y,
                mul(x, x)?,
                mul(x, y)?,
                mul(y, y)?,
                v,
                mul(x, v)?,
                mul(y, v)?,
            ];
            for (acc, term) in s.iter_mut().zip(terms) {
                *acc = acc.checked_add(term).ok_or(Int2dError::Overflow)?;
            }
        }
        let scale = 1_i128 << k;
        let mut coeffs = BTreeMap::new();
        for (key, (v0, s)) in sums {
            let (a, b, c) = solve_plane(&s, scale)?;
            let c = to_i64(add(i128::from(c), mul(v0, scale)?)?)?;
            coeffs.insert(key, (a, b, c));
        }
        Ok(Self {
            i0,
            j0,
            tile,
            k,
            coeffs,
        })
    }

    /// The tile edge
    #[must_use]
    pub fn tile(&self) -> u32 {
        self.tile
    }

    /// The denominator exponent `k` (coefficients are multiples of `2^-k`)
    #[must_use]
    pub fn denominator_bits(&self) -> u8 {
        self.k
    }

    /// The integer coefficients `(a, b, c)` of each fitted tile, by tile index
    #[must_use]
    pub fn coefficients(&self) -> &BTreeMap<(i64, i64), (i64, i64, i64)> {
        &self.coeffs
    }

    /// The tiles' origin `(i0, j0)` (the cell at tile-local `(0, 0)` of tile `(0, 0)`)
    #[must_use]
    pub fn origin(&self) -> (i64, i64) {
        (self.i0, self.j0)
    }

    /// A law from stored parts (what [`Self::origin`], [`Self::tile`],
    /// [`Self::denominator_bits`] and [`Self::coefficients`] return), with the same checks as
    /// [`Self::fit`]; the result has the same [`IntLaw2::law_id`] as the law the parts came from
    ///
    /// # Errors
    ///
    /// [`Int2dError::InvalidParameter`] for `tile = 0`, `k > MAX_DENOMINATOR_BITS` or no tile
    pub fn from_parts(
        origin: (i64, i64),
        tile: u32,
        k: u8,
        coeffs: BTreeMap<(i64, i64), (i64, i64, i64)>,
    ) -> Result<Self, Int2dError> {
        check_tile_and_k(tile, k)?;
        if coeffs.is_empty() {
            return Err(Int2dError::InvalidParameter("a law has at least one tile"));
        }
        Ok(Self {
            i0: origin.0,
            j0: origin.1,
            tile,
            k,
            coeffs,
        })
    }

    /// The law as bytes: magic `I2PT`, version, origin, tile, `k`, the number of tiles, each
    /// tile's index and coefficients (big-endian, in index order), and the SHA-256 of all of it
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(4 + 2 + 8 + 8 + 4 + 1 + 8 + self.coeffs.len() * 40 + 32);
        b.extend_from_slice(&PLANE_PER_TILE_MAGIC);
        b.extend_from_slice(&PLANE_PER_TILE_VERSION.to_be_bytes());
        b.extend_from_slice(&self.i0.to_be_bytes());
        b.extend_from_slice(&self.j0.to_be_bytes());
        b.extend_from_slice(&self.tile.to_be_bytes());
        b.push(self.k);
        b.extend_from_slice(&len_u64(self.coeffs.len()).to_be_bytes());
        for (&(ti, tj), &(a, bb, c)) in &self.coeffs {
            for x in [ti, tj, a, bb, c] {
                b.extend_from_slice(&x.to_be_bytes());
            }
        }
        let check: [u8; 32] = Sha256::digest(&b).into();
        b.extend_from_slice(&check);
        b
    }

    /// The law [`Self::to_bytes`] wrote
    ///
    /// # Errors
    ///
    /// [`Int2dError::BadHeader`] for another magic or version, a wrong length, tiles out of
    /// index order or a SHA-256 that does not match, and the checks of [`Self::from_parts`]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Int2dError> {
        const FIXED: usize = 4 + 2 + 8 + 8 + 4 + 1 + 8;
        let body_len = bytes
            .len()
            .checked_sub(32)
            .filter(|&n| n >= FIXED)
            .ok_or(Int2dError::BadHeader("truncated law"))?;
        let (body, check) = bytes.split_at(body_len);
        if body[..4] != PLANE_PER_TILE_MAGIC {
            return Err(Int2dError::BadHeader("law magic"));
        }
        if u16::from_be_bytes([body[4], body[5]]) != PLANE_PER_TILE_VERSION {
            return Err(Int2dError::BadHeader("law version"));
        }
        if Sha256::digest(body).as_slice() != check {
            return Err(Int2dError::HashMismatch);
        }
        let i64_at = |at: usize| {
            let mut a = [0_u8; 8];
            a.copy_from_slice(&body[at..at + 8]);
            i64::from_be_bytes(a)
        };
        let (i0, j0) = (i64_at(6), i64_at(14));
        let tile = u32::from_be_bytes([body[22], body[23], body[24], body[25]]);
        let k = body[26];
        let mut nb = [0_u8; 8];
        nb.copy_from_slice(&body[27..35]);
        let n = usize::try_from(u64::from_be_bytes(nb))
            .map_err(|_| Int2dError::BadHeader("law length"))?;
        if n.checked_mul(40).and_then(|x| x.checked_add(FIXED)) != Some(body.len()) {
            return Err(Int2dError::BadHeader("law length"));
        }
        let mut coeffs = BTreeMap::new();
        let mut last: Option<(i64, i64)> = None;
        for t in 0..n {
            let at = FIXED + t * 40;
            let key = (i64_at(at), i64_at(at + 8));
            if last.is_some_and(|l| l >= key) {
                return Err(Int2dError::BadHeader("law tiles out of order"));
            }
            last = Some(key);
            coeffs.insert(key, (i64_at(at + 16), i64_at(at + 24), i64_at(at + 32)));
        }
        Self::from_parts((i0, j0), tile, k, coeffs)
    }
}

fn check_tile_and_k(tile: u32, k: u8) -> Result<(), Int2dError> {
    if tile == 0 {
        return Err(Int2dError::InvalidParameter("tile must be at least 1"));
    }
    if k > MAX_DENOMINATOR_BITS {
        return Err(Int2dError::InvalidParameter(
            "k must be at most MAX_DENOMINATOR_BITS",
        ));
    }
    Ok(())
}

/// Magic of a serialized [`PlanePerTile`] ([`PlanePerTile::to_bytes`])
pub const PLANE_PER_TILE_MAGIC: [u8; 4] = *b"I2PT";
/// Version of the [`PlanePerTile`] serialization
pub const PLANE_PER_TILE_VERSION: u16 = 1;

/// tile index and tile-local coordinates of cell `(i, j)`
fn tile_of(i0: i64, j0: i64, t: i64, i: i64, j: i64) -> Result<((i64, i64), i64, i64), Int2dError> {
    let di = i.checked_sub(i0).ok_or(Int2dError::Overflow)?;
    let dj = j.checked_sub(j0).ok_or(Int2dError::Overflow)?;
    Ok((
        (di.div_euclid(t), dj.div_euclid(t)),
        di.rem_euclid(t),
        dj.rem_euclid(t),
    ))
}

fn mul(a: i128, b: i128) -> Result<i128, Int2dError> {
    a.checked_mul(b).ok_or(Int2dError::Overflow)
}

fn sub(a: i128, b: i128) -> Result<i128, Int2dError> {
    a.checked_sub(b).ok_or(Int2dError::Overflow)
}

fn add(a: i128, b: i128) -> Result<i128, Int2dError> {
    a.checked_add(b).ok_or(Int2dError::Overflow)
}

/// det of [[a, b, c], [d, e, f], [g, h, i]]
#[allow(clippy::many_single_char_names)]
fn det3(m: [[i128; 3]; 3]) -> Result<i128, Int2dError> {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let t1 = mul(a, sub(mul(e, i)?, mul(f, h)?)?)?;
    let t2 = mul(b, sub(mul(d, i)?, mul(f, g)?)?)?;
    let t3 = mul(c, sub(mul(d, h)?, mul(e, g)?)?)?;
    add(sub(t1, t2)?, t3)
}

/// `(a, b, c)` scaled by `scale` for the sums `[n, Sx, Sy, Sxx, Sxy, Syy, Sv, Sxv, Syv]`
fn solve_plane(s: &[i128; 9], scale: i128) -> Result<(i64, i64, i64), Int2dError> {
    let [n, sx, sy, sxx, sxy, syy, sv, sxv, syv] = *s;
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let rhs = [sxv, syv, sv];
    let mut det = det3(m)?;
    if det == 0 {
        return solve_degenerate(s, scale);
    }
    let mut cols = [0_i128; 3];
    for (col, out) in cols.iter_mut().enumerate() {
        let mut mc = m;
        for row in 0..3 {
            mc[row][col] = rhs[row];
        }
        *out = det3(mc)?;
    }
    if det < 0 {
        det = det.checked_neg().ok_or(Int2dError::Overflow)?;
        for c in &mut cols {
            *c = c.checked_neg().ok_or(Int2dError::Overflow)?;
        }
    }
    let a = round_half_up(mul(cols[0], scale)?, det)?;
    let b = round_half_up(mul(cols[1], scale)?, det)?;
    let c = round_half_up(mul(cols[2], scale)?, det)?;
    Ok((to_i64(a)?, to_i64(b)?, to_i64(c)?))
}

/// The fit of a tile whose points do not determine a plane: a line in `y` (`a = 0`) if the
/// points have at least 2 distinct `y`, else a line in `x` (`b = 0`), else the mean. Points on
/// one line that is not parallel to an axis are reproduced exactly by the line in `y`.
fn solve_degenerate(s: &[i128; 9], scale: i128) -> Result<(i64, i64, i64), Int2dError> {
    let [n, sx, sy, sxx, _sxy, syy, sv, sxv, syv] = *s;
    for (s1, s11, s1v, axis_y) in [(sy, syy, syv, true), (sx, sxx, sxv, false)] {
        // [[s11, s1], [s1, n]] (slope, c) = [s1v, sv]
        let mut det = sub(mul(s11, n)?, mul(s1, s1)?)?;
        if det == 0 {
            continue;
        }
        let mut slope = sub(mul(s1v, n)?, mul(s1, sv)?)?;
        let mut c = sub(mul(s11, sv)?, mul(s1, s1v)?)?;
        if det < 0 {
            det = det.checked_neg().ok_or(Int2dError::Overflow)?;
            slope = slope.checked_neg().ok_or(Int2dError::Overflow)?;
            c = c.checked_neg().ok_or(Int2dError::Overflow)?;
        }
        let slope = to_i64(round_half_up(mul(slope, scale)?, det)?)?;
        let c = to_i64(round_half_up(mul(c, scale)?, det)?)?;
        return Ok(if axis_y { (0, slope, c) } else { (slope, 0, c) });
    }
    let c = round_half_up(mul(sv, scale)?, n)?;
    Ok((0, 0, to_i64(c)?))
}

impl IntLaw2 for PlanePerTile {
    fn predict(&self, i: i64, j: i64) -> Result<i64, Int2dError> {
        let (key, x, y) = tile_of(self.i0, self.j0, i64::from(self.tile), i, j)?;
        let &(a, b, c) = self
            .coeffs
            .get(&key)
            .ok_or(Int2dError::OutsideLaw { cell: (i, j) })?;
        let num = add(
            add(
                mul(i128::from(a), i128::from(x))?,
                mul(i128::from(b), i128::from(y))?,
            )?,
            i128::from(c),
        )?;
        to_i64(round_half_up(num, 1_i128 << self.k)?)
    }

    fn law_id(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        update_length_prefixed(&mut h, LAW_ID_DOMAIN);
        update_length_prefixed(&mut h, PLANE_PER_TILE_KIND);
        h.update(self.i0.to_be_bytes());
        h.update(self.j0.to_be_bytes());
        h.update(self.tile.to_be_bytes());
        h.update([self.k]);
        h.update(len_u64(self.coeffs.len()).to_be_bytes());
        for (&(ti, tj), &(a, b, c)) in &self.coeffs {
            for x in [ti, tj, a, b, c] {
                h.update(x.to_be_bytes());
            }
        }
        h.finalize().into()
    }
}

// ------------------------------------------------------------------ residuals

/// How the residuals are stored
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Exactly
    Lossless,
    /// Rounded to multiples of `q` (`r' = q * round_half_up(r / q)`), so every value comes
    /// back within `floor(q / 2)`; `q = 1` is lossless
    Bounded {
        /// the step, at least 1
        q: u64,
    },
}

/// `value - prediction` for each cell, in the domain's order
///
/// # Errors
///
/// [`Int2dError::LengthMismatch`], and the law's errors and [`Int2dError::Overflow`]
pub fn residuals(
    domain: &Domain,
    values: &[i64],
    law: &dyn IntLaw2,
) -> Result<Vec<i64>, Int2dError> {
    let n = domain.len();
    if n == 0 {
        return Err(Int2dError::EmptyDomain);
    }
    if n != values.len() {
        return Err(Int2dError::LengthMismatch {
            cells: n,
            values: values.len(),
        });
    }
    domain
        .cells()
        .zip(values)
        .map(|((i, j), &v)| {
            v.checked_sub(law.predict(i, j)?)
                .ok_or(Int2dError::Overflow)
        })
        .collect()
}

#[cfg(feature = "std")]
fn zigzag(x: i64) -> u64 {
    // the standard zigzag map: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ...
    u64::from_ne_bytes(((x << 1) ^ (x >> 63)).to_ne_bytes())
}

#[cfg(feature = "std")]
fn unzigzag(u: u64) -> i64 {
    let half = i64::from_ne_bytes((u >> 1).to_ne_bytes());
    let sign = i64::from_ne_bytes((u & 1).to_ne_bytes());
    half ^ -sign
}

#[cfg(feature = "std")]
/// the low 7 bits of `u`
fn low7(u: u64) -> u8 {
    u.to_le_bytes()[0] & 0x7f
}

#[cfg(feature = "std")]
fn put_varint(out: &mut Vec<u8>, mut u: u64) {
    while u >= 0x80 {
        out.push(low7(u) | 0x80);
        u >>= 7;
    }
    out.push(low7(u));
}

#[cfg(feature = "std")]
fn get_varint(b: &[u8], at: &mut usize) -> Result<u64, Int2dError> {
    let mut u = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *b
            .get(*at)
            .ok_or(Int2dError::BadPayload("truncated varint"))?;
        *at += 1;
        let part = u64::from(byte & 0x7f);
        if shift == 63 && part > 1 {
            return Err(Int2dError::BadPayload("varint above 64 bits"));
        }
        u |= part << shift;
        if byte & 0x80 == 0 {
            if byte == 0 && shift > 0 {
                return Err(Int2dError::BadPayload("non-minimal varint"));
            }
            return Ok(u);
        }
    }
    Err(Int2dError::BadPayload("varint above 64 bits"))
}

#[cfg(feature = "std")]
/// SHA-256 of the count and the values, big-endian
fn values_digest(values: &[i64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(len_u64(values.len()).to_be_bytes());
    for v in values {
        h.update(v.to_be_bytes());
    }
    h.finalize().into()
}

// ------------------------------------------------------------------ archive

/// Magic of the archive
pub const MAGIC: [u8; 4] = *b"ALI2";
/// Version of the archive this module writes and reads
pub const VERSION: u16 = 1;
#[cfg(feature = "std")]
const MODE_LOSSLESS: u8 = 0;
#[cfg(feature = "std")]
const MODE_BOUNDED: u8 = 1;
#[cfg(feature = "std")]
const CODEC_DEFLATE: u8 = 0;
#[cfg(feature = "std")]
const CODEC_RAW: u8 = 1;

/// How the residual stream is stored in the archive
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Codec {
    /// zlib, level 6 (the default)
    #[default]
    Deflate,
    /// The zigzag + LEB128 varints as they are, for a caller that compresses the archive (or
    /// compares backends) itself
    Raw,
}
#[cfg(feature = "std")]
/// magic, version, law_id, domain digest, mode, q, count, max error, 2 digests, codec, length
const HEADER_LEN: usize = 4 + 2 + 32 + 32 + 1 + 8 + 8 + 8 + 32 + 32 + 1 + 8;
#[cfg(feature = "std")]
/// the SHA-256 of all the bytes before it, at the end of the archive
const TRAILER_LEN: usize = 32;

/// An encoded archive and the largest error it carries
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// the archive
    pub bytes: Vec<u8>,
    /// the largest `|value - decoded|` over all cells (0 for [`Mode::Lossless`]; at most
    /// `floor(q / 2)` for [`Mode::Bounded`])
    pub max_abs_error: u64,
}

/// Encode `values` on `domain` as residuals against `law`
///
/// The header holds the magic and version, the law identifier, the domain digest, the mode,
/// `q`, the number of values, the largest error, the SHA-256 of the original values and of
/// the values a decoder reconstructs (equal for [`Mode::Lossless`]), the codec and the
/// payload length. The payload is the residuals (the quotients `round_half_up(r / q)` for
/// [`Mode::Bounded`]) as zigzag + LEB128 varints, deflated (zlib, level 6). The archive ends
/// with the SHA-256 of everything before it. `Bounded { q: 1 }` is written as lossless.
///
/// # Errors
///
/// [`Int2dError::EmptyDomain`], [`Int2dError::LengthMismatch`],
/// [`Int2dError::InvalidParameter`] for `q = 0`, the law's errors, [`Int2dError::Overflow`],
/// [`Int2dError::Backend`]
#[cfg(feature = "std")]
pub fn encode(
    domain: &Domain,
    values: &[i64],
    law: &dyn IntLaw2,
    mode: Mode,
) -> Result<Encoded, Int2dError> {
    encode_with(domain, values, law, mode, Codec::Deflate)
}

/// [`encode`] with the residual stream stored as `codec` says ([`Codec::Raw`] leaves the
/// varints uncompressed; the header, the digests and the trailer are the same)
///
/// # Errors
///
/// As [`encode`]
#[cfg(feature = "std")]
pub fn encode_with(
    domain: &Domain,
    values: &[i64],
    law: &dyn IntLaw2,
    mode: Mode,
    codec: Codec,
) -> Result<Encoded, Int2dError> {
    let r = residuals(domain, values, law)?;
    let (mode_byte, q) = match mode {
        // q = 1 keeps every value, so it is written as the lossless mode (one encoding each)
        Mode::Lossless | Mode::Bounded { q: 1 } => (MODE_LOSSLESS, 1_u64),
        Mode::Bounded { q: 0 } => return Err(Int2dError::InvalidParameter("q must be at least 1")),
        Mode::Bounded { q } => (MODE_BOUNDED, q),
    };
    let qi = i128::from(q);
    let mut raw = Vec::with_capacity(r.len() * 2);
    let mut rebuilt = Vec::with_capacity(r.len());
    let mut max_err = 0_u64;
    for ((&res, &v), (i, j)) in r.iter().zip(values).zip(domain.cells()) {
        let quot = to_i64(round_half_up(i128::from(res), qi)?)?;
        put_varint(&mut raw, zigzag(quot));
        let back = add(i128::from(law.predict(i, j)?), mul(i128::from(quot), qi)?)?;
        let back = to_i64(back)?;
        let err = (i128::from(v) - i128::from(back)).unsigned_abs();
        max_err = max_err.max(u64::try_from(err).map_err(|_| Int2dError::Overflow)?);
        rebuilt.push(back);
    }
    let (codec_byte, payload) = match codec {
        Codec::Deflate => (
            CODEC_DEFLATE,
            crate::compression::zlib_compress(&raw, 6)
                .map_err(|e| Int2dError::Backend(e.to_string()))?,
        ),
        Codec::Raw => (CODEC_RAW, raw),
    };
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_be_bytes());
    bytes.extend_from_slice(&law.law_id());
    bytes.extend_from_slice(&domain.digest());
    bytes.push(mode_byte);
    bytes.extend_from_slice(&q.to_be_bytes());
    bytes.extend_from_slice(&len_u64(values.len()).to_be_bytes());
    bytes.extend_from_slice(&max_err.to_be_bytes());
    bytes.extend_from_slice(&values_digest(values));
    bytes.extend_from_slice(&values_digest(&rebuilt));
    bytes.push(codec_byte);
    bytes.extend_from_slice(&len_u64(payload.len()).to_be_bytes());
    bytes.extend_from_slice(&payload);
    let check: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&check);
    Ok(Encoded {
        bytes,
        max_abs_error: max_err,
    })
}

#[cfg(feature = "std")]
fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], Int2dError> {
    let end = at.checked_add(n).ok_or(Int2dError::BadHeader("length"))?;
    let s = b
        .get(*at..end)
        .ok_or(Int2dError::BadHeader("truncated header"))?;
    *at = end;
    Ok(s)
}

#[cfg(feature = "std")]
fn take_u64(b: &[u8], at: &mut usize) -> Result<u64, Int2dError> {
    let s = take(b, at, 8)?;
    let mut a = [0_u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_be_bytes(a))
}

/// zlib-inflate `payload`, stopping after `limit` bytes of output (so a small payload cannot
/// expand without bound in memory)
#[cfg(feature = "std")]
fn inflate_bounded(payload: &[u8], limit: u64) -> Result<Vec<u8>, Int2dError> {
    use std::io::Read;
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(payload)
        .take(limit)
        .read_to_end(&mut raw)
        .map_err(|e| Int2dError::Backend(e.to_string()))?;
    Ok(raw)
}

/// Decode an archive written by [`encode`] for `domain` with `law`
///
/// Checks the magic and version, the SHA-256 at the end of the archive, that the law
/// identifier and the domain digest are those of
/// `law` and `domain`, the number of values, the payload length, that the residual stream
/// holds exactly that many varints, and the SHA-256 of the reconstructed values (and, for
/// [`Mode::Lossless`], that it equals the digest of the original values).
///
/// # Errors
///
/// [`Int2dError::BadHeader`], [`Int2dError::LawIdMismatch`], [`Int2dError::DomainMismatch`],
/// [`Int2dError::BadPayload`], [`Int2dError::HashMismatch`], [`Int2dError::Backend`], the
/// law's errors and [`Int2dError::Overflow`]
#[cfg(feature = "std")]
pub fn decode(bytes: &[u8], domain: &Domain, law: &dyn IntLaw2) -> Result<Vec<i64>, Int2dError> {
    let mut at = 0;
    if take(bytes, &mut at, 4)? != MAGIC {
        return Err(Int2dError::BadHeader("magic"));
    }
    let v = take(bytes, &mut at, 2)?;
    if u16::from_be_bytes([v[0], v[1]]) != VERSION {
        return Err(Int2dError::BadHeader("version"));
    }
    let body_len = bytes
        .len()
        .checked_sub(TRAILER_LEN)
        .filter(|&n| n >= HEADER_LEN)
        .ok_or(Int2dError::BadHeader("truncated archive"))?;
    let (body, trailer) = bytes.split_at(body_len);
    if Sha256::digest(body).as_slice() != trailer {
        return Err(Int2dError::HashMismatch);
    }
    let bytes = body;
    if take(bytes, &mut at, 32)? != law.law_id() {
        return Err(Int2dError::LawIdMismatch);
    }
    if take(bytes, &mut at, 32)? != domain.digest() {
        return Err(Int2dError::DomainMismatch);
    }
    let mode = take(bytes, &mut at, 1)?[0];
    let q = take_u64(bytes, &mut at)?;
    match (mode, q) {
        (MODE_LOSSLESS, 1) => {}
        (MODE_BOUNDED, q) if q >= 2 => {}
        _ => return Err(Int2dError::BadHeader("mode")),
    }
    let count = take_u64(bytes, &mut at)?;
    let n = domain.len();
    if count != len_u64(n) {
        return Err(Int2dError::BadHeader("count"));
    }
    let max_err = take_u64(bytes, &mut at)?;
    // lossless is q = 1, so this also refuses a non-zero error in a lossless archive
    if max_err > q / 2 {
        return Err(Int2dError::BadHeader("error bound"));
    }
    let original = take(bytes, &mut at, 32)?.to_vec();
    let rebuilt_digest = take(bytes, &mut at, 32)?.to_vec();
    if mode == MODE_LOSSLESS && original != rebuilt_digest {
        return Err(Int2dError::BadHeader("lossless digests differ"));
    }
    let codec = take(bytes, &mut at, 1)?[0];
    if codec != CODEC_DEFLATE && codec != CODEC_RAW {
        return Err(Int2dError::BadHeader("codec"));
    }
    let plen =
        usize::try_from(take_u64(bytes, &mut at)?).map_err(|_| Int2dError::BadHeader("length"))?;
    let payload = take(bytes, &mut at, plen)?;
    if at != bytes.len() {
        return Err(Int2dError::BadHeader("trailing bytes"));
    }
    // a varint is at most 10 bytes: inflate no more than that per value, +1 to see an excess
    let limit = len_u64(n).saturating_mul(10).saturating_add(1);
    let raw = if codec == CODEC_RAW {
        payload.to_vec()
    } else {
        inflate_bounded(payload, limit)?
    };
    if len_u64(raw.len()) >= limit {
        return Err(Int2dError::BadPayload("payload expands past the count"));
    }
    let qi = i128::from(q);
    let mut out = Vec::with_capacity(n);
    let mut p = 0;
    for (i, j) in domain.cells() {
        let quot = unzigzag(get_varint(&raw, &mut p)?);
        let back = add(i128::from(law.predict(i, j)?), mul(i128::from(quot), qi)?)?;
        out.push(to_i64(back)?);
    }
    if p != raw.len() {
        return Err(Int2dError::BadPayload("more residuals than cells"));
    }
    if values_digest(&out).as_slice() != rebuilt_digest.as_slice() {
        return Err(Int2dError::HashMismatch);
    }
    Ok(out)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// the inflate step stops at the cap: 1 MiB of zeros deflates to about 1 KiB, and with a
    /// cap of 11 bytes no more than 11 come out
    #[test]
    fn inflate_stops_at_the_limit() {
        let big = alloc::vec![0_u8; 1 << 20];
        let payload = crate::compression::zlib_compress(&big, 6).unwrap();
        assert!(payload.len() < 4096);
        let out = inflate_bounded(&payload, 11).unwrap();
        assert_eq!(out.len(), 11);
        assert_eq!(inflate_bounded(&payload, 1 << 21).unwrap().len(), 1 << 20);
    }
}
