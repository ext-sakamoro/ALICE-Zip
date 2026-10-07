//! A fitted law kept together with what it was fitted from and where it holds
//!
//! The generators in [`crate::generators`] store *how to regenerate* a signal
//! instead of its samples. This module adds what is needed to treat such a
//! function as a claim about data rather than as a compressed copy of it:
//!
//! | item | meaning |
//! |------|---------|
//! | [`SignalLaw`] | `y = f(x)` (a polynomial of fixed degree) with its parameters |
//! | [`ValidRange`] | the closed `x` interval the evidence covers; evaluation outside it is refused |
//! | [`ResidualStats`] | measured `y - f(x)` over the evidence (not a quantity reported by the fit) |
//! | [`Provenance`] | where the evidence came from and how the law was obtained |
//! | [`OracleCase`] | a reference value the law has to reproduce within a tolerance |
//! | [`SignalLawParts`] | every field public, for storing a law elsewhere ([`SignalLaw::to_parts`] / [`SignalLaw::from_parts`]) |
//! | [`Verdict`] | what new evidence does to the law: supports it, updates its parameters, grows its residual, breaks it, or lies outside its range |
//!
//! Evaluation never extrapolates: [`SignalLaw::evaluate`] returns
//! [`LawError::OutOfRange`] for an `x` outside [`SignalLaw::domain`] (or not
//! finite) instead of a value.
//!
//! # Judging new evidence
//!
//! [`SignalLaw::ingest`] compares the evidence with the law and returns a
//! [`Verdict`]; it does not change the law. With
//! `band = max(policy.abs_tolerance, residual().rms)` and `rms_new` the RMS of
//! `y - f(x)` over the new points:
//!
//! 1. no points → [`Verdict::NoEvidence`]
//! 2. any point outside the valid range (or not finite) → [`Verdict::OutOfRange`]
//! 3. `rms_new ≤ band` → [`Verdict::Supports`]
//! 4. a refit of the same form on old + new evidence has RMS `≤ band` →
//!    [`Verdict::ParameterUpdate`] (carries the refitted law)
//! 5. `rms_new ≤ band · policy.break_factor` → [`Verdict::ResidualGrew`]
//! 6. otherwise → [`Verdict::Breaks`]
//!
//! ```
//! use alice_zip::law::{IngestPolicy, Provenance, SignalLaw, Verdict};
//!
//! // y = 1 + 2x measured at five points
//! let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 1.0 + 2.0 * i as f64)).collect();
//! let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("bench run 1", "least squares"))?;
//! assert!((law.evaluate(2.5)? - 6.0).abs() < 1e-12); // a condition that was not measured
//! assert!(law.evaluate(9.0).is_err());               // outside the measured range
//!
//! let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
//! let more = [(0.5, 2.0), (3.5, 8.0)];
//! assert!(matches!(law.ingest(&more, &policy), Verdict::Supports { .. }));
//! # Ok::<(), alice_zip::law::LawError>(())
//! ```

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::generators::polynomial::{horner_ascending, least_squares_fit};

// (test builds link std, whose inherent methods shadow the trait → allow)
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::math::FloatExt;

/// Why a law could not be fitted or evaluated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LawError {
    /// Fewer points than parameters (`degree + 1`)
    TooFewPoints,
    /// All evidence has the same `x`, so there is no range to fit over
    DegenerateDomain,
    /// An `x` or `y` of the evidence is NaN or infinite
    NonFinite,
    /// The evidence does not determine the parameters (repeated `x`)
    RankDeficient,
    /// The requested `x` is outside the valid range, or not finite
    OutOfRange,
}

impl core::fmt::Display for LawError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::TooFewPoints => "fewer points than parameters",
            Self::DegenerateDomain => "evidence covers a single x",
            Self::NonFinite => "evidence contains a non-finite value",
            Self::RankDeficient => "evidence does not determine the parameters",
            Self::OutOfRange => "x is outside the valid range",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LawError {}

/// Closed interval `[lo, hi]` of `x` covered by the evidence
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidRange {
    /// Smallest `x` of the evidence
    pub lo: f64,
    /// Largest `x` of the evidence
    pub hi: f64,
}

impl ValidRange {
    /// Whether `x` lies in the closed interval (`false` for NaN)
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }
}

/// `y - f(x)` measured over a set of points
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualStats {
    /// Number of points
    pub n: usize,
    /// Root mean square of `y - f(x)`
    pub rms: f64,
    /// Largest `|y - f(x)|`
    pub max_abs: f64,
}

/// Where the evidence came from and how the law was obtained
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Source of the evidence (a reference, a dataset id, a run)
    pub source: String,
    /// How the law was obtained from it
    pub method: String,
}

impl Provenance {
    /// Provenance from a source and a method description
    #[must_use]
    pub fn new(source: &str, method: &str) -> Self {
        Self {
            source: String::from(source),
            method: String::from(method),
        }
    }
}

/// A reference value the law has to reproduce
#[derive(Debug, Clone, PartialEq)]
pub struct OracleCase {
    /// Condition
    pub x: f64,
    /// Reference value at `x`
    pub expected: f64,
    /// Largest accepted `|f(x) - expected|`
    pub tolerance: f64,
    /// Where the reference value comes from
    pub source: String,
}

impl OracleCase {
    /// A reference value with its tolerance and source
    #[must_use]
    pub fn new(x: f64, expected: f64, tolerance: f64, source: &str) -> Self {
        Self {
            x,
            expected,
            tolerance,
            source: String::from(source),
        }
    }
}

/// Result of checking one [`OracleCase`]
#[derive(Debug, Clone, PartialEq)]
pub struct OracleOutcome {
    /// Value of the law at the case's `x`, or why there is none
    pub value: Result<f64, LawError>,
    /// `|value - expected|` when the law has a value
    pub error: Option<f64>,
    /// `error ≤ tolerance` (false when the law has no value)
    pub passed: bool,
}

/// Thresholds used by [`SignalLaw::ingest`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IngestPolicy {
    /// Smallest RMS deviation regarded as disagreement (measurement precision)
    pub abs_tolerance: f64,
    /// Deviation up to `band · break_factor` is a grown residual, beyond it the law breaks
    pub break_factor: f64,
}

/// What new evidence does to a law (see the module docs for the rules)
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// No evidence was given
    NoEvidence,
    /// `outside` points lie outside the valid range; nothing was judged
    OutOfRange {
        /// Number of points outside the range (or not finite)
        outside: usize,
    },
    /// The evidence agrees with the law
    Supports {
        /// RMS of the new points about the law
        rms: f64,
    },
    /// The same form fits old and new evidence with other parameters
    ParameterUpdate {
        /// RMS of the new points about the current law
        previous_rms: f64,
        /// The law refitted on old + new evidence
        updated: Box<SignalLaw>,
    },
    /// The deviation exceeds the band but not the break threshold
    ResidualGrew {
        /// RMS of the new points about the law
        rms: f64,
    },
    /// The evidence is not described by this form
    Breaks {
        /// RMS of the new points about the law
        rms: f64,
    },
}

/// The stored form of a [`SignalLaw`]: every field public, for persisting the
/// law in another format and restoring it with [`SignalLaw::from_parts`]
#[derive(Debug, Clone, PartialEq)]
pub struct SignalLawParts {
    /// Ascending coefficients in `u = (x - lo) / (hi - lo)`
    pub coefficients: Vec<f64>,
    /// Valid range
    pub domain: ValidRange,
    /// Evidence the law stands on
    pub evidence: Vec<(f64, f64)>,
    /// Residual as stored; [`SignalLaw::from_parts`] re-measures it from the evidence
    pub residual: ResidualStats,
    /// Source and method
    pub provenance: Provenance,
    /// Reference values
    pub oracles: Vec<OracleCase>,
}

/// `y = f(x)` as a polynomial of fixed degree, with its evidence, residual,
/// valid range, provenance and oracle cases
#[derive(Debug, Clone, PartialEq)]
pub struct SignalLaw {
    /// Ascending coefficients in the normalised variable `u = (x - lo) / (hi - lo)`
    coeffs: Vec<f64>,
    domain: ValidRange,
    evidence: Vec<(f64, f64)>,
    residual: ResidualStats,
    provenance: Provenance,
    oracles: Vec<OracleCase>,
}

impl SignalLaw {
    /// Least-squares polynomial of `degree` through `points` (`(x, y)` pairs)
    ///
    /// The fit is done in `u = (x - lo) / (hi - lo) ∈ [0, 1]`, so the
    /// conditioning does not depend on where the evidence lies on the `x`
    /// axis. The residual is measured afterwards over the same points.
    ///
    /// # Errors
    ///
    /// [`LawError::TooFewPoints`] / [`LawError::NonFinite`] /
    /// [`LawError::DegenerateDomain`] / [`LawError::RankDeficient`]
    pub fn fit_polynomial(
        points: &[(f64, f64)],
        degree: usize,
        provenance: Provenance,
    ) -> Result<Self, LawError> {
        if points.len() < degree + 1 || points.is_empty() {
            return Err(LawError::TooFewPoints);
        }
        if points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(LawError::NonFinite);
        }
        let lo = points.iter().fold(f64::INFINITY, |a, p| a.min(p.0));
        let hi = points.iter().fold(f64::NEG_INFINITY, |a, p| a.max(p.0));
        if hi <= lo {
            return Err(LawError::DegenerateDomain);
        }
        let span = hi - lo;
        let us: Vec<f64> = points.iter().map(|p| (p.0 - lo) / span).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        let coeffs = least_squares_fit(&us, &ys, degree).ok_or(LawError::RankDeficient)?;
        let mut law = Self {
            coeffs,
            domain: ValidRange { lo, hi },
            evidence: points.to_vec(),
            residual: ResidualStats {
                n: 0,
                rms: 0.0,
                max_abs: 0.0,
            },
            provenance,
            oracles: Vec::new(),
        };
        law.residual = law.residual_over(points);
        Ok(law)
    }

    /// Adds a reference value the law has to reproduce
    #[must_use]
    pub fn with_oracle(mut self, case: OracleCase) -> Self {
        self.oracles.push(case);
        self
    }

    /// Ascending coefficients in the normalised variable `u = (x - lo) / (hi - lo)`
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    /// The law as plain parts, for storing it elsewhere
    #[must_use]
    pub fn to_parts(&self) -> SignalLawParts {
        SignalLawParts {
            coefficients: self.coeffs.clone(),
            domain: self.domain,
            evidence: self.evidence.clone(),
            residual: self.residual,
            provenance: self.provenance.clone(),
            oracles: self.oracles.clone(),
        }
    }

    /// Restores a law from stored parts without refitting it
    ///
    /// The coefficients are taken as stored; the residual is measured again
    /// over the stored evidence (a stored residual is not trusted).
    ///
    /// # Errors
    ///
    /// [`LawError::TooFewPoints`] for no coefficients, no evidence or fewer
    /// points than coefficients; [`LawError::NonFinite`] for a non-finite
    /// coefficient, bound or evidence value; [`LawError::DegenerateDomain`]
    /// for `hi <= lo`; [`LawError::OutOfRange`] for evidence outside the domain
    pub fn from_parts(parts: SignalLawParts) -> Result<Self, LawError> {
        let SignalLawParts {
            coefficients,
            domain,
            evidence,
            provenance,
            oracles,
            ..
        } = parts;
        if coefficients.is_empty() || evidence.is_empty() || evidence.len() < coefficients.len() {
            return Err(LawError::TooFewPoints);
        }
        if coefficients.iter().any(|c| !c.is_finite())
            || !domain.lo.is_finite()
            || !domain.hi.is_finite()
            || evidence
                .iter()
                .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err(LawError::NonFinite);
        }
        if domain.hi <= domain.lo {
            return Err(LawError::DegenerateDomain);
        }
        if evidence.iter().any(|(x, _)| !domain.contains(*x)) {
            return Err(LawError::OutOfRange);
        }
        let mut law = Self {
            coeffs: coefficients,
            domain,
            evidence,
            residual: ResidualStats {
                n: 0,
                rms: 0.0,
                max_abs: 0.0,
            },
            provenance,
            oracles,
        };
        law.residual = law.residual_over(&law.evidence);
        Ok(law)
    }

    /// Degree of the polynomial
    #[must_use]
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// The closed `x` interval covered by the evidence
    #[must_use]
    pub fn domain(&self) -> ValidRange {
        self.domain
    }

    /// Residual measured over the evidence the law was fitted to
    #[must_use]
    pub fn residual(&self) -> ResidualStats {
        self.residual
    }

    /// Source and method
    #[must_use]
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The `(x, y)` points the law was fitted to
    #[must_use]
    pub fn evidence(&self) -> &[(f64, f64)] {
        &self.evidence
    }

    /// Oracle cases attached with [`Self::with_oracle`]
    #[must_use]
    pub fn oracles(&self) -> &[OracleCase] {
        &self.oracles
    }

    /// `f(x)` for `x` inside the valid range
    ///
    /// # Errors
    ///
    /// [`LawError::OutOfRange`] when `x` is outside [`Self::domain`] or not finite
    pub fn evaluate(&self, x: f64) -> Result<f64, LawError> {
        if !self.domain.contains(x) {
            return Err(LawError::OutOfRange);
        }
        Ok(self.eval_unchecked(x))
    }

    fn eval_unchecked(&self, x: f64) -> f64 {
        let u = (x - self.domain.lo) / (self.domain.hi - self.domain.lo);
        horner_ascending(&self.coeffs, u)
    }

    /// `y - f(x)` over `points`, all of which must lie in the valid range
    fn residual_over(&self, points: &[(f64, f64)]) -> ResidualStats {
        let mut sum_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for &(x, y) in points {
            let d = y - self.eval_unchecked(x);
            sum_sq += d * d;
            max_abs = max_abs.max(d.abs());
        }
        #[allow(clippy::cast_precision_loss)]
        let rms = if points.is_empty() {
            0.0
        } else {
            (sum_sq / points.len() as f64).sqrt()
        };
        ResidualStats {
            n: points.len(),
            rms,
            max_abs,
        }
    }

    /// Checks every attached oracle case
    #[must_use]
    pub fn check_oracles(&self) -> Vec<OracleOutcome> {
        self.oracles
            .iter()
            .map(|case| {
                let value = self.evaluate(case.x);
                let error = value.as_ref().ok().map(|v| (v - case.expected).abs());
                OracleOutcome {
                    value,
                    error,
                    passed: error.is_some_and(|e| e <= case.tolerance),
                }
            })
            .collect()
    }

    /// Judges new evidence against the law without changing it
    ///
    /// See the module documentation for the order of the rules.
    #[must_use]
    pub fn ingest(&self, points: &[(f64, f64)], policy: &IngestPolicy) -> Verdict {
        if points.is_empty() {
            return Verdict::NoEvidence;
        }
        let outside = points
            .iter()
            .filter(|(x, y)| !self.domain.contains(*x) || !y.is_finite())
            .count();
        if outside > 0 {
            return Verdict::OutOfRange { outside };
        }
        let band = policy.abs_tolerance.max(self.residual.rms);
        let rms = self.residual_over(points).rms;
        if rms <= band {
            return Verdict::Supports { rms };
        }
        let mut combined = self.evidence.clone();
        combined.extend_from_slice(points);
        if let Ok(refit) = Self::fit_polynomial(&combined, self.degree(), self.provenance.clone()) {
            if refit.residual.rms <= band {
                let updated = Self {
                    oracles: self.oracles.clone(),
                    ..refit
                };
                return Verdict::ParameterUpdate {
                    previous_rms: rms,
                    updated: Box::new(updated),
                };
            }
        }
        if rms <= band * policy.break_factor {
            Verdict::ResidualGrew { rms }
        } else {
            Verdict::Breaks { rms }
        }
    }
}
