"""
null_calibration.py
====================

Implements Phase I of Strategy 8-U-NC (strategy8_nc.pdf), Sections 3.2-3.7:
the null-calibrated conformal sorter. This is the genuinely new statistical
core that replaces Strategy 8-U's 2-component GMM (gmm.py) -- Phase II
(prompts.py, tristate_logits.py) is UNCHANGED and is reused as-is (the spec
says so explicitly in its own Section 4: "unchanged in structure from
Strategy 8-U ... except that O_pos and O_neg now arise from the
null-calibrated conformal sorter").

Given a per-image candidate pool O_init (candidate_pool.py / synonyms.py --
also unchanged) and a freshly-sampled, per-image probe pool P
(probe_sampling.py + build_probe_pool.py, both new), this module:

  1. builds the raw 3D evidence vector r_w = [s_det(w), s_clip(w),
     sqrt(s_area(w))] for every w in O_init union P (Eq. 4) -- feature
     EXTRACTION itself (OWL-ViT / CLIP forward passes) is unchanged from
     Strategy 8-U (feature_extractors.py::FeatureExtractor); this module
     only adds the sqrt(s_area) variance-stabilizing transform Eq. 4
     specifies, applied identically to candidates and probes.
  2. standardizes every vector using PROBE-ONLY mean/std (Eq. 6-7) --
     deliberately never uses candidate statistics for normalization, and
     deliberately recomputed fresh for every single image (no cross-image
     state at all, unlike fit_gmm.py's FeatureScaler, which pools many
     images). See Section 6, point 1 ("No cross-image parameters").
  3. fits a one-class multivariate Gaussian null model (mu_0, Sigma_0) from
     the normalized PROBE vectors only, with Ledoit-Wolf-style shrinkage
     toward a scaled identity (Eq. 8-9).
  4. scores every candidate AND every probe (the latter forms the
     conformal reference population) by its SIGNED Mahalanobis distance to
     the null, using the one-sided projection test along
     u = [1,1,1]/sqrt(3) (Eq. 10-11).
  5. converts each candidate's signed distance into a distribution-free
     conformal p-value by ranking it against the probes' own distances
     (Eq. 12).
  6. hard-splits O_init into O_pos / O_neg at a nominal false-verification
     rate epsilon (Eq. 13-14).

Per-image, not pooled: unlike gmm.py's GlobalGMM (which had to pool many
images' candidates to get a stable fit, since a single image's candidate
pool is far too small -- 10-25 points -- to fit a full 2-component 3D
covariance), the null model here only ever needs the probe pool, whose
size K is a free, image-independent design choice (K in [50, 100] is
plenty for a stable 3D one-class covariance with shrinkage). So there is
no small-sample problem motivating a pooled/global fit here, and the
paper's per-image design (Section 3.4: "it is discarded once the image has
been processed and is never reused for ... any other image") is followed
literally, rather than the deviation gmm.py had to make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Eq. 11's fixed evidence direction: u = (1/sqrt(3)) * [1, 1, 1]^T, the
# direction of simultaneously increasing s_det, s_clip, and (sqrt-)s_area.
_U_DIRECTION = np.ones(3) / np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Eq. 4: raw evidence vector (sqrt transform on s_area only)
# ---------------------------------------------------------------------------
def raw_feature_vector(s_det: float, s_clip: float, s_area: float) -> np.ndarray:
    """Eq. 4: r_w = [s_det(w), s_clip(w), sqrt(s_area(w))]^T.

    The square root is applied to s_area ONLY (Section 3.2's prose is
    explicit: "the feature used is its square root, sqrt(s_area(w)),
    applied as a variance-stabilizing transform"); s_det and s_clip are
    used as-is. Negative/zero areas (objects with no bounding box, e.g.
    VLM-only mentions per Section 3.2's convention) map to sqrt(0) = 0.
    """
    return np.array([float(s_det), float(s_clip), np.sqrt(max(float(s_area), 0.0))])


def raw_feature_matrix(feature_triples: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    """Vectorized form of raw_feature_vector for a list of (s_det, s_clip, s_area)
    raw (pre-sqrt) triples, in the SAME order as the caller's word list."""
    if len(feature_triples) == 0:
        return np.zeros((0, 3))
    arr = np.array(feature_triples, dtype=float)
    arr[:, 2] = np.sqrt(np.maximum(arr[:, 2], 0.0))
    return arr


# ---------------------------------------------------------------------------
# Eq. 6-7: probe-only standardization, fresh per image
# ---------------------------------------------------------------------------
@dataclass
class ProbeNormalizer:
    """Eq. 6-7: standardization statistics (mu_raw, sigma_raw) computed
    ONLY from one image's probe population, then applied to both the
    probes themselves and that same image's candidates."""

    mean: np.ndarray  # mu_raw, (3,)
    std: np.ndarray   # sigma_raw, (3,)

    @classmethod
    def fit(cls, probe_raw: np.ndarray) -> "ProbeNormalizer":
        """Eq. 6: sample mean/std over the probe population (ddof=1, as
        the paper's 1/(K-1) normalization specifies)."""
        probe_raw = np.asarray(probe_raw, dtype=float)
        if probe_raw.ndim != 2 or probe_raw.shape[1] != 3:
            raise ValueError(f"probe_raw must be (K, 3), got shape {probe_raw.shape}")
        if probe_raw.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 probes to estimate mean/std, got {probe_raw.shape[0]}"
            )
        mean = probe_raw.mean(axis=0)
        std = probe_raw.std(axis=0, ddof=1)
        # Guard a (near-)constant probe dimension (e.g. every probe has
        # s_area exactly 0) so division below never blows up; this is a
        # numerical safety net, not something the paper's idealized math
        # needs, since K probes drawn from a continuous evidence
        # distribution essentially never have exactly zero variance.
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Eq. 7: x~_w = (r_w - mu_raw) / sigma_raw, elementwise."""
        X = np.atleast_2d(np.asarray(X_raw, dtype=float))
        return (X - self.mean[None, :]) / self.std[None, :]

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeNormalizer":
        return cls(mean=np.array(d["mean"], dtype=float), std=np.array(d["std"], dtype=float))


# ---------------------------------------------------------------------------
# Eq. 8-9: one-class null model with Ledoit-Wolf-style shrinkage
# ---------------------------------------------------------------------------
def _ledoit_wolf_shrunk_covariance(
    X: np.ndarray, shrinkage: Optional[float] = None, reg_covar: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """Eq. 9: Sigma_0 = (1 - lambda) * S + lambda * (tr(S)/3) * I_3.

    `X` here is the ALREADY-NORMALIZED probe matrix (K, 3), and S is its
    sample covariance (Eq. 8, ddof=1).

    If `shrinkage` (lambda) is None, it is selected via the standard
    analytic Ledoit-Wolf formula (Ledoit & Wolf, 2004), as the paper
    allows ("or selected by the standard analytic Ledoit-Wolf formula
    computed on {x~_p}"); otherwise the given fixed intensity is used
    directly against the SAME shrinkage target form (scaled identity)
    the paper specifies. We compute S ourselves (rather than trusting
    sklearn's internal empirical covariance, which by default divides by
    K, not K-1) so the fixed-lambda path is self-contained and uses
    exactly the S defined by Eq. 8.

    `reg_covar` adds a small floor to the diagonal AFTER shrinkage, mirroring
    gmm.py's `_safe_cov` -- this guards a fully degenerate edge case the
    paper's idealized math doesn't need to worry about (K probes drawn from
    a continuous evidence distribution essentially never have EXACTLY zero
    variance in every dimension) but that a defensive implementation should
    handle anyway: if S happens to be exactly the zero matrix (e.g. a
    pathological/duplicated probe batch), tr(S)/3 is also 0, so the
    shrinkage target itself is 0 and Sigma_0 would be singular regardless
    of lambda. A single degenerate image's null model must not crash the
    whole run.
    """
    n, d = X.shape
    mean = X.mean(axis=0)
    diff = X - mean[None, :]
    S = (diff.T @ diff) / max(n - 1, 1)  # Eq. 8's sample covariance

    if shrinkage is None:
        from sklearn.covariance import ledoit_wolf

        _, lam = ledoit_wolf(X)
        lam = float(np.clip(lam, 0.0, 1.0))
    else:
        lam = float(shrinkage)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"shrinkage must be in [0, 1], got {lam}")

    target = (np.trace(S) / d) * np.eye(d)
    Sigma0 = (1.0 - lam) * S + lam * target
    Sigma0 = Sigma0 + reg_covar * np.eye(d)
    return Sigma0, lam


@dataclass
class NullModel:
    """Eq. 8-9: the one-class Gaussian null (mu_0, Sigma_0) fit from a
    SINGLE image's normalized probe vectors. Per Section 3.4: "mu_0 ~ 0 by
    construction of the standardization ... it is nonetheless recomputed
    explicitly for robustness" -- we do so here rather than assuming 0."""

    mean: np.ndarray        # mu_0, (3,)
    covariance: np.ndarray  # Sigma_0, (3, 3)
    shrinkage: float        # lambda actually used
    n_probes: int

    def __post_init__(self):
        # Cached Cholesky factor for repeated Mahalanobis-distance queries
        # (probes + candidates of the same image all reuse this). Extremely
        # defensive fallback mirroring gmm.py's own Cholesky guard: should
        # not trigger given reg_covar > 0 in _ledoit_wolf_shrunk_covariance,
        # but a single bad per-image fit must not crash a whole batch run.
        try:
            self._L = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            self.covariance = self.covariance + 1e-4 * np.eye(self.covariance.shape[0])
            self._L = np.linalg.cholesky(self.covariance)

    @classmethod
    def fit(cls, probe_normalized: np.ndarray, shrinkage: Optional[float] = None) -> "NullModel":
        probe_normalized = np.asarray(probe_normalized, dtype=float)
        n = probe_normalized.shape[0]
        if n < 4:
            raise ValueError(
                f"Need at least 4 probes to fit a stable 3D covariance "
                f"(with shrinkage this is a soft floor, not the paper's "
                f"typical K in [50, 100]), got {n}"
            )
        mean = probe_normalized.mean(axis=0)  # Eq. 8, recomputed explicitly
        Sigma0, lam = _ledoit_wolf_shrunk_covariance(probe_normalized, shrinkage=shrinkage)
        return cls(mean=mean, covariance=Sigma0, shrinkage=lam, n_probes=n)

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Eq. 10 (unsigned): d(w) = sqrt((x~_w - mu_0)^T Sigma_0^-1 (x~_w - mu_0))."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        diff = X - self.mean[None, :]
        z = np.linalg.solve(self._L, diff.T)  # L z = diff^T  =>  z = L^-1 diff^T
        maha2 = np.sum(z ** 2, axis=0)
        return np.sqrt(np.maximum(maha2, 0.0))

    def signed_distance(self, X: np.ndarray) -> np.ndarray:
        """Eq. 11: one-sided projection test.

            D(w) = d(w)   if u . (x~_w - mu_0) > 0
                 = -inf    otherwise

        using the fixed direction u = [1,1,1]/sqrt(3) (Section 3.5).
        A word with D(w) = -inf carries no evidence of presence beyond the
        null and, per Section 3.5, must always receive the maximal
        conformal p-value -- this falls out of Eq. 12 automatically with
        no special-casing needed (see conformal_p_values' docstring).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = self.mahalanobis(X)
        proj = (X - self.mean[None, :]) @ _U_DIRECTION
        return np.where(proj > 0, d, -np.inf)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "shrinkage": self.shrinkage,
            "n_probes": self.n_probes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NullModel":
        return cls(
            mean=np.array(d["mean"], dtype=float),
            covariance=np.array(d["covariance"], dtype=float),
            shrinkage=float(d["shrinkage"]),
            n_probes=int(d["n_probes"]),
        )


# ---------------------------------------------------------------------------
# Eq. 12: conformal p-values
# ---------------------------------------------------------------------------
def conformal_p_values(D_candidates: np.ndarray, D_probes: np.ndarray) -> np.ndarray:
    """Eq. 12: p(o_i) = (1 + |{p in P : D(p) >= D(o_i)}|) / (K + 1).

    Note this handles D(o_i) = -inf (Eq. 11's one-sided rejection)
    correctly with no special-casing: every probe (even one that is
    itself -inf) satisfies D(p) >= -inf, so the count is exactly K and
    p(o_i) = (K+1)/(K+1) = 1.0, the maximal p-value -- matching Section
    3.5's requirement that such words are "never verified".
    """
    D_candidates = np.atleast_1d(np.asarray(D_candidates, dtype=float))
    D_probes = np.atleast_1d(np.asarray(D_probes, dtype=float))
    K = D_probes.shape[0]
    if K == 0:
        raise ValueError("D_probes must be non-empty")
    counts = np.sum(D_probes[None, :] >= D_candidates[:, None], axis=1)
    return (1.0 + counts) / (K + 1.0)


# ---------------------------------------------------------------------------
# Full per-image Phase I result
# ---------------------------------------------------------------------------
@dataclass
class ConformalSortResult:
    """Full per-image output of Phase I (Sections 3.1-3.6): everything
    needed to reproduce O_pos / O_neg (Eq. 13-14) at ANY epsilon without
    redoing feature extraction or refitting the null model -- only
    `split(epsilon)` needs to be re-run, which is a trivial list
    comprehension over already-computed p-values."""

    candidate_names: List[str]
    candidate_p_values: List[float]
    candidate_signed_distances: List[float]
    probe_names: List[str]
    probe_signed_distances: List[float]
    null_model: NullModel
    normalizer: ProbeNormalizer

    def split(self, epsilon: float) -> Tuple[List[str], List[str]]:
        """Eq. 13-14: O_pos = {o_i in O_init : p(o_i) <= epsilon}, O_neg =
        O_init \\ O_pos."""
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        pos = [n for n, p in zip(self.candidate_names, self.candidate_p_values) if p <= epsilon]
        neg = [n for n, p in zip(self.candidate_names, self.candidate_p_values) if p > epsilon]
        return pos, neg

    def p_value_of(self, candidate_name: str) -> Optional[float]:
        try:
            idx = self.candidate_names.index(candidate_name)
        except ValueError:
            return None
        return self.candidate_p_values[idx]

    def to_dict(self) -> dict:
        return {
            "candidate_names": self.candidate_names,
            "candidate_p_values": self.candidate_p_values,
            "candidate_signed_distances": self.candidate_signed_distances,
            "probe_names": self.probe_names,
            "probe_signed_distances": self.probe_signed_distances,
            "null_model": self.null_model.to_dict(),
            "normalizer": self.normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalSortResult":
        return cls(
            candidate_names=list(d["candidate_names"]),
            candidate_p_values=[float(p) for p in d["candidate_p_values"]],
            candidate_signed_distances=[float(x) for x in d["candidate_signed_distances"]],
            probe_names=list(d["probe_names"]),
            probe_signed_distances=[float(x) for x in d["probe_signed_distances"]],
            null_model=NullModel.from_dict(d["null_model"]),
            normalizer=ProbeNormalizer.from_dict(d["normalizer"]),
        )


def sort_one_image(
    candidate_features: Dict[str, Tuple[float, float, float]],
    probe_features: Dict[str, Tuple[float, float, float]],
    shrinkage: Optional[float] = None,
) -> ConformalSortResult:
    """Runs Sections 3.2-3.6 for a single image.

    Args:
        candidate_features: {canonical_name: (s_det, s_clip, s_area)} for
            every o_i in O_init, RAW (pre-sqrt, pre-normalization) values
            straight from feature_extractors.py::FeatureExtractor.extract.
        probe_features: {probe_word: (s_det, s_clip, s_area)} for every
            p in the sampled probe pool P (probe_sampling.py), same raw
            form, extracted with the SAME feature pipeline (Section 3.2's
            "using exactly the same feature pipeline regardless of
            whether w is a candidate or a probe").
        shrinkage: fixed Ledoit-Wolf lambda (Section 3.4), or None to use
            the analytic formula.

    Returns:
        A ConformalSortResult with p-values for every candidate and the
        signed distances for every probe (the conformal reference set).
    """
    if len(probe_features) < 4:
        raise ValueError(
            f"Need at least 4 probes to fit a stable null model, got "
            f"{len(probe_features)} -- the paper's typical K is 50-100 "
            f"(Section 3.1)."
        )

    probe_names = list(probe_features.keys())
    probe_raw = raw_feature_matrix([probe_features[p] for p in probe_names])

    normalizer = ProbeNormalizer.fit(probe_raw)
    probe_norm = normalizer.transform(probe_raw)

    null_model = NullModel.fit(probe_norm, shrinkage=shrinkage)
    D_probes = null_model.signed_distance(probe_norm)

    candidate_names = list(candidate_features.keys())
    if candidate_names:
        cand_raw = raw_feature_matrix([candidate_features[c] for c in candidate_names])
        cand_norm = normalizer.transform(cand_raw)
        D_candidates = null_model.signed_distance(cand_norm)
        p_values = conformal_p_values(D_candidates, D_probes)
    else:
        D_candidates = np.zeros(0)
        p_values = np.zeros(0)

    return ConformalSortResult(
        candidate_names=candidate_names,
        candidate_p_values=[float(p) for p in p_values],
        candidate_signed_distances=[float(x) for x in D_candidates],
        probe_names=probe_names,
        probe_signed_distances=[float(x) for x in D_probes],
        null_model=null_model,
        normalizer=normalizer,
    )
