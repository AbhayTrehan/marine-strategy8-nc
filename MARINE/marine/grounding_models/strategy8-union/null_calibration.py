"""
null_calibration.py
====================

Implements Phase I of Strategy 8-U-NC (strategy8_nc.pdf), Sections 3.2-3.7:
the null-calibrated conformal sorter.

DIMENSION-AGNOSTIC: this module works for ANY feature dimension d >= 2.
The original spec uses d=3 (s_det, s_clip, sqrt(s_area)); the updated
pipeline adds s_gdino as a 4th dimension (d=4). The dimension is inferred
from the data at runtime — nothing is hardcoded to 3 or 4.

Feature CONSTRUCTION (which raw values to include, which transforms to
apply — e.g. the sqrt on s_area) is the caller's job, not this module's.
This module takes already-transformed d-dimensional evidence vectors and
does the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Evidence direction: u = [1,1,...,1] / sqrt(d)
# ---------------------------------------------------------------------------
def _evidence_direction(d: int) -> np.ndarray:
    """Eq. 11's fixed evidence direction generalized to d dimensions:
    u = (1/sqrt(d)) * [1, 1, ..., 1]^T — the direction of simultaneously
    increasing evidence in ALL feature dimensions."""
    return np.ones(d) / np.sqrt(float(d))


# ---------------------------------------------------------------------------
# Feature vector helpers (caller-side convenience)
# ---------------------------------------------------------------------------
def build_feature_vector(
    raw_values: Sequence[float],
    sqrt_indices: Sequence[int] = (2,),
) -> np.ndarray:
    """Build a transformed feature vector from raw scores.

    Args:
        raw_values: raw feature values in pipeline order, e.g.
            [s_det, s_clip, s_area] (d=3) or
            [s_det, s_clip, s_area, s_gdino] (d=4).
        sqrt_indices: which indices to apply the variance-stabilizing
            sqrt transform to (default: index 2 = s_area, per Section 3.2).
    """
    arr = np.array(raw_values, dtype=float)
    for idx in sqrt_indices:
        if idx < len(arr):
            arr[idx] = np.sqrt(max(arr[idx], 0.0))
    return arr


def build_feature_matrix(
    feature_tuples: Sequence[Sequence[float]],
    sqrt_indices: Sequence[int] = (2,),
) -> np.ndarray:
    """Vectorized form of build_feature_vector for a list of raw tuples."""
    if len(feature_tuples) == 0:
        return np.zeros((0, 0))
    arr = np.array(feature_tuples, dtype=float)
    for idx in sqrt_indices:
        if idx < arr.shape[1]:
            arr[:, idx] = np.sqrt(np.maximum(arr[:, idx], 0.0))
    return arr


# ---------------------------------------------------------------------------
# Eq. 6-7: probe-only standardization, fresh per image
# ---------------------------------------------------------------------------
@dataclass
class ProbeNormalizer:
    """Eq. 6-7: standardization statistics computed ONLY from one image's
    probe population, then applied to both probes and candidates."""

    mean: np.ndarray  # (d,)
    std: np.ndarray   # (d,)

    @classmethod
    def fit(cls, probe_features: np.ndarray) -> "ProbeNormalizer":
        """Eq. 6: sample mean/std over the probe population (ddof=1)."""
        probe_features = np.asarray(probe_features, dtype=float)
        if probe_features.ndim != 2 or probe_features.shape[1] < 1:
            raise ValueError(f"probe_features must be (K, d) with d >= 1, got shape {probe_features.shape}")
        if probe_features.shape[0] < 2:
            raise ValueError(f"Need at least 2 probes, got {probe_features.shape[0]}")
        mean = probe_features.mean(axis=0)
        std = probe_features.std(axis=0, ddof=1)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Eq. 7: x~_w = (r_w - mu_raw) / sigma_raw, elementwise."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
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
    """Eq. 9: Sigma_0 = (1-lambda)*S + lambda*(tr(S)/d)*I_d, generalized
    to arbitrary dimension d."""
    n, d = X.shape
    mean = X.mean(axis=0)
    diff = X - mean[None, :]
    S = (diff.T @ diff) / max(n - 1, 1)

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
    """Eq. 8-9: one-class Gaussian null (mu_0, Sigma_0) fit from a single
    image's normalized probe vectors. Dimension-agnostic."""

    mean: np.ndarray
    covariance: np.ndarray
    shrinkage: float
    n_probes: int

    def __post_init__(self):
        try:
            self._L = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            self.covariance = self.covariance + 1e-4 * np.eye(self.covariance.shape[0])
            self._L = np.linalg.cholesky(self.covariance)

    @classmethod
    def fit(cls, probe_normalized: np.ndarray, shrinkage: Optional[float] = None) -> "NullModel":
        probe_normalized = np.asarray(probe_normalized, dtype=float)
        n, d = probe_normalized.shape
        if n < d + 1:
            raise ValueError(
                f"Need at least {d+1} probes for a stable {d}D covariance, got {n}"
            )
        mean = probe_normalized.mean(axis=0)
        Sigma0, lam = _ledoit_wolf_shrunk_covariance(probe_normalized, shrinkage=shrinkage)
        return cls(mean=mean, covariance=Sigma0, shrinkage=lam, n_probes=n)

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Eq. 10 (unsigned)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        diff = X - self.mean[None, :]
        z = np.linalg.solve(self._L, diff.T)
        maha2 = np.sum(z ** 2, axis=0)
        return np.sqrt(np.maximum(maha2, 0.0))

    def signed_distance(self, X: np.ndarray) -> np.ndarray:
        """Eq. 11: one-sided projection test, d-dimensional."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = X.shape[1]
        u = _evidence_direction(d)
        dist = self.mahalanobis(X)
        proj = (X - self.mean[None, :]) @ u
        return np.where(proj > 0, dist, -np.inf)

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
    """Eq. 12: p(o_i) = (1 + |{p in P : D(p) >= D(o_i)}|) / (K + 1)."""
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
    """Full per-image output of Phase I."""

    candidate_names: List[str]
    candidate_p_values: List[float]
    candidate_signed_distances: List[float]
    probe_names: List[str]
    probe_signed_distances: List[float]
    null_model: NullModel
    normalizer: ProbeNormalizer
    n_features: int = 3  # d, recorded for downstream consumers

    def split(self, epsilon: float) -> Tuple[List[str], List[str]]:
        """Eq. 13-14."""
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
            "n_features": self.n_features,
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
            n_features=int(d.get("n_features", 3)),
        )


def sort_one_image(
    candidate_features: Dict[str, Sequence[float]],
    probe_features: Dict[str, Sequence[float]],
    shrinkage: Optional[float] = None,
    sqrt_indices: Sequence[int] = (2,),
) -> ConformalSortResult:
    """Runs Sections 3.2-3.6 for a single image. Dimension-agnostic.

    Args:
        candidate_features: {name: (raw_feat_1, ..., raw_feat_d)} for each
            candidate, raw (pre-transform). For d=3: (s_det, s_clip, s_area).
            For d=4: (s_det, s_clip, s_area, s_gdino).
        probe_features: {name: (raw_feat_1, ..., raw_feat_d)}, same format.
        shrinkage: fixed Ledoit-Wolf lambda, or None for analytic formula.
        sqrt_indices: which feature indices get the variance-stabilizing
            sqrt transform (default: (2,) = s_area only).
    """
    if len(probe_features) < 5:
        raise ValueError(f"Need at least 5 probes, got {len(probe_features)}")

    probe_names = list(probe_features.keys())
    probe_dims = {len(v) for v in probe_features.values()}
    if len(probe_dims) > 1:
        raise ValueError(
            f"Inconsistent feature dimensions across probes: {sorted(probe_dims)}. "
            f"Every probe must have the same number of raw feature values "
            f"(all 3D, or all 4D with s_gdino included)."
        )
    if candidate_features:
        cand_dims = {len(v) for v in candidate_features.values()}
        if len(cand_dims) > 1:
            raise ValueError(
                f"Inconsistent feature dimensions across candidates: {sorted(cand_dims)}."
            )
        d_cand = next(iter(cand_dims))
        d_probe = next(iter(probe_dims))
        if d_cand != d_probe:
            raise ValueError(
                f"Candidate features are {d_cand}D but probe features are {d_probe}D. "
                f"This usually means candidates were enriched with s_gdino "
                f"(enrich_gdino.py) but probes weren't (build_probe_pool.py "
                f"without --gdino_model), or vice versa -- both sides must "
                f"use the SAME feature set for a given run."
            )

    probe_raw = build_feature_matrix([probe_features[p] for p in probe_names], sqrt_indices=sqrt_indices)
    d = probe_raw.shape[1]

    normalizer = ProbeNormalizer.fit(probe_raw)
    probe_norm = normalizer.transform(probe_raw)

    null_model = NullModel.fit(probe_norm, shrinkage=shrinkage)
    D_probes = null_model.signed_distance(probe_norm)

    candidate_names = list(candidate_features.keys())
    if candidate_names:
        cand_raw = build_feature_matrix([candidate_features[c] for c in candidate_names], sqrt_indices=sqrt_indices)
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
        n_features=d,
    )
