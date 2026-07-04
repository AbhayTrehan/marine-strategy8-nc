"""Run with: python3 tests/test_null_calibration.py

Unit tests for null_calibration.py (Strategy 8-U-NC Phase I, Sections
3.2-3.7). These use small, hand-constructed or synthetic-Gaussian arrays
purely to verify the MATH is implemented correctly (equation-by-equation
against the paper) -- they are NOT a substitute for running the real
pipeline against real images/probes on the server, and none of these
numbers should be read as "sanity check" results.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from null_calibration import (
    ConformalSortResult,
    NullModel,
    ProbeNormalizer,
    conformal_p_values,
    raw_feature_matrix,
    raw_feature_vector,
    sort_one_image,
)


def test_raw_feature_vector_sqrt_area_only():
    r = raw_feature_vector(s_det=0.8, s_clip=0.25, s_area=0.09)
    assert abs(r[0] - 0.8) < 1e-12
    assert abs(r[1] - 0.25) < 1e-12
    assert abs(r[2] - 0.3) < 1e-12  # sqrt(0.09) = 0.3
    print("test_raw_feature_vector_sqrt_area_only OK")


def test_raw_feature_vector_negative_area_clamped():
    r = raw_feature_vector(s_det=0.1, s_clip=0.2, s_area=-0.5)
    assert r[2] == 0.0
    print("test_raw_feature_vector_negative_area_clamped OK")


def test_raw_feature_matrix_matches_vector():
    triples = [(0.8, 0.25, 0.09), (0.1, 0.2, 0.04)]
    mat = raw_feature_matrix(triples)
    assert mat.shape == (2, 3)
    assert abs(mat[0, 2] - 0.3) < 1e-12
    assert abs(mat[1, 2] - 0.2) < 1e-12
    print("test_raw_feature_matrix_matches_vector OK")


def test_probe_normalizer_zero_mean_after_transform():
    rng = np.random.RandomState(0)
    probe_raw = rng.normal(loc=[0.1, 0.2, 0.05], scale=[0.05, 0.03, 0.02], size=(60, 3))
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)
    # by construction, normalized probes have ~zero mean, ~unit std
    assert np.allclose(probe_norm.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(probe_norm.std(axis=0, ddof=1), 1.0, atol=1e-8)
    print("test_probe_normalizer_zero_mean_after_transform OK")


def test_probe_normalizer_guards_constant_dimension():
    probe_raw = np.zeros((10, 3))
    probe_raw[:, 0] = np.linspace(0.1, 0.2, 10)
    # dims 1 and 2 are exactly constant -- must not divide by zero
    norm = ProbeNormalizer.fit(probe_raw)
    assert norm.std[1] == 1.0 and norm.std[2] == 1.0
    out = norm.transform(probe_raw)
    assert np.all(np.isfinite(out))
    print("test_probe_normalizer_guards_constant_dimension OK")


def test_null_model_mean_near_zero_after_probe_normalization():
    rng = np.random.RandomState(1)
    probe_raw = rng.normal(loc=[0.1, 0.2, 0.05], scale=[0.05, 0.03, 0.02], size=(80, 3))
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)
    null = NullModel.fit(probe_norm, shrinkage=0.1)
    # Section 3.4: "mu_0 ~ 0 by construction of the standardization"
    assert np.allclose(null.mean, 0.0, atol=1e-8)
    print("test_null_model_mean_near_zero_after_probe_normalization OK")


def test_shrinkage_moves_covariance_toward_scaled_identity():
    rng = np.random.RandomState(2)
    # strongly correlated probes -> far from identity without shrinkage
    base = rng.normal(size=(100, 1))
    probe_raw = np.hstack([base + 0.01 * rng.normal(size=(100, 1)) for _ in range(3)])
    probe_raw = probe_raw + np.array([0.3, 0.3, 0.3])  # keep values positive-ish
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)

    null_no_shrink = NullModel.fit(probe_norm, shrinkage=0.0)
    null_full_shrink = NullModel.fit(probe_norm, shrinkage=1.0)

    # off-diagonal terms should shrink toward 0 (scaled identity has none)
    off_diag_no_shrink = np.abs(null_no_shrink.covariance - np.diag(np.diag(null_no_shrink.covariance))).sum()
    off_diag_full_shrink = np.abs(null_full_shrink.covariance - np.diag(np.diag(null_full_shrink.covariance))).sum()
    assert off_diag_full_shrink < off_diag_no_shrink
    assert off_diag_full_shrink < 1e-8  # lambda=1 -> EXACTLY scaled identity
    print("test_shrinkage_moves_covariance_toward_scaled_identity OK")


def test_analytic_ledoit_wolf_when_shrinkage_none():
    rng = np.random.RandomState(3)
    probe_raw = rng.normal(loc=[0.1, 0.2, 0.05], scale=[0.05, 0.03, 0.02], size=(50, 3))
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)
    null = NullModel.fit(probe_norm, shrinkage=None)
    assert 0.0 <= null.shrinkage <= 1.0
    print("test_analytic_ledoit_wolf_when_shrinkage_none OK")


def test_signed_distance_rejects_below_average_direction():
    """Eq. 11: a word whose evidence is LOWER than the null mean in all
    three dimensions (negative projection onto u) must get D(w) = -inf,
    even though its unsigned Mahalanobis distance could be large."""
    rng = np.random.RandomState(4)
    probe_raw = rng.normal(loc=[0.3, 0.3, 0.3], scale=[0.05, 0.05, 0.05], size=(60, 3))
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)
    null = NullModel.fit(probe_norm, shrinkage=0.1)

    # a hallucinated-looking candidate: near-zero evidence in all 3 dims,
    # i.e. clearly BELOW the (positive) probe mean -> negative projection
    low_evidence = norm.transform(raw_feature_matrix([(0.001, 0.001, 0.0)]))
    signed = null.signed_distance(low_evidence)
    assert signed[0] == -np.inf, signed
    # sanity: its UNSIGNED distance is large and finite (it's an outlier,
    # just in the "wrong" direction)
    unsigned = null.mahalanobis(low_evidence)
    assert unsigned[0] > 0 and np.isfinite(unsigned[0])
    print("test_signed_distance_rejects_below_average_direction OK")


def test_signed_distance_accepts_above_average_direction():
    """A word with evidence clearly HIGHER than the null in all three
    dimensions must get a finite, positive signed distance."""
    rng = np.random.RandomState(5)
    probe_raw = rng.normal(loc=[0.05, 0.1, 0.02], scale=[0.02, 0.02, 0.01], size=(60, 3))
    norm = ProbeNormalizer.fit(probe_raw)
    probe_norm = norm.transform(probe_raw)
    null = NullModel.fit(probe_norm, shrinkage=0.1)

    high_evidence = norm.transform(raw_feature_matrix([(0.8, 0.3, 0.3)]))
    signed = null.signed_distance(high_evidence)
    assert np.isfinite(signed[0]) and signed[0] > 0
    print("test_signed_distance_accepts_above_average_direction OK")


def test_conformal_p_value_range_and_extremes():
    D_probes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # a candidate at the very top of the probe distribution -> smallest p
    p_top = conformal_p_values(np.array([100.0]), D_probes)[0]
    assert abs(p_top - (1.0 / 6.0)) < 1e-12  # 0 probes >= 100 -> (1+0)/6
    # a candidate at the very bottom -> largest p (never verified)
    p_bottom = conformal_p_values(np.array([-100.0]), D_probes)[0]
    assert abs(p_bottom - 1.0) < 1e-12  # all 5 probes >= -100 -> (1+5)/6 = 1
    print("test_conformal_p_value_range_and_extremes OK")


def test_conformal_p_value_minus_inf_gets_maximal_p():
    """Section 3.5: D(w) = -inf must be automatically assigned the maximal
    conformal p-value with no special-casing in conformal_p_values."""
    D_probes = np.array([1.0, -np.inf, 2.0, -np.inf, 3.0])
    p = conformal_p_values(np.array([-np.inf]), D_probes)[0]
    assert abs(p - 1.0) < 1e-12
    print("test_conformal_p_value_minus_inf_gets_maximal_p OK")


def test_conformal_validity_marginal_coverage():
    """The core theoretical guarantee (Section 3.6): if a candidate is
    truly exchangeable with the probe population (i.e. it IS in fact just
    another draw from the null/absent distribution), then
    Pr[p(o_i) <= eps] <= eps for any eps, marginally over many random
    draws. We check this empirically over many trials by treating a
    freshly-drawn (K+1)-th null sample as the "candidate" each time and
    confirming the false-verification rate tracks eps rather than
    blowing past it. This IS a from-first-principles statistical test of
    the mechanism the whole sanity-check exists to validate, using
    synthetic Gaussian data (not real image data -- that must come from
    the real pipeline on the server), so it's placed here alongside the
    plain unit tests deliberately."""
    rng = np.random.RandomState(42)
    K = 80
    n_trials = 4000
    eps_values = [0.05, 0.1, 0.2]
    false_verify_counts = {eps: 0 for eps in eps_values}

    mean = np.array([0.1, 0.2, 0.05])
    cov = np.diag([0.05, 0.03, 0.02]) ** 2

    for _ in range(n_trials):
        sample = rng.multivariate_normal(mean, cov, size=K + 1)
        probes_raw = sample[:K]
        candidate_raw = sample[K:K + 1]  # exchangeable with the probes

        norm = ProbeNormalizer.fit(probes_raw)
        probe_norm = norm.transform(probes_raw)
        null = NullModel.fit(probe_norm, shrinkage=0.1)
        D_probes = null.signed_distance(probe_norm)

        cand_norm = norm.transform(candidate_raw)
        D_cand = null.signed_distance(cand_norm)
        p = conformal_p_values(D_cand, D_probes)[0]

        for eps in eps_values:
            if p <= eps:
                false_verify_counts[eps] += 1

    for eps in eps_values:
        rate = false_verify_counts[eps] / n_trials
        # allow generous Monte Carlo slack; the guarantee is Pr[...] <= eps,
        # so we mainly check it doesn't blow past eps by a wide margin
        assert rate <= eps + 0.03, (
            f"eps={eps}: empirical false-verification rate {rate:.4f} "
            f"exceeds the conformal guarantee by more than MC slack"
        )
        print(f"  eps={eps}: empirical false-verification rate = {rate:.4f} (guarantee: <= {eps})")
    print("test_conformal_validity_marginal_coverage OK")


def test_sort_one_image_end_to_end_and_split():
    rng = np.random.RandomState(6)
    # probes: absent-object-like evidence, low and tight
    probe_feats = {
        f"probe_{i}": (
            float(rng.normal(0.05, 0.02)),
            float(rng.normal(0.15, 0.02)),
            float(max(rng.normal(0.0, 0.01), 0.0)),
        )
        for i in range(60)
    }
    # candidates: one clearly-real object (high evidence), one clearly-hallucinated
    candidate_feats = {
        "real_dog": (0.85, 0.32, 0.15),
        "hallucinated_fork": (0.04, 0.14, 0.0),
    }
    result = sort_one_image(candidate_feats, probe_feats, shrinkage=0.1)
    assert set(result.candidate_names) == {"real_dog", "hallucinated_fork"}

    p_real = result.p_value_of("real_dog")
    p_hall = result.p_value_of("hallucinated_fork")
    assert p_real < p_hall, (p_real, p_hall)
    assert p_hall == 1.0, p_hall  # hallucinated candidate's evidence is BELOW the null mean

    o_pos, o_neg = result.split(epsilon=0.1)
    assert "real_dog" in o_pos
    assert "hallucinated_fork" in o_neg
    assert "hallucinated_fork" not in o_pos

    # round-trip through to_dict/from_dict
    restored = ConformalSortResult.from_dict(result.to_dict())
    assert restored.split(0.1) == (o_pos, o_neg)
    print("test_sort_one_image_end_to_end_and_split OK")


def test_sort_one_image_empty_candidates():
    probe_feats = {f"probe_{i}": (0.05, 0.15, 0.0) for i in range(10)}
    result = sort_one_image({}, probe_feats)
    assert result.candidate_names == []
    o_pos, o_neg = result.split(0.1)
    assert o_pos == [] and o_neg == []
    print("test_sort_one_image_empty_candidates OK")


def test_sort_one_image_raises_on_too_few_probes():
    try:
        sort_one_image({"dog": (0.8, 0.3, 0.1)}, {"p1": (0.1, 0.1, 0.0)})
        assert False, "expected ValueError for too few probes"
    except ValueError:
        pass
    print("test_sort_one_image_raises_on_too_few_probes OK")


def test_epsilon_out_of_range_raises():
    probe_feats = {f"probe_{i}": (0.05, 0.15, 0.0) for i in range(10)}
    result = sort_one_image({"dog": (0.8, 0.3, 0.1)}, probe_feats)
    for bad_eps in [0.0, 1.0, -0.1, 1.5]:
        try:
            result.split(bad_eps)
            assert False, f"expected ValueError for epsilon={bad_eps}"
        except ValueError:
            pass
    print("test_epsilon_out_of_range_raises OK")


if __name__ == "__main__":
    test_raw_feature_vector_sqrt_area_only()
    test_raw_feature_vector_negative_area_clamped()
    test_raw_feature_matrix_matches_vector()
    test_probe_normalizer_zero_mean_after_transform()
    test_probe_normalizer_guards_constant_dimension()
    test_null_model_mean_near_zero_after_probe_normalization()
    test_shrinkage_moves_covariance_toward_scaled_identity()
    test_analytic_ledoit_wolf_when_shrinkage_none()
    test_signed_distance_rejects_below_average_direction()
    test_signed_distance_accepts_above_average_direction()
    test_conformal_p_value_range_and_extremes()
    test_conformal_p_value_minus_inf_gets_maximal_p()
    test_conformal_validity_marginal_coverage()
    test_sort_one_image_end_to_end_and_split()
    test_sort_one_image_empty_candidates()
    test_sort_one_image_raises_on_too_few_probes()
    test_epsilon_out_of_range_raises()
    print("\nALL null_calibration.py TESTS PASSED")
