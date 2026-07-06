"""Run with: python3 tests/test_null_calibration.py

Unit tests for the dimension-agnostic null_calibration.py. Tests both
d=3 (original) and d=4 (with GroundingDINO) configurations.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from null_calibration import (
    ConformalSortResult, NullModel, ProbeNormalizer,
    build_feature_vector, build_feature_matrix,
    conformal_p_values, sort_one_image, _evidence_direction,
)


def test_evidence_direction_unit_norm():
    for d in [2, 3, 4, 7]:
        u = _evidence_direction(d)
        assert len(u) == d
        assert abs(np.linalg.norm(u) - 1.0) < 1e-12
    print("test_evidence_direction_unit_norm OK")


def test_build_feature_vector_3d():
    v = build_feature_vector([0.8, 0.25, 0.09], sqrt_indices=[2])
    assert abs(v[0] - 0.8) < 1e-12
    assert abs(v[1] - 0.25) < 1e-12
    assert abs(v[2] - 0.3) < 1e-12  # sqrt(0.09)
    print("test_build_feature_vector_3d OK")


def test_build_feature_vector_4d():
    v = build_feature_vector([0.8, 0.25, 0.09, 0.6], sqrt_indices=[2])
    assert abs(v[2] - 0.3) < 1e-12  # sqrt(0.09)
    assert abs(v[3] - 0.6) < 1e-12  # s_gdino untouched
    print("test_build_feature_vector_4d OK")


def test_build_feature_matrix_4d():
    mat = build_feature_matrix([(0.8, 0.25, 0.09, 0.6), (0.1, 0.2, 0.04, 0.3)], sqrt_indices=[2])
    assert mat.shape == (2, 4)
    assert abs(mat[0, 2] - 0.3) < 1e-12
    assert abs(mat[0, 3] - 0.6) < 1e-12
    assert abs(mat[1, 2] - 0.2) < 1e-12
    print("test_build_feature_matrix_4d OK")


def test_probe_normalizer_arbitrary_dim():
    for d in [3, 4]:
        rng = np.random.RandomState(0)
        probe = rng.normal(size=(60, d))
        norm = ProbeNormalizer.fit(probe)
        transformed = norm.transform(probe)
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(transformed.std(axis=0, ddof=1), 1.0, atol=1e-8)
    print("test_probe_normalizer_arbitrary_dim OK")


def test_null_model_fit_4d():
    rng = np.random.RandomState(1)
    probe = rng.normal(size=(60, 4))
    norm = ProbeNormalizer.fit(probe)
    probe_norm = norm.transform(probe)
    null = NullModel.fit(probe_norm, shrinkage=0.1)
    assert null.covariance.shape == (4, 4)
    assert null.mean.shape == (4,)
    print("test_null_model_fit_4d OK")


def test_signed_distance_4d_rejects_below_mean():
    rng = np.random.RandomState(2)
    probe = rng.normal(loc=0.3, scale=0.05, size=(60, 4))
    norm = ProbeNormalizer.fit(probe)
    probe_norm = norm.transform(probe)
    null = NullModel.fit(probe_norm, shrinkage=0.1)
    low = norm.transform(np.array([[0.001, 0.001, 0.0, 0.001]]))
    signed = null.signed_distance(low)
    assert signed[0] == -np.inf
    print("test_signed_distance_4d_rejects_below_mean OK")


def test_conformal_p_value_mechanics():
    D_probes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p_top = conformal_p_values(np.array([100.0]), D_probes)[0]
    assert abs(p_top - 1.0/6.0) < 1e-12
    p_bottom = conformal_p_values(np.array([-100.0]), D_probes)[0]
    assert abs(p_bottom - 1.0) < 1e-12
    print("test_conformal_p_value_mechanics OK")


def test_conformal_validity_marginal_coverage_4d():
    """Monte Carlo check of the conformal guarantee for d=4."""
    rng = np.random.RandomState(42)
    K = 80
    n_trials = 3000
    eps_values = [0.05, 0.1, 0.2]
    false_verify_counts = {eps: 0 for eps in eps_values}
    mean = np.array([0.1, 0.2, 0.05, 0.15])
    cov = np.diag([0.05, 0.03, 0.02, 0.04]) ** 2
    for _ in range(n_trials):
        sample = rng.multivariate_normal(mean, cov, size=K + 1)
        probes_raw = sample[:K]
        candidate_raw = sample[K:K+1]
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
        assert rate <= eps + 0.03, f"eps={eps}: rate {rate:.4f}"
        print(f"  eps={eps}: empirical false-verification rate = {rate:.4f}")
    print("test_conformal_validity_marginal_coverage_4d OK")


def test_sort_one_image_3d():
    rng = np.random.RandomState(6)
    probe_feats = {f"p{i}": (float(rng.normal(0.05, 0.02)), float(rng.normal(0.15, 0.02)), float(max(rng.normal(0.0, 0.01), 0.0))) for i in range(60)}
    cand_feats = {"real_dog": (0.85, 0.32, 0.15), "hall_fork": (0.04, 0.14, 0.0)}
    result = sort_one_image(cand_feats, probe_feats, shrinkage=0.1, sqrt_indices=[2])
    assert result.n_features == 3
    assert result.p_value_of("real_dog") < result.p_value_of("hall_fork")
    o_pos, o_neg = result.split(0.1)
    assert "real_dog" in o_pos
    assert "hall_fork" in o_neg
    print("test_sort_one_image_3d OK")


def test_sort_one_image_4d():
    rng = np.random.RandomState(7)
    probe_feats = {f"p{i}": (float(rng.normal(0.05, 0.02)), float(rng.normal(0.15, 0.02)), float(max(rng.normal(0.0, 0.01), 0.0)), float(rng.normal(0.05, 0.02))) for i in range(60)}
    cand_feats = {"real_dog": (0.85, 0.32, 0.15, 0.7), "hall_fork": (0.04, 0.14, 0.0, 0.03)}
    result = sort_one_image(cand_feats, probe_feats, shrinkage=0.1, sqrt_indices=[2])
    assert result.n_features == 4
    assert result.p_value_of("real_dog") < result.p_value_of("hall_fork")
    print("test_sort_one_image_4d OK")


def test_sort_result_roundtrip_4d():
    rng = np.random.RandomState(8)
    probe_feats = {f"p{i}": (0.05, 0.15, 0.0, 0.05) for i in range(20)}
    cand_feats = {"dog": (0.8, 0.3, 0.1, 0.7)}
    result = sort_one_image(cand_feats, probe_feats, sqrt_indices=[2])
    restored = ConformalSortResult.from_dict(result.to_dict())
    assert restored.n_features == 4
    assert restored.split(0.1) == result.split(0.1)
    print("test_sort_result_roundtrip_4d OK")


if __name__ == "__main__":
    test_evidence_direction_unit_norm()
    test_build_feature_vector_3d()
    test_build_feature_vector_4d()
    test_build_feature_matrix_4d()
    test_probe_normalizer_arbitrary_dim()
    test_null_model_fit_4d()
    test_signed_distance_4d_rejects_below_mean()
    test_conformal_p_value_mechanics()
    test_conformal_validity_marginal_coverage_4d()
    test_sort_one_image_3d()
    test_sort_one_image_4d()
    test_sort_result_roundtrip_4d()
    print("\nALL null_calibration.py TESTS PASSED")
