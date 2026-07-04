"""Run with: python3 tests/test_chair_histogram_nc.py

Tests the PURE logic of chair_histogram_nc.py (labeling, data collection,
plotting, false-verification-rate computation) using a small, hand-built
synthetic ground-truth lookup table -- NOT real CHAIR/COCO data (that
requires the real annotation files on the server; see
build_ground_truth_labels, which is exercised only there). These tests
verify the WIRING and MATH are correct so that when real data is plugged
in on the server, the machinery just works.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chair_histogram_nc import (
    HistogramData,
    collect_histogram_data,
    empirical_false_verification_rates,
    label_candidate,
    plot_probe_vs_candidate_histograms,
)
from synonyms import load_coco_synonym_map


def test_label_candidate_non_coco_is_unjudgeable():
    assert label_candidate("some_ram_only_word", is_coco_category=False, gt_objects={"dog"}) is None
    print("test_label_candidate_non_coco_is_unjudgeable OK")


def test_label_candidate_coco_present_is_real():
    assert label_candidate("dog", is_coco_category=True, gt_objects={"dog", "person"}) is True
    print("test_label_candidate_coco_present_is_real OK")


def test_label_candidate_coco_absent_is_hallucinated():
    assert label_candidate("fork", is_coco_category=True, gt_objects={"dog", "person"}) is False
    print("test_label_candidate_coco_absent_is_hallucinated OK")


def test_label_candidate_uses_synonym_canonicalization():
    # "puppy" isn't itself a COCO node word, but maps to "dog" via the
    # synonym table -- ground truth is keyed by node words ("dog"), so
    # this must still resolve correctly.
    synonyms_map = load_coco_synonym_map()
    assert synonyms_map.get("puppy") == "dog"
    assert label_candidate("puppy", is_coco_category=True, gt_objects={"dog"}, synonyms_map=synonyms_map) is True
    print("test_label_candidate_uses_synonym_canonicalization OK")


def _make_test_data():
    candidate_pool_cache = {
        "img1.jpg": {
            "candidates": [
                {"canonical": "dog", "is_coco_category": True},
                {"canonical": "fork", "is_coco_category": True},
                {"canonical": "grass", "is_coco_category": False},  # non-COCO, excluded
            ]
        },
        "img2.jpg": {
            "candidates": [
                {"canonical": "cat", "is_coco_category": True},
            ]
        },
    }
    sort_results = {
        "img1.jpg": {
            "candidate_names": ["dog", "fork", "grass"],
            "candidate_p_values": [0.02, 1.0, 0.5],
            "candidate_signed_distances": [3.5, float("-inf"), 0.5],
            "probe_signed_distances": [0.1, 0.2, -0.3, float("-inf"), 0.05],
        },
        "img2.jpg": {
            "candidate_names": ["cat"],
            "candidate_p_values": [0.01],
            "candidate_signed_distances": [4.0],
            "probe_signed_distances": [0.15, -0.1],
        },
    }
    ground_truth = {
        "img1.jpg": {"dog"},       # fork is absent -> hallucinated
        "img2.jpg": {"cat"},
    }
    return candidate_pool_cache, sort_results, ground_truth


def test_collect_histogram_data_basic_counts():
    cand_cache, sort_results, gt = _make_test_data()
    data = collect_histogram_data(["img1.jpg", "img2.jpg"], cand_cache, sort_results, gt)

    assert data.n_images == 2
    assert data.n_judged == 3  # dog, fork, cat
    assert data.n_unjudged == 1  # grass
    assert data.real_candidate_p_values == [0.02, 0.01]
    assert data.hallucinated_candidate_p_values == [1.0]
    # -inf distances propagate through as NaN markers (excluded from finite plots)
    assert len(data.hallucinated_candidate_distances) == 1
    assert data.hallucinated_candidate_distances[0] != data.hallucinated_candidate_distances[0]  # is NaN
    # probe distances pool across both images, excluding literal -inf
    assert len(data.probe_distances) == 6  # 5 (img1) + 2 (img2) - 1 (-inf dropped)
    print("test_collect_histogram_data_basic_counts OK")


def test_collect_histogram_data_skips_missing_images():
    cand_cache, sort_results, gt = _make_test_data()
    data = collect_histogram_data(
        ["img1.jpg", "img_not_cached.jpg"], cand_cache, sort_results, gt
    )
    assert data.n_images == 1
    print("test_collect_histogram_data_skips_missing_images OK")


def test_collect_histogram_data_missing_ground_truth_defaults_to_all_hallucinated():
    cand_cache, sort_results, _ = _make_test_data()
    data = collect_histogram_data(["img1.jpg"], cand_cache, sort_results, ground_truth={})
    # with no ground truth entry, gt_objects defaults to empty set -> every
    # COCO candidate is judged hallucinated
    assert data.real_candidate_p_values == []
    assert set(data.hallucinated_candidate_p_values) == {0.02, 1.0}
    print("test_collect_histogram_data_missing_ground_truth_defaults_to_all_hallucinated OK")


def test_empirical_false_verification_rates():
    data = HistogramData(
        hallucinated_candidate_p_values=[0.01, 0.03, 0.15, 0.5, 0.9],
    )
    rates = empirical_false_verification_rates(data, [0.05, 0.2, 0.5])
    assert abs(rates[0.05] - 2 / 5) < 1e-9   # 0.01, 0.03 <= 0.05
    assert abs(rates[0.2] - 3 / 5) < 1e-9    # + 0.15
    assert abs(rates[0.5] - 4 / 5) < 1e-9    # + 0.5
    print("test_empirical_false_verification_rates OK")


def test_empirical_false_verification_rates_no_hallucinated_candidates():
    data = HistogramData(hallucinated_candidate_p_values=[])
    rates = empirical_false_verification_rates(data, [0.1])
    assert rates[0.1] != rates[0.1]  # NaN
    print("test_empirical_false_verification_rates_no_hallucinated_candidates OK")


def test_plot_probe_vs_candidate_histograms_writes_file():
    cand_cache, sort_results, gt = _make_test_data()
    data = collect_histogram_data(["img1.jpg", "img2.jpg"], cand_cache, sort_results, gt)
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "hist.png")
        result = plot_probe_vs_candidate_histograms(data, epsilons=[0.05, 0.2], output_path=out_path)
        assert result == out_path
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    print("test_plot_probe_vs_candidate_histograms_writes_file OK")


def test_plot_probe_vs_candidate_histograms_handles_empty_data():
    """Must not crash on a degenerate/empty dataset (e.g. no images had
    both a candidate cache entry and a sort result)."""
    data = HistogramData()
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "hist_empty.png")
        plot_probe_vs_candidate_histograms(data, epsilons=[0.1], output_path=out_path)
        assert os.path.exists(out_path)
    print("test_plot_probe_vs_candidate_histograms_handles_empty_data OK")


if __name__ == "__main__":
    test_label_candidate_non_coco_is_unjudgeable()
    test_label_candidate_coco_present_is_real()
    test_label_candidate_coco_absent_is_hallucinated()
    test_label_candidate_uses_synonym_canonicalization()
    test_collect_histogram_data_basic_counts()
    test_collect_histogram_data_skips_missing_images()
    test_collect_histogram_data_missing_ground_truth_defaults_to_all_hallucinated()
    test_empirical_false_verification_rates()
    test_empirical_false_verification_rates_no_hallucinated_candidates()
    test_plot_probe_vs_candidate_histograms_writes_file()
    test_plot_probe_vs_candidate_histograms_handles_empty_data()
    print("\nALL chair_histogram_nc.py TESTS PASSED")
