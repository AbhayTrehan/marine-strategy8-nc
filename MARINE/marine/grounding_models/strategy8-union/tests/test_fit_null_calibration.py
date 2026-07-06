"""Run with: python3 tests/test_fit_null_calibration.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fit_null_calibration import (
    build_question_file_nc,
    classify_image_candidates_nc,
    load_sort_results,
    save_sort_results,
    sort_all_images,
)


def _make_caches():
    candidate_pool_cache = {
        "img1.jpg": {
            "candidates": [
                {"canonical": "dog", "s_det": 0.85, "s_clip": 0.32, "s_area": 0.15},
                {"canonical": "fork", "s_det": 0.03, "s_clip": 0.14, "s_area": 0.0},
            ]
        },
        "img2.jpg": {
            "candidates": [
                {"canonical": "cat", "s_det": 0.9, "s_clip": 0.35, "s_area": 0.2},
            ]
        },
        # deliberately no probe pool for this one -- should be skipped, not crash
        "img3_missing_probes.jpg": {
            "candidates": [{"canonical": "car", "s_det": 0.5, "s_clip": 0.2, "s_area": 0.1}]
        },
    }
    probe_pool_cache = {
        "img1.jpg": {
            "probes": [
                {"word": f"probe{i}", "s_det": 0.05, "s_clip": 0.15, "s_area": 0.0}
                for i in range(20)
            ]
        },
        "img2.jpg": {
            "probes": [
                {"word": f"probe{i}", "s_det": 0.05, "s_clip": 0.15, "s_area": 0.0}
                for i in range(20)
            ]
        },
    }
    return candidate_pool_cache, probe_pool_cache


def test_sort_all_images_skips_missing_probe_pool():
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache)
    assert set(results.keys()) == {"img1.jpg", "img2.jpg"}
    print("test_sort_all_images_skips_missing_probe_pool OK")


def test_sort_all_images_respects_image_filter():
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache, image_filter=["img1.jpg"])
    assert set(results.keys()) == {"img1.jpg"}
    print("test_sort_all_images_respects_image_filter OK")


def test_classify_image_candidates_nc_splits_correctly():
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache)
    o_pos, o_neg, p_values = classify_image_candidates_nc(results["img1.jpg"], epsilon=0.1)
    assert "dog" in o_pos
    assert "fork" in o_neg
    assert p_values["dog"] < p_values["fork"]
    print("test_classify_image_candidates_nc_splits_correctly OK")


def test_sort_results_save_and_load_roundtrip():
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sort_results.json")
        save_sort_results(results, path)
        restored = load_sort_results(path)
    assert set(restored.keys()) == set(results.keys())
    for img in results:
        assert restored[img].split(0.1) == results[img].split(0.1)
    print("test_sort_results_save_and_load_roundtrip OK")


def test_build_question_file_nc_matches_original_schema():
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache)

    with tempfile.TemporaryDirectory() as d:
        qfile = os.path.join(d, "questions.json")
        with open(qfile, "w") as f:
            json.dump([
                {"id": 1, "image": "img1.jpg", "conversations": [
                    {"from": "human", "value": "Generate a short caption of the image."},
                    {"from": "gpt", "value": ""},
                ]},
                {"id": 2, "image": "img2.jpg", "conversations": [
                    {"from": "human", "value": "Generate a short caption of the image."},
                    {"from": "gpt", "value": ""},
                ]},
            ], f)

        out_questions, per_image = build_question_file_nc(qfile, results, epsilon=0.1)

    assert len(out_questions) == 2
    q1 = next(q for q in out_questions if q["image"] == "img1.jpg")
    convs = {c["from"]: c["value"] for c in q1["conversations"]}
    assert "guidance_pos" in convs and "guidance_neg" in convs
    assert "dog" in convs["guidance_pos"]
    assert "fork" in convs["guidance_neg"]
    assert per_image["img1.jpg"]["epsilon"] == 0.1
    print("test_build_question_file_nc_matches_original_schema OK")


def test_epsilon_sweep_reuses_same_sort_no_recompute():
    """The whole point of caching ConformalSortResult: sweeping epsilon for
    the sanity-check report (0.05, 0.1, 0.2, ...) must be pure re-splitting,
    no refitting."""
    cand_cache, probe_cache = _make_caches()
    results = sort_all_images(cand_cache, probe_cache)
    r = results["img1.jpg"]
    splits = {eps: r.split(eps) for eps in [0.05, 0.1, 0.2, 0.5]}
    # tighter epsilon should never verify MORE objects than a looser one
    assert len(splits[0.05][0]) <= len(splits[0.1][0]) <= len(splits[0.2][0]) <= len(splits[0.5][0])
    print("test_epsilon_sweep_reuses_same_sort_no_recompute OK")


if __name__ == "__main__":
    test_sort_all_images_skips_missing_probe_pool()
    test_sort_all_images_respects_image_filter()
    test_classify_image_candidates_nc_splits_correctly()
    test_sort_results_save_and_load_roundtrip()
    test_build_question_file_nc_matches_original_schema()
    test_epsilon_sweep_reuses_same_sort_no_recompute()
    print("\nALL fit_null_calibration.py TESTS PASSED")
