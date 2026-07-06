"""Run with: python3 tests/test_report_nc.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

from report_nc import build_candidate_table_nc, generate_report_nc, build_probe_summary_html


def _make_image(d, name):
    path = os.path.join(d, name)
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(path)
    return path


def _make_artifacts(d):
    _make_image(d, "img0.jpg")
    _make_image(d, "img1.jpg")

    candidate_pool_cache = {
        "img0.jpg": {
            "image": "img0.jpg",
            "pass1_caption": "A dog is sitting near a fork on the table.",
            "raw": {"ram": ["dog", "table"], "detr": ["dog"], "vlm": ["dog", "fork", "table"]},
            "candidates": [
                {"canonical": "dog", "sources": ["ram", "detr", "vlm"], "raw_mentions": ["dog"],
                 "is_coco_category": True, "s_det": 0.82, "s_clip": 0.31, "s_area": 0.12},
                {"canonical": "fork", "sources": ["vlm"], "raw_mentions": ["fork"],
                 "is_coco_category": True, "s_det": 0.03, "s_clip": 0.14, "s_area": 0.0},
            ],
        },
        "img1.jpg": {
            "image": "img1.jpg",
            "pass1_caption": "A cat.",
            "raw": {"ram": [], "detr": [], "vlm": ["cat"]},
            "candidates": [
                {"canonical": "cat", "sources": ["vlm"], "raw_mentions": ["cat"],
                 "is_coco_category": True, "s_det": 0.6, "s_clip": 0.2, "s_area": 0.1},
            ],
        },
    }

    probe_pool_cache = {
        "img0.jpg": {"K": 40, "tau_low": 0.3, "probes": [{"word": f"p{i}"} for i in range(40)]},
        "img1.jpg": {"K": 40, "tau_low": 0.3, "probes": [{"word": f"p{i}"} for i in range(40)]},
    }

    sort_results = {
        "img0.jpg": {
            "candidate_names": ["dog", "fork"],
            "candidate_p_values": [0.02, 1.0],
            "candidate_signed_distances": [3.5, float("-inf")],
            "probe_names": [f"p{i}" for i in range(40)],
            "probe_signed_distances": [0.1] * 40,
            "null_model": {"mean": [0.0, 0.0, 0.0], "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           "shrinkage": 0.1, "n_probes": 40},
            "normalizer": {"mean": [0.1, 0.2, 0.05], "std": [0.05, 0.03, 0.02]},
        },
        "img1.jpg": {
            "candidate_names": ["cat"],
            "candidate_p_values": [0.01],
            "candidate_signed_distances": [4.0],
            "probe_names": [f"p{i}" for i in range(40)],
            "probe_signed_distances": [0.1] * 40,
            "null_model": {"mean": [0.0, 0.0, 0.0], "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           "shrinkage": 0.2, "n_probes": 40},
            "normalizer": {"mean": [0.1, 0.2, 0.05], "std": [0.05, 0.03, 0.02]},
        },
    }
    sort_results_path = os.path.join(d, "sort_results.json")
    with open(sort_results_path, "w") as f:
        json.dump(sort_results, f)

    questions = [
        {"id": 1, "image": "img0.jpg", "conversations": [
            {"from": "human", "value": "Generate a short caption of the image."},
            {"from": "gpt", "value": ""},
            {"from": "guidance_pos", "value": "Focusing on the visible objects in this image: dog. generate a short caption of the image."},
            {"from": "guidance_neg", "value": "Focusing on the visible objects in this image: fork. generate a short caption of the image."},
        ]},
        {"id": 2, "image": "img1.jpg", "conversations": [
            {"from": "human", "value": "Generate a short caption of the image."},
            {"from": "gpt", "value": ""},
            {"from": "guidance_pos", "value": "Focusing on the visible objects in this image: cat. generate a short caption of the image."},
            {"from": "guidance_neg", "value": "Generate a short caption of the image."},
        ]},
    ]
    question_path = os.path.join(d, "questions.json")
    with open(question_path, "w") as f:
        json.dump(questions, f)

    answers = [
        {"question_id": 1, "text": "A dog sits near a table."},
        {"question_id": 2, "text": "A cat."},
    ]
    answers_path = os.path.join(d, "answers.jsonl")
    with open(answers_path, "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")

    return candidate_pool_cache, probe_pool_cache, sort_results_path, question_path, answers_path, d


def test_generate_report_nc_basic():
    d = tempfile.mkdtemp()
    cache, probe_cache, sort_path, q_path, ans_path, image_dir = _make_artifacts(d)
    out_path = os.path.join(d, "report_nc.html")

    result_path = generate_report_nc(
        report_images=["img0.jpg", "img1.jpg"],
        candidate_pool_cache=cache,
        probe_pool_cache=probe_cache,
        sort_results_file=sort_path,
        epsilons=[0.05, 0.2],
        image_dir=image_dir,
        output_path=out_path,
        question_file=q_path,
        answers_file=ans_path,
        config_info={"K": 40, "tau_low": 0.3},
    )
    assert result_path == out_path
    with open(out_path) as f:
        content = f.read()

    assert "img0.jpg" in content and "img1.jpg" in content
    assert "A dog is sitting near a fork on the table." in content
    assert "A dog sits near a table." in content
    assert "0.0200" in content  # dog's p-value
    assert "POSITIVE" in content and "HALLUCINATED" in content
    assert "no ground-truth annotations" in content.lower() or "does not use chair/coco ground-truth" in content.lower() or "ground-truth" in content.lower()
    print("test_generate_report_nc_basic OK")


def test_generate_report_nc_requires_nonempty_epsilons():
    d = tempfile.mkdtemp()
    cache, probe_cache, sort_path, q_path, ans_path, image_dir = _make_artifacts(d)
    try:
        generate_report_nc(
            report_images=["img0.jpg"], candidate_pool_cache=cache, probe_pool_cache=probe_cache,
            sort_results_file=sort_path, epsilons=[], image_dir=image_dir,
            output_path=os.path.join(d, "x.html"),
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_generate_report_nc_requires_nonempty_epsilons OK")


def test_build_candidate_table_nc_orders_by_p_value():
    candidates = [
        {"canonical": "fork", "sources": ["vlm"], "raw_mentions": ["fork"], "s_det": 0.02, "s_clip": 0.14, "s_area": 0.0},
        {"canonical": "dog", "sources": ["ram"], "raw_mentions": ["dog"], "s_det": 0.9, "s_clip": 0.3, "s_area": 0.1},
    ]
    p_values = {"fork": 1.0, "dog": 0.02}
    signed_distances = {"fork": float("-inf"), "dog": 3.5}
    table = build_candidate_table_nc(candidates, p_values, signed_distances, epsilons=[0.1])
    assert table.index("dog") < table.index("fork")
    assert "POSITIVE" in table and "HALLUCINATED" in table
    assert "-inf" in table
    print("test_build_candidate_table_nc_orders_by_p_value OK")


def test_build_candidate_table_nc_multiple_epsilon_columns():
    candidates = [{"canonical": "dog", "sources": ["ram"], "raw_mentions": ["dog"], "s_det": 0.9, "s_clip": 0.3, "s_area": 0.1}]
    p_values = {"dog": 0.15}
    signed_distances = {"dog": 1.0}
    # at eps=0.1 dog should be HALLUCINATED (p=0.15 > 0.1); at eps=0.2 dog should be POSITIVE
    table = build_candidate_table_nc(candidates, p_values, signed_distances, epsilons=[0.1, 0.2])
    assert table.count("badge-pos") == 1
    assert table.count("badge-neg") == 1
    print("test_build_candidate_table_nc_multiple_epsilon_columns OK")


def test_build_candidate_table_nc_empty():
    table = build_candidate_table_nc([], {}, {}, epsilons=[0.1])
    assert "No candidate objects" in table
    print("test_build_candidate_table_nc_empty OK")


def test_build_probe_summary_html_missing_probe_rec():
    out = build_probe_summary_html(None, None)
    assert "No probe pool cached" in out
    print("test_build_probe_summary_html_missing_probe_rec OK")


def test_build_probe_summary_html_shows_k_and_shrinkage():
    probe_rec = {"K": 80, "tau_low": 0.3}
    sort_result_dict = {"null_model": {"mean": [0.0, 0.0, 0.0], "shrinkage": 0.15}}
    out = build_probe_summary_html(probe_rec, sort_result_dict)
    assert "80" in out
    assert "0.3" in out
    assert "0.150" in out
    print("test_build_probe_summary_html_shows_k_and_shrinkage OK")


if __name__ == "__main__":
    test_generate_report_nc_basic()
    test_generate_report_nc_requires_nonempty_epsilons()
    test_build_candidate_table_nc_orders_by_p_value()
    test_build_candidate_table_nc_multiple_epsilon_columns()
    test_build_candidate_table_nc_empty()
    test_build_probe_summary_html_missing_probe_rec()
    test_build_probe_summary_html_shows_k_and_shrinkage()
    print("\nALL report_nc.py TESTS PASSED")
