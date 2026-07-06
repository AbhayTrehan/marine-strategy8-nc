"""Run with: python3 tests/test_cooccurrence.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile

from cooccurrence import (
    CooccurrenceScorer,
    build_cooccurrence_table,
    load_cooccurrence_table,
    save_cooccurrence_table,
    score_from_table,
)


def _write_instances_json(path, images):
    """images: list of (image_id, [category_name, ...])"""
    categories = sorted({c for _, cats in images for c in cats})
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    data = {
        "categories": [{"id": cat_id[name], "name": name} for name in categories],
        "annotations": [
            {"image_id": img_id, "category_id": cat_id[c]}
            for img_id, cats in images
            for c in cats
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_build_cooccurrence_table_basic_counts():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "instances_val2014.json")
        _write_instances_json(path, [
            (1, ["dining table", "chair", "chair"]),  # chair appears twice, still one co-occurrence
            (2, ["dining table", "chair"]),
            (3, ["dog", "frisbee"]),
        ])
        table = build_cooccurrence_table([path])
        assert table["dining table"]["chair"] == 2, table
        assert table["chair"]["dining table"] == 2, table  # symmetric
        assert "dog" not in table.get("dining table", {})
        assert table["dog"]["frisbee"] == 1
        print("test_build_cooccurrence_table_basic_counts OK")


def test_build_cooccurrence_table_no_self_pairs():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "instances_val2014.json")
        _write_instances_json(path, [(1, ["dog", "dog"])])
        table = build_cooccurrence_table([path])
        assert "dog" not in table.get("dog", {})
        print("test_build_cooccurrence_table_no_self_pairs OK")


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        inst_path = os.path.join(d, "instances_val2014.json")
        _write_instances_json(inst_path, [(1, ["dog", "frisbee"])])
        table = build_cooccurrence_table([inst_path])
        out_path = os.path.join(d, "cooc.json")
        save_cooccurrence_table(table, out_path)
        loaded = load_cooccurrence_table(out_path)
        assert loaded["dog"]["frisbee"] == 1
        print("test_save_and_load_roundtrip OK")


def test_score_from_table_max_aggregation():
    table = {"chair": {"dining table": 50, "couch": 5}}
    score = score_from_table("chair", ["dining table", "couch"], table, aggregation="max")
    assert score == 50.0
    print("test_score_from_table_max_aggregation OK")


def test_score_from_table_sum_aggregation():
    table = {"chair": {"dining table": 50, "couch": 5}}
    score = score_from_table("chair", ["dining table", "couch"], table, aggregation="sum")
    assert score == 55.0
    print("test_score_from_table_sum_aggregation OK")


def test_score_from_table_unknown_word_returns_zero():
    table = {"chair": {"dining table": 50}}
    assert score_from_table("nonexistent_word", ["dining table"], table) == 0.0
    print("test_score_from_table_unknown_word_returns_zero OK")


def test_score_from_table_no_present_object_overlap_returns_zero():
    table = {"chair": {"dining table": 50}}
    assert score_from_table("chair", ["bicycle", "car"], table) == 0.0
    print("test_score_from_table_no_present_object_overlap_returns_zero OK")


def test_cooccurrence_scorer_bound_to_image():
    table = {"chair": {"dining table": 50}, "fork": {"dining table": 2}}
    scorer = CooccurrenceScorer(table, present_objects=["dining table"])
    assert scorer("chair") == 50.0
    assert scorer("fork") == 2.0
    assert scorer("umbrella") == 0.0
    print("test_cooccurrence_scorer_bound_to_image OK")


if __name__ == "__main__":
    test_build_cooccurrence_table_basic_counts()
    test_build_cooccurrence_table_no_self_pairs()
    test_save_and_load_roundtrip()
    test_score_from_table_max_aggregation()
    test_score_from_table_sum_aggregation()
    test_score_from_table_unknown_word_returns_zero()
    test_score_from_table_no_present_object_overlap_returns_zero()
    test_cooccurrence_scorer_bound_to_image()
    print("\nALL cooccurrence.py TESTS PASSED")
