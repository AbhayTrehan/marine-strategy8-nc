"""Run with: python3 tests/test_gdino_scorer.py

Tests GDINOScorer's pure logic (text query formatting, batch looping,
per-query error isolation) without loading the real transformers
GroundingDINO model — that requires downloading weights and is exercised
for real only on the server. We bypass __init__ (which loads the model)
and stub out _score_one instead, mirroring how test_feature_extractors.py
would test OwlViTScorer's batching logic in isolation from the model load.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gdino_scorer import GDINOScorer


def _make_stub_scorer(score_fn):
    """Build a GDINOScorer instance without running __init__ (no model
    load), with _score_one replaced by a controllable stub."""
    scorer = GDINOScorer.__new__(GDINOScorer)
    scorer.device = "cpu"
    scorer.score_threshold = 1e-6
    scorer._score_one = lambda image, name: score_fn(image, name)
    return scorer


def test_query_text_lowercases_and_adds_period():
    assert GDINOScorer._query_text("Dog") == "dog."
    assert GDINOScorer._query_text("cell phone") == "cell phone."
    print("test_query_text_lowercases_and_adds_period OK")


def test_query_text_idempotent_on_existing_period():
    assert GDINOScorer._query_text("dog.") == "dog."
    assert GDINOScorer._query_text("dog. ") == "dog."
    print("test_query_text_idempotent_on_existing_period OK")


def test_query_text_strips_whitespace():
    assert GDINOScorer._query_text("  dog  ") == "dog."
    print("test_query_text_strips_whitespace OK")


def test_score_batch_empty_list():
    scorer = _make_stub_scorer(lambda image, name: 0.5)
    assert scorer.score_batch("fake_image", []) == []
    print("test_score_batch_empty_list OK")


def test_score_batch_preserves_order():
    scores_by_word = {"dog": 0.8, "cat": 0.3, "car": 0.1}
    scorer = _make_stub_scorer(lambda image, name: scores_by_word[name])
    result = scorer.score_batch("fake_image", ["dog", "cat", "car"])
    assert result == [0.8, 0.3, 0.1]
    print("test_score_batch_preserves_order OK")


def test_score_batch_isolates_single_query_failure():
    """One bad query must not crash the whole batch -- it scores 0.0 and
    the rest of the batch proceeds normally."""
    def flaky(image, name):
        if name == "bad_word":
            raise RuntimeError("simulated failure")
        return 0.9

    scorer = _make_stub_scorer(flaky)
    result = scorer.score_batch("fake_image", ["good1", "bad_word", "good2"])
    assert result == [0.9, 0.0, 0.9]
    print("test_score_batch_isolates_single_query_failure OK")


def test_score_map_zips_correctly():
    scores_by_word = {"dog": 0.8, "cat": 0.3}
    scorer = _make_stub_scorer(lambda image, name: scores_by_word[name])
    result = scorer.score_map("fake_image", ["dog", "cat"])
    assert result == {"dog": 0.8, "cat": 0.3}
    print("test_score_map_zips_correctly OK")


if __name__ == "__main__":
    test_query_text_lowercases_and_adds_period()
    test_query_text_idempotent_on_existing_period()
    test_query_text_strips_whitespace()
    test_score_batch_empty_list()
    test_score_batch_preserves_order()
    test_score_batch_isolates_single_query_failure()
    test_score_map_zips_correctly()
    print("\nALL gdino_scorer.py TESTS PASSED")
