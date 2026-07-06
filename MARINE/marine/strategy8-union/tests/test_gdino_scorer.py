"""Run with: python3 tests/test_gdino_scorer.py

Tests GDINOScorer's pure tensor math (gdino_postprocess) with synthetic
torch tensors, and its batching/error-isolation logic with a stub
subclass. gdino_postprocess needs real torch (available here); the
GDINOScorer class itself (model loading, real forward pass) needs GPU +
downloaded weights and is exercised for real only on the server.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from gdino_scorer import GDINOScorer, gdino_postprocess


def test_gdino_postprocess_empty_logits():
    logits = torch.zeros((0, 10))
    assert gdino_postprocess(logits) == 0.0
    print("test_gdino_postprocess_empty_logits OK")


def test_gdino_postprocess_max_over_queries_and_tokens():
    # 3 queries, 4 text-token positions. Query 1 has the strongest signal
    # at token position 2 (logit=5.0 -> sigmoid ~0.9933).
    logits = torch.tensor([
        [0.1, 0.2, 0.0, -1.0],
        [-2.0, -2.0, 5.0, -2.0],
        [0.5, 0.5, 0.5, 0.5],
    ])
    result = gdino_postprocess(logits)
    expected = torch.sigmoid(torch.tensor(5.0)).item()
    assert abs(result - expected) < 1e-4, (result, expected)
    print("test_gdino_postprocess_max_over_queries_and_tokens OK")


def test_gdino_postprocess_all_low_confidence():
    logits = torch.full((5, 8), -10.0)  # sigmoid(-10) ~ 4.5e-5, near zero
    result = gdino_postprocess(logits)
    assert 0.0 <= result < 0.001
    print("test_gdino_postprocess_all_low_confidence OK")


def test_gdino_postprocess_all_high_confidence():
    logits = torch.full((5, 8), 10.0)  # sigmoid(10) ~ 0.9999
    result = gdino_postprocess(logits)
    assert result > 0.999
    print("test_gdino_postprocess_all_high_confidence OK")


def test_gdino_postprocess_single_query_single_token():
    logits = torch.tensor([[2.0]])
    result = gdino_postprocess(logits)
    expected = torch.sigmoid(torch.tensor(2.0)).item()
    assert abs(result - expected) < 1e-6
    print("test_gdino_postprocess_single_query_single_token OK")


def test_gdino_postprocess_matches_manual_sigmoid_max():
    """Cross-check against the fully manual computation for a random
    tensor, to catch any dim-ordering mistakes."""
    torch.manual_seed(0)
    logits = torch.randn(20, 15)
    result = gdino_postprocess(logits)
    manual = torch.sigmoid(logits).max(dim=-1).values.max().item()
    assert abs(result - manual) < 1e-6
    print("test_gdino_postprocess_matches_manual_sigmoid_max OK")


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


def _make_stub_scorer(score_fn):
    scorer = GDINOScorer.__new__(GDINOScorer)
    scorer.device = "cpu"
    scorer._score_one = lambda image, name: score_fn(image, name)
    return scorer


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
    test_gdino_postprocess_empty_logits()
    test_gdino_postprocess_max_over_queries_and_tokens()
    test_gdino_postprocess_all_low_confidence()
    test_gdino_postprocess_all_high_confidence()
    test_gdino_postprocess_single_query_single_token()
    test_gdino_postprocess_matches_manual_sigmoid_max()
    test_query_text_lowercases_and_adds_period()
    test_query_text_idempotent_on_existing_period()
    test_query_text_strips_whitespace()
    test_score_batch_empty_list()
    test_score_batch_preserves_order()
    test_score_batch_isolates_single_query_failure()
    test_score_map_zips_correctly()
    print("\nALL gdino_scorer.py TESTS PASSED")
