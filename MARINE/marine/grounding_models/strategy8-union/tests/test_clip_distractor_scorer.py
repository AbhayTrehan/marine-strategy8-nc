"""Run with: python3 tests/test_clip_distractor_scorer.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from clip_distractor_scorer import (
    ClipSemanticDistractorScorer,
    score_vocabulary_by_semantic_relevance,
    shortlist_by_semantic_relevance,
)


def _make_fake_embeddings(n, d=512, rng=None):
    """Fake L2-normalized embeddings for testing the math."""
    rng = rng or np.random.RandomState(0)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def test_score_vocabulary_basic_semantics():
    vocab = ["chair", "fork", "giraffe", "monitor", "keyboard"]
    d = 64
    rng = np.random.RandomState(1)
    emb = _make_fake_embeddings(len(vocab), d, rng)
    # Make "monitor" very similar to "keyboard" (simulate semantic closeness)
    emb[3] = emb[4] + rng.normal(0, 0.01, d).astype(np.float32)
    emb[3] /= np.linalg.norm(emb[3])

    # Candidate is "keyboard" — "monitor" should score highest
    scores = score_vocabulary_by_semantic_relevance(["keyboard"], vocab, emb)
    assert len(scores) == 5
    monitor_score = scores[vocab.index("monitor")]
    giraffe_score = scores[vocab.index("giraffe")]
    assert monitor_score > giraffe_score, (monitor_score, giraffe_score)
    print("test_score_vocabulary_basic_semantics OK")


def test_score_vocabulary_candidate_not_in_vocab_graceful():
    vocab = ["chair", "fork", "giraffe"]
    emb = _make_fake_embeddings(3, 64)
    scores = score_vocabulary_by_semantic_relevance(["nonexistent_word"], vocab, emb)
    assert np.all(scores == 0.0)
    print("test_score_vocabulary_candidate_not_in_vocab_graceful OK")


def test_score_vocabulary_multiple_candidates():
    vocab = ["a", "b", "c", "d"]
    d = 64
    rng = np.random.RandomState(2)
    emb = _make_fake_embeddings(4, d, rng)
    # Make "a" close to "c", "b" close to "d"
    emb[2] = emb[0] + rng.normal(0, 0.01, d).astype(np.float32)
    emb[2] /= np.linalg.norm(emb[2])
    emb[3] = emb[1] + rng.normal(0, 0.01, d).astype(np.float32)
    emb[3] /= np.linalg.norm(emb[3])
    # Candidates: "a" and "b" — "c" and "d" should both score high (max over candidates)
    scores = score_vocabulary_by_semantic_relevance(["a", "b"], vocab, emb)
    assert scores[2] > 0.9  # "c" close to candidate "a"
    assert scores[3] > 0.9  # "d" close to candidate "b"
    print("test_score_vocabulary_multiple_candidates OK")


def test_clip_semantic_distractor_scorer_callable():
    vocab = ["chair", "fork", "giraffe"]
    emb = _make_fake_embeddings(3, 64, np.random.RandomState(3))
    scorer = ClipSemanticDistractorScorer(vocab, emb, candidate_words=["chair"])
    assert isinstance(scorer("chair"), float)
    assert isinstance(scorer("nonexistent"), float)
    assert scorer("nonexistent") == 0.0
    print("test_clip_semantic_distractor_scorer_callable OK")


def test_shortlist_no_reduction_when_small():
    vocab = ["a", "b", "c", "d", "e"]
    emb = _make_fake_embeddings(5, 64)
    survivors = ["a", "b", "c"]
    result = shortlist_by_semantic_relevance(
        survivors, vocab, emb, candidate_words=["a"], max_shortlist=10
    )
    assert set(result) == set(survivors)
    print("test_shortlist_no_reduction_when_small OK")


def test_shortlist_reduces_to_max_shortlist():
    rng_emb = np.random.RandomState(4)
    vocab = [f"w{i}" for i in range(500)]
    emb = _make_fake_embeddings(500, 64, rng_emb)
    survivors = [f"w{i}" for i in range(400)]
    rng = np.random.default_rng(5)
    result = shortlist_by_semantic_relevance(
        survivors, vocab, emb, candidate_words=["w0"],
        max_shortlist=100, rng=rng,
    )
    assert len(result) == 100
    assert len(set(result)) == 100  # no duplicates
    print("test_shortlist_reduces_to_max_shortlist OK")


def test_shortlist_hard_fraction_respected():
    rng_emb = np.random.RandomState(6)
    d = 64
    vocab = [f"w{i}" for i in range(200)]
    emb = _make_fake_embeddings(200, d, rng_emb)
    # Make w1 very close to w0 (the candidate)
    emb[1] = emb[0] + rng_emb.normal(0, 0.001, d).astype(np.float32)
    emb[1] /= np.linalg.norm(emb[1])

    survivors = [f"w{i}" for i in range(1, 200)]  # exclude w0 (it's the candidate)
    rng = np.random.default_rng(7)
    result = shortlist_by_semantic_relevance(
        survivors, vocab, emb, candidate_words=["w0"],
        max_shortlist=20, hard_fraction=0.75, rng=rng,
    )
    assert len(result) == 20
    # w1 should almost always be in the shortlist (most similar to w0)
    assert "w1" in result
    print("test_shortlist_hard_fraction_respected OK")


if __name__ == "__main__":
    test_score_vocabulary_basic_semantics()
    test_score_vocabulary_candidate_not_in_vocab_graceful()
    test_score_vocabulary_multiple_candidates()
    test_clip_semantic_distractor_scorer_callable()
    test_shortlist_no_reduction_when_small()
    test_shortlist_reduces_to_max_shortlist()
    test_shortlist_hard_fraction_respected()
    print("\nALL clip_distractor_scorer.py TESTS PASSED")
