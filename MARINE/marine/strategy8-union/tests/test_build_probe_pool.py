"""Run with: python3 tests/test_build_probe_pool.py

Tests build_probe_pool.py's ORCHESTRATION logic (wiring probe_sampling.py +
a feature extractor + CLIP distractor scorer + optional GDINO scorer + I/O
together correctly) using stubs -- no torch/GPU/real models needed.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from build_probe_pool import build_probe_pool_cache, load_probe_pool_cache


class _StubOwlViT:
    def score_batch(self, image, words):
        return [(0.0, 0.0) for _ in words]


class _StubFeatureExtractor:
    def __init__(self):
        self.owlvit = _StubOwlViT()

    def extract(self, image, object_names):
        return {w: (0.1, 0.2 + 0.001 * i, 0.01) for i, w in enumerate(object_names)}


class _StubGDINOScorer:
    def score_batch(self, image, object_names):
        return [0.42 for _ in object_names]


def _fake_vocab_embeddings(vocabulary, d=32, seed=0):
    rng = np.random.RandomState(seed)
    emb = rng.normal(size=(len(vocabulary), d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


_TEST_VOCAB = [
    "dog", "puppy", "cat", "kitten", "bench", "kite", "umbrella", "backpack",
    "bicycle", "car", "bus", "train", "boat", "bird", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "chair", "fork", "vase",
]


def test_build_probe_pool_cache_basic_schema_and_exclusion_3d():
    candidate_pool_cache = {
        "img1.jpg": {"candidates": [{"canonical": "dog"}, {"canonical": "bench"}]},
        "img2.jpg": {"candidates": [{"canonical": "cat"}]},
    }
    vocab_embeddings = _fake_vocab_embeddings(_TEST_VOCAB)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            gdino_scorer=None,
            vocabulary=_TEST_VOCAB,
            vocab_embeddings=vocab_embeddings,
            K=5,
            tau_low=0.3,
            output_path=out_path,
        )
        cache = load_probe_pool_cache(out_path)

    assert set(cache.keys()) == {"img1.jpg", "img2.jpg"}
    for img, rec in cache.items():
        assert rec["K"] == 5
        assert len(rec["probes"]) == 5
        words = {p["word"] for p in rec["probes"]}
        assert len(words) == 5
        for p in rec["probes"]:
            # 3D mode (no gdino_scorer): s_gdino key must be ABSENT, not a
            # fake 0.0 -- otherwise fit_null_calibration's dimension
            # auto-detection would misread this as real 4D data.
            assert set(p.keys()) == {"word", "s_det", "s_clip", "s_area"}

    img1_words = {p["word"] for p in cache["img1.jpg"]["probes"]}
    assert "dog" not in img1_words and "puppy" not in img1_words
    assert "bench" not in img1_words

    img2_words = {p["word"] for p in cache["img2.jpg"]["probes"]}
    assert "cat" not in img2_words and "kitten" not in img2_words
    print("test_build_probe_pool_cache_basic_schema_and_exclusion_3d OK")


def test_build_probe_pool_cache_4d_with_gdino():
    candidate_pool_cache = {
        "img1.jpg": {"candidates": [{"canonical": "dog"}]},
    }
    vocab_embeddings = _fake_vocab_embeddings(_TEST_VOCAB)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            gdino_scorer=_StubGDINOScorer(),
            vocabulary=_TEST_VOCAB,
            vocab_embeddings=vocab_embeddings,
            K=5,
            tau_low=0.3,
            output_path=out_path,
        )
        cache = load_probe_pool_cache(out_path)

    for p in cache["img1.jpg"]["probes"]:
        # 4D mode: s_gdino key MUST be present with the stub's real value
        assert set(p.keys()) == {"word", "s_det", "s_clip", "s_area", "s_gdino"}
        assert p["s_gdino"] == 0.42
    print("test_build_probe_pool_cache_4d_with_gdino OK")


def test_build_probe_pool_cache_deterministic_given_seed():
    candidate_pool_cache = {"img1.jpg": {"candidates": [{"canonical": "dog"}]}}
    vocab = [f"word{i}" for i in range(50)]
    vocab_embeddings = _fake_vocab_embeddings(vocab)

    def run_once():
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "probe_pool_cache.jsonl")
            build_probe_pool_cache(
                candidate_pool_cache=candidate_pool_cache,
                image_dir=d,
                feature_extractor=_StubFeatureExtractor(),
                gdino_scorer=None,
                vocabulary=vocab,
                vocab_embeddings=vocab_embeddings,
                K=10,
                tau_low=0.3,
                output_path=out_path,
                seed=123,
            )
            return load_probe_pool_cache(out_path)

    cache_a = run_once()
    cache_b = run_once()
    words_a = [p["word"] for p in cache_a["img1.jpg"]["probes"]]
    words_b = [p["word"] for p in cache_b["img1.jpg"]["probes"]]
    assert words_a == words_b, "same seed should give the same probe pool"
    print("test_build_probe_pool_cache_deterministic_given_seed OK")


def test_build_probe_pool_cache_degrades_gracefully_below_k():
    candidate_pool_cache = {
        "img_bad.jpg": {"candidates": [
            {"canonical": "dog"}, {"canonical": "bench"}, {"canonical": "kite"},
            {"canonical": "umbrella"}, {"canonical": "backpack"},
        ]},
        "img_good.jpg": {"candidates": [{"canonical": "dog"}]},
    }
    vocab = ["dog", "puppy", "bench", "kite", "umbrella", "backpack", "bicycle", "car"]
    vocab_embeddings = _fake_vocab_embeddings(vocab)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            gdino_scorer=None,
            vocabulary=vocab,
            vocab_embeddings=vocab_embeddings,
            K=5,
            tau_low=0.3,
            output_path=out_path,
            min_K=2,
        )
        cache = load_probe_pool_cache(out_path)

    assert "img_bad.jpg" in cache
    assert cache["img_bad.jpg"]["K"] == 2
    assert cache["img_bad.jpg"]["K_requested"] == 5
    assert "img_good.jpg" in cache
    assert cache["img_good.jpg"]["K"] == 5
    print("test_build_probe_pool_cache_degrades_gracefully_below_k OK")


def test_build_probe_pool_cache_unsalvageable_image_skipped_not_crashed():
    candidate_pool_cache = {
        "img_impossible.jpg": {"candidates": [
            {"canonical": "dog"}, {"canonical": "bench"}, {"canonical": "kite"},
            {"canonical": "umbrella"}, {"canonical": "backpack"}, {"canonical": "bicycle"},
            {"canonical": "car"},
        ]},
        "img_good.jpg": {"candidates": [{"canonical": "dog"}]},
    }
    vocab = ["dog", "puppy", "bench", "kite", "umbrella", "backpack", "bicycle", "car"]
    vocab_embeddings = _fake_vocab_embeddings(vocab)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            gdino_scorer=None,
            vocabulary=vocab,
            vocab_embeddings=vocab_embeddings,
            K=5,
            tau_low=0.3,
            output_path=out_path,
            min_K=2,
        )
        cache = load_probe_pool_cache(out_path)

    assert "img_impossible.jpg" not in cache
    assert "img_good.jpg" in cache
    print("test_build_probe_pool_cache_unsalvageable_image_skipped_not_crashed OK")


def test_build_probe_pool_cache_shortlisting_bounds_owlvit_calls():
    """With a large vocabulary, the number of words passed to
    low_conf_score_fn (OWL-ViT) per image should be bounded by
    max_owlvit_candidates, not scale with vocabulary size."""
    candidate_pool_cache = {"img1.jpg": {"candidates": [{"canonical": "dog"}]}}
    large_vocab = [f"word{i}" for i in range(1000)]
    vocab_embeddings = _fake_vocab_embeddings(large_vocab)

    call_sizes = []

    class _CountingOwlViT:
        def score_batch(self, image, words):
            call_sizes.append(len(words))
            return [(0.0, 0.0) for _ in words]

    class _CountingFeatureExtractor(_StubFeatureExtractor):
        def __init__(self):
            self.owlvit = _CountingOwlViT()

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_CountingFeatureExtractor(),
            gdino_scorer=None,
            vocabulary=large_vocab,
            vocab_embeddings=vocab_embeddings,
            K=80,
            tau_low=0.3,
            output_path=out_path,
            max_owlvit_candidates=200,
        )

    assert len(call_sizes) == 1
    assert call_sizes[0] <= 200, f"OWL-ViT called with {call_sizes[0]} words, expected <= 200"
    print("test_build_probe_pool_cache_shortlisting_bounds_owlvit_calls OK")


if __name__ == "__main__":
    import build_probe_pool as _bpp

    class _FakeImage:
        def convert(self, mode):
            return self

    _bpp.Image.open = lambda path: _FakeImage()

    test_build_probe_pool_cache_basic_schema_and_exclusion_3d()
    test_build_probe_pool_cache_4d_with_gdino()
    test_build_probe_pool_cache_deterministic_given_seed()
    test_build_probe_pool_cache_degrades_gracefully_below_k()
    test_build_probe_pool_cache_unsalvageable_image_skipped_not_crashed()
    test_build_probe_pool_cache_shortlisting_bounds_owlvit_calls()
    print("\nALL build_probe_pool.py TESTS PASSED")
