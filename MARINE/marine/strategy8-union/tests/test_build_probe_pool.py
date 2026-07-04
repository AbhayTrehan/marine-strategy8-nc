"""Run with: python3 tests/test_build_probe_pool.py

Tests build_probe_pool.py's ORCHESTRATION logic (does it wire
probe_sampling.py + a feature extractor + I/O together correctly) using a
stub feature extractor and stub "images" (plain strings) -- no torch/GPU/
real OWL-ViT/CLIP needed, since those are exercised for real only on the
server. This mirrors how tests/test_candidate_pool.py would ideally be
structured, except that file imports candidate_pool.py directly, which
pulls in `torch` at module level; build_probe_pool.py deliberately keeps
`torch`-requiring imports inside main() only, specifically so this
wiring can be tested without torch installed.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_probe_pool import build_probe_pool_cache, load_probe_pool_cache


class _StubOwlViT:
    """Every word gets s_det=0.0 (never weakly detected), s_area=0.0."""

    def score_batch(self, image, words):
        return [(0.0, 0.0) for _ in words]


class _StubFeatureExtractor:
    """Deterministic, fake (s_det, s_clip, s_area) per word -- enough to
    verify the OUTPUT SCHEMA and WIRING, not real vision-model numbers."""

    def __init__(self):
        self.owlvit = _StubOwlViT()

    def extract(self, image, object_names):
        return {w: (0.1, 0.2 + 0.001 * i, 0.01) for i, w in enumerate(object_names)}


def test_build_probe_pool_cache_basic_schema_and_exclusion():
    candidate_pool_cache = {
        "img1.jpg": {"candidates": [{"canonical": "dog"}, {"canonical": "bench"}]},
        "img2.jpg": {"candidates": [{"canonical": "cat"}]},
    }
    vocabulary = [
        "dog", "puppy", "cat", "kitten", "bench", "kite", "umbrella", "backpack",
        "bicycle", "car", "bus", "train", "boat", "bird", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "chair", "fork", "vase",
    ]

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,  # never actually read -- PIL.Image.open is stubbed out below
            feature_extractor=_StubFeatureExtractor(),
            vocabulary=vocabulary,
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
        assert len(words) == 5  # no duplicates
        for p in rec["probes"]:
            assert set(p.keys()) == {"word", "s_det", "s_clip", "s_area"}

    img1_words = {p["word"] for p in cache["img1.jpg"]["probes"]}
    assert "dog" not in img1_words and "puppy" not in img1_words  # candidate + synonym excluded
    assert "bench" not in img1_words

    img2_words = {p["word"] for p in cache["img2.jpg"]["probes"]}
    assert "cat" not in img2_words and "kitten" not in img2_words
    print("test_build_probe_pool_cache_basic_schema_and_exclusion OK")


def test_build_probe_pool_cache_deterministic_given_seed():
    candidate_pool_cache = {"img1.jpg": {"candidates": [{"canonical": "dog"}]}}
    vocabulary = [f"word{i}" for i in range(50)]

    def run_once():
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "probe_pool_cache.jsonl")
            build_probe_pool_cache(
                candidate_pool_cache=candidate_pool_cache,
                image_dir=d,
                feature_extractor=_StubFeatureExtractor(),
                vocabulary=vocabulary,
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


def test_build_probe_pool_cache_skips_bad_image_without_crashing_batch():
    # img_bad's candidates ("dog", "puppy", "bench", ...) exclude almost the
    # entire tiny vocabulary, leaving too few survivors even for min_K;
    # img_good has a small candidate list and plenty of survivors.
    candidate_pool_cache = {
        "img_bad.jpg": {"candidates": [
            {"canonical": "dog"}, {"canonical": "bench"}, {"canonical": "kite"},
            {"canonical": "umbrella"}, {"canonical": "backpack"},
        ]},
        "img_good.jpg": {"candidates": [{"canonical": "dog"}]},
    }
    vocab = ["dog", "puppy", "bench", "kite", "umbrella", "backpack", "bicycle", "car"]

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            vocabulary=vocab,
            K=5,
            tau_low=0.3,
            output_path=out_path,
            min_K=2,
        )
        cache = load_probe_pool_cache(out_path)

    # img_bad excludes dog/bench/kite/umbrella/backpack, leaving only
    # bicycle+car (2 words) -- exactly at min_K=2, so it succeeds with a
    # degraded K rather than being skipped.
    assert "img_bad.jpg" in cache
    assert cache["img_bad.jpg"]["K"] == 2
    assert cache["img_bad.jpg"]["K_requested"] == 5
    assert "img_good.jpg" in cache
    assert cache["img_good.jpg"]["K"] == 5
    print("test_build_probe_pool_cache_skips_bad_image_without_crashing_batch OK")


def test_build_probe_pool_cache_truly_unsalvageable_image_is_skipped_not_crashed():
    candidate_pool_cache = {
        "img_impossible.jpg": {"candidates": [
            {"canonical": "dog"}, {"canonical": "bench"}, {"canonical": "kite"},
            {"canonical": "umbrella"}, {"canonical": "backpack"}, {"canonical": "bicycle"},
            {"canonical": "car"},
        ]},
        "img_good.jpg": {"candidates": [{"canonical": "dog"}]},
    }
    vocab = ["dog", "puppy", "bench", "kite", "umbrella", "backpack", "bicycle", "car"]

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "probe_pool_cache.jsonl")
        # every vocab word except "puppy" is excluded as a candidate; only
        # 1 survivor remains, below min_K=2 -> img_impossible must be
        # skipped, but img_good must still succeed (no crash).
        build_probe_pool_cache(
            candidate_pool_cache=candidate_pool_cache,
            image_dir=d,
            feature_extractor=_StubFeatureExtractor(),
            vocabulary=vocab,
            K=5,
            tau_low=0.3,
            output_path=out_path,
            min_K=2,
        )
        cache = load_probe_pool_cache(out_path)

    assert "img_impossible.jpg" not in cache
    assert "img_good.jpg" in cache
    print("test_build_probe_pool_cache_truly_unsalvageable_image_is_skipped_not_crashed OK")


if __name__ == "__main__":
    # PIL.Image.open would fail against the dummy temp dirs above (no real
    # image files) -- monkeypatch it to a no-op stub for this test module,
    # since build_probe_pool_cache only needs SOMETHING to pass through to
    # feature_extractor.extract, which is itself stubbed above.
    import build_probe_pool as _bpp

    class _FakeImage:
        def convert(self, mode):
            return self

    _bpp.Image.open = lambda path: _FakeImage()

    test_build_probe_pool_cache_basic_schema_and_exclusion()
    test_build_probe_pool_cache_deterministic_given_seed()
    test_build_probe_pool_cache_skips_bad_image_without_crashing_batch()
    test_build_probe_pool_cache_truly_unsalvageable_image_is_skipped_not_crashed()
    print("\nALL build_probe_pool.py TESTS PASSED")
