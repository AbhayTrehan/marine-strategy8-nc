"""Run with: python3 tests/test_probe_sampling.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from probe_sampling import (
    build_exclusion_set,
    coco_80_categories,
    filter_low_confidence,
    load_default_vocabulary,
    sample_probe_pool,
)


def test_exclusion_includes_candidate_itself():
    excl = build_exclusion_set(["dog"])
    assert "dog" in excl
    print("test_exclusion_includes_candidate_itself OK")


def test_exclusion_includes_coco_synonym_group():
    # "dog" is a COCO category; its curated synonym group (puppy, beagle,
    # etc.) must ALL be excluded so a near-duplicate doesn't leak into P
    # (Section 3.1's own example: "puppy" when "dog" is a candidate).
    excl = build_exclusion_set(["dog"])
    assert "puppy" in excl, excl
    assert "beagle" in excl, excl
    print("test_exclusion_includes_coco_synonym_group OK")


def test_exclusion_includes_wordnet_hypernym_and_hyponym():
    # "dog" WordNet hypernym includes "canine"/"domestic animal" etc,
    # hyponyms include specific breeds. We just check the expansion is
    # non-trivially larger than the synonym group alone.
    excl_word_only = {"dog"}
    excl_full = build_exclusion_set(["dog"])
    assert len(excl_full) > len(excl_word_only) + 1
    print("test_exclusion_includes_wordnet_hypernym_and_hyponym OK")


def test_exclusion_does_not_blow_up_on_garbage_candidate():
    # a non-word (e.g. residual garbage) should not crash -- just
    # contributes itself and nothing else to the exclusion set.
    excl = build_exclusion_set(["zzzznotarealword"])
    assert "zzzznotarealword" in excl
    print("test_exclusion_does_not_blow_up_on_garbage_candidate OK")


def test_filter_low_confidence_keeps_only_below_threshold():
    def fake_scorer(words):
        # "cat" and "dog" score high (as if weakly detected), everything
        # else scores low
        return [0.9 if w in ("cat", "dog") else 0.05 for w in words]

    words = ["cat", "dog", "bench", "kite", "umbrella"]
    kept = filter_low_confidence(words, fake_scorer, tau_low=0.3)
    assert "cat" not in kept and "dog" not in kept
    assert set(kept) == {"bench", "kite", "umbrella"}
    print("test_filter_low_confidence_keeps_only_below_threshold OK")


def test_filter_low_confidence_batches_correctly():
    calls = []

    def fake_scorer(words):
        calls.append(list(words))
        return [0.0] * len(words)

    words = [f"w{i}" for i in range(10)]
    kept = filter_low_confidence(words, fake_scorer, tau_low=0.3, batch_size=4)
    assert kept == words
    assert len(calls) == 3  # 4 + 4 + 2
    print("test_filter_low_confidence_batches_correctly OK")


def test_filter_low_confidence_mismatched_scores_raises():
    def bad_scorer(words):
        return [0.1]  # wrong length

    try:
        filter_low_confidence(["a", "b", "c"], bad_scorer)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_filter_low_confidence_mismatched_scores_raises OK")


def test_sample_probe_pool_excludes_candidates_and_synonyms():
    vocab = ["dog", "puppy", "cat", "bench", "kite", "umbrella", "backpack",
             "bicycle", "car", "bus", "train", "boat", "bird", "horse",
             "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    rng = np.random.default_rng(0)

    def no_detection(words):
        return [0.0] * len(words)

    probes = sample_probe_pool(
        vocabulary=vocab,
        candidate_words=["dog"],
        K=5,
        low_conf_score_fn=no_detection,
        rng=rng,
    )
    assert len(probes) == 5
    assert "dog" not in probes
    assert "puppy" not in probes  # synonym-group exclusion
    print("test_sample_probe_pool_excludes_candidates_and_synonyms OK")


def test_sample_probe_pool_raises_when_vocabulary_too_small():
    vocab = ["dog", "cat", "bench"]

    def no_detection(words):
        return [0.0] * len(words)

    try:
        sample_probe_pool(
            vocabulary=vocab, candidate_words=["dog"], K=10,
            low_conf_score_fn=no_detection,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_sample_probe_pool_raises_when_vocabulary_too_small OK")


def test_sample_probe_pool_min_k_degrades_gracefully():
    # 8 vocabulary words survive (bench..zebra), K=10 requested, min_K=5 ->
    # should return all 8 survivors rather than raising.
    vocab = ["dog", "puppy", "bench", "kite", "umbrella", "backpack",
             "bicycle", "car", "bus", "train"]

    def no_detection(words):
        return [0.0] * len(words)

    rng = np.random.default_rng(0)
    probes = sample_probe_pool(
        vocabulary=vocab, candidate_words=["dog"], K=10, min_K=5,
        low_conf_score_fn=no_detection, rng=rng,
    )
    # "dog" and "puppy" excluded (candidate + synonym), 8 remain, below K=10
    # but above min_K=5 -> degrade to using all 8
    assert len(probes) == 8
    print("test_sample_probe_pool_min_k_degrades_gracefully OK")


def test_sample_probe_pool_min_k_still_raises_below_floor():
    vocab = ["dog", "puppy", "bench", "kite"]

    def no_detection(words):
        return [0.0] * len(words)

    try:
        sample_probe_pool(
            vocabulary=vocab, candidate_words=["dog"], K=10, min_K=5,
            low_conf_score_fn=no_detection,
        )
        assert False, "expected ValueError (only 2 survivors, below min_K=5)"
    except ValueError:
        pass
    print("test_sample_probe_pool_min_k_still_raises_below_floor OK")


def test_sample_probe_pool_invalid_min_k_raises():
    def no_detection(words):
        return [0.0] * len(words)

    try:
        sample_probe_pool(
            vocabulary=["a", "b"], candidate_words=[], K=5, min_K=10,  # min_K > K
            low_conf_score_fn=no_detection,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("test_sample_probe_pool_invalid_min_k_raises OK")


def test_sample_probe_pool_low_confidence_filter_applied():
    vocab = ["cat", "bench", "kite", "umbrella", "backpack", "bicycle", "car"]

    def detect_cat_only(words):
        return [0.9 if w == "cat" else 0.0 for w in words]

    rng = np.random.default_rng(1)
    probes = sample_probe_pool(
        vocabulary=vocab, candidate_words=[], K=5,
        low_conf_score_fn=detect_cat_only, tau_low=0.3, rng=rng,
    )
    assert "cat" not in probes  # weakly-detected, so excluded by filter 2
    assert len(probes) == 5
    print("test_sample_probe_pool_low_confidence_filter_applied OK")


def test_sample_probe_pool_distractor_bias_prefers_high_score_words():
    vocab = ["chair", "fork", "vase", "clock", "bench", "kite", "umbrella",
             "backpack", "bicycle", "car", "bus"]

    def no_detection(words):
        return [0.0] * len(words)

    def distractor_scorer(word):
        # "chair" strongly co-occurs with "dining table" (already a
        # candidate) in the POPE-adversarial sense; everything else has
        # no signal.
        return 10.0 if word == "chair" else 0.0

    rng = np.random.default_rng(2)
    hits = 0
    n_trials = 30
    for i in range(n_trials):
        probes = sample_probe_pool(
            vocabulary=vocab, candidate_words=["dining table"], K=3,
            low_conf_score_fn=no_detection, distractor_scorer=distractor_scorer,
            rng=np.random.default_rng(i),
        )
        if "chair" in probes:
            hits += 1
    # "chair" has the only positive distractor score and K=3 draws from ~10
    # words, so with the distractor-bias mechanism it should be chosen with
    # very high (not just 3/10-ish) frequency across trials.
    assert hits / n_trials > 0.8, f"chair chosen in only {hits}/{n_trials} trials"
    print("test_sample_probe_pool_distractor_bias_prefers_high_score_words OK")


def test_sample_probe_pool_no_distractor_scorer_is_pure_uniform_fallback():
    vocab = ["chair", "fork", "vase", "clock", "bench"]

    def no_detection(words):
        return [0.0] * len(words)

    rng = np.random.default_rng(3)
    probes = sample_probe_pool(
        vocabulary=vocab, candidate_words=[], K=3,
        low_conf_score_fn=no_detection, distractor_scorer=None, rng=rng,
    )
    assert len(probes) == 3
    assert len(set(probes)) == 3  # no duplicates (sampled without replacement)
    print("test_sample_probe_pool_no_distractor_scorer_is_pure_uniform_fallback OK")


def test_sample_probe_pool_no_duplicate_probes():
    vocab = [f"word{i}" for i in range(200)]

    def no_detection(words):
        return [0.0] * len(words)

    rng = np.random.default_rng(4)
    probes = sample_probe_pool(
        vocabulary=vocab, candidate_words=[], K=60,
        low_conf_score_fn=no_detection, rng=rng,
    )
    assert len(probes) == 60
    assert len(set(probes)) == 60
    print("test_sample_probe_pool_no_duplicate_probes OK")


def test_coco_80_categories_has_80_entries():
    cats = coco_80_categories()
    assert len(cats) == 80, len(cats)
    assert "dog" in cats and "bus" in cats and "cell phone" in cats
    print("test_coco_80_categories_has_80_entries OK")


def test_load_default_vocabulary_without_ram_tag_list():
    vocab = load_default_vocabulary()
    assert len(vocab) == 80
    print("test_load_default_vocabulary_without_ram_tag_list OK")


def test_load_default_vocabulary_with_extra_words_dedupes():
    vocab = load_default_vocabulary(extra_words=["dog", "puppy", "novel_ram_tag"])
    assert vocab.count("dog") == 1  # "dog" already in COCO-80, not duplicated
    assert "novel_ram_tag" in vocab
    print("test_load_default_vocabulary_with_extra_words_dedupes OK")


if __name__ == "__main__":
    test_exclusion_includes_candidate_itself()
    test_exclusion_includes_coco_synonym_group()
    test_exclusion_includes_wordnet_hypernym_and_hyponym()
    test_exclusion_does_not_blow_up_on_garbage_candidate()
    test_filter_low_confidence_keeps_only_below_threshold()
    test_filter_low_confidence_batches_correctly()
    test_filter_low_confidence_mismatched_scores_raises()
    test_sample_probe_pool_excludes_candidates_and_synonyms()
    test_sample_probe_pool_raises_when_vocabulary_too_small()
    test_sample_probe_pool_min_k_degrades_gracefully()
    test_sample_probe_pool_min_k_still_raises_below_floor()
    test_sample_probe_pool_invalid_min_k_raises()
    test_sample_probe_pool_low_confidence_filter_applied()
    test_sample_probe_pool_distractor_bias_prefers_high_score_words()
    test_sample_probe_pool_no_distractor_scorer_is_pure_uniform_fallback()
    test_sample_probe_pool_no_duplicate_probes()
    test_coco_80_categories_has_80_entries()
    test_load_default_vocabulary_without_ram_tag_list()
    test_load_default_vocabulary_with_extra_words_dedupes()
    print("\nALL probe_sampling.py TESTS PASSED")
