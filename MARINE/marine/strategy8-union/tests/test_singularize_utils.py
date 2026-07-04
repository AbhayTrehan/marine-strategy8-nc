"""
Run with: python3 tests/test_singularize_utils.py
(plain-assert script, no pytest dependency required)

Regression tests for the TextBlob -> WordNet-morphy singularization fix.
Every "already singular" word below is a real word that a 500-real-caption
audit found TextBlob's `.singularize()` corrupting into a non-word (see
singularize_utils.py's module docstring for the full writeup); every
"plural" word is a sanity check that normal pluralization still works.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from singularize_utils import robust_singularize_word, robust_singularize_phrase


def test_already_singular_words_are_not_corrupted():
    # These are the exact real words found corrupted by TextBlob during
    # the audit of the 500-image candidate_pool_cache.jsonl captions.
    already_singular = [
        "bus", "tennis", "glass", "grass", "dress", "class", "lens",
        "chess", "virus", "campus", "bias", "gas", "plus", "octopus",
        "walrus", "curl", "pass",
    ]
    for w in already_singular:
        got = robust_singularize_word(w)
        assert got == w, f"{w!r} was changed to {got!r} (should stay unchanged)"
    print("test_already_singular_words_are_not_corrupted OK")


def test_normal_plurals_still_singularize():
    cases = {
        "dogs": "dog", "cats": "cat", "benches": "bench", "boxes": "box",
        "children": "child", "women": "woman", "buses": "bus",
        "classes": "class", "leaves": "leaf", "knives": "knife",
        "babies": "baby", "skis": "ski", "tomatoes": "tomato",
        "feet": "foot", "mice": "mouse", "sandwiches": "sandwich",
        "watches": "watch", "umbrellas": "umbrella", "bicycles": "bicycle",
        "dresses": "dress",
    }
    for plural, expected in cases.items():
        got = robust_singularize_word(plural)
        assert got == expected, f"{plural!r} -> {got!r}, expected {expected!r}"
    print("test_normal_plurals_still_singularize OK")


def test_idempotent():
    words = [
        "bus", "tennis", "glass", "grass", "dress", "dogs", "cats",
        "children", "feet", "mice", "curl", "pass",
    ]
    for w in words:
        once = robust_singularize_word(w)
        twice = robust_singularize_word(once)
        assert once == twice, f"not idempotent: {w!r} -> {once!r} -> {twice!r}"
    print("test_idempotent OK")


def test_phrase_only_singularizes_last_token():
    assert robust_singularize_phrase("dining tables") == "dining table"
    assert robust_singularize_phrase("birthday cakes") == "birthday cake"
    # modifier itself must not be mangled even if it looks pluralizable
    assert robust_singularize_phrase("glasses cases") == "glasses case"
    print("test_phrase_only_singularizes_last_token OK")


def test_double_application_matches_single_application():
    # Regression for the specific compounding bug: text_objects.py
    # singularizes once, then synonyms.py singularizes again downstream.
    # With the fix this must be a no-op the second time.
    for w in ["glasses", "bus", "tennis", "grass", "dress"]:
        once = robust_singularize_word(w)
        twice = robust_singularize_word(once)
        assert once == twice, (w, once, twice)
    print("test_double_application_matches_single_application OK")


def test_non_alpha_and_empty_left_untouched():
    assert robust_singularize_word("") == ""
    assert robust_singularize_word("2024") == "2024"
    assert robust_singularize_word("cell-phone") == "cell-phone"  # not purely alphabetic
    print("test_non_alpha_and_empty_left_untouched OK")


if __name__ == "__main__":
    test_already_singular_words_are_not_corrupted()
    test_normal_plurals_still_singularize()
    test_idempotent()
    test_phrase_only_singularizes_last_token()
    test_double_application_matches_single_application()
    test_non_alpha_and_empty_left_untouched()
    print("\nALL singularize_utils.py TESTS PASSED")
