"""
Run with: python3 tests/test_synonyms.py
(plain-assert script, no pytest dependency required)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synonyms import UnionCanonicalizer, build_raw_mentions, basic_clean, singularize, coco_canonical, load_coco_synonym_map


def _by_canonical(cands):
    return {c.canonical: c for c in cands}


def test_basic_clean_and_singularize():
    assert basic_clean("  A Dog ") == "dog"
    assert basic_clean("the Cell-Phone!!") == "cell phone"
    assert singularize("dogs") == "dog"
    assert singularize("dining tables") == "dining table"
    print("test_basic_clean_and_singularize OK")


def test_coco_canonical_direct_and_head_fallback():
    m = load_coco_synonym_map()
    assert coco_canonical("telephone", m) == "cell phone"
    assert coco_canonical("a young dog", m) == "dog"  # head-word fallback
    assert coco_canonical("xyzzy-not-an-object", m) is None
    print("test_coco_canonical_direct_and_head_fallback OK")


def test_coco_synonyms_merge_across_sources():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["telephone"],
        detr_tags=["cell phone"],
        vlm_objects=["a mobile phone"],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    c = cands[0]
    assert c.canonical == "cell phone"
    assert c.sources == {"ram", "detr", "vlm"}
    assert c.is_coco_category is True
    print("test_coco_synonyms_merge_across_sources OK")


def test_wordnet_merge_for_noncoco_words():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["cloth", "fabric", "material"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    assert cands[0].is_coco_category is False
    assert set(cands[0].raw_mentions) == {"cloth", "fabric", "material"}
    print("test_wordnet_merge_for_noncoco_words OK")


def test_head_noun_merge():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["glass vase", "vase"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    # vase is a COCO category, so head-noun-merging a non-coco "glass vase"
    # into it should make the cluster coco-canonical "vase"
    assert cands[0].canonical == "vase"
    assert cands[0].is_coco_category is True
    print("test_head_noun_merge OK")


def test_distinct_coco_categories_never_cross_merge():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["cat", "dog"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    canonicals = {c.canonical for c in cands}
    assert canonicals == {"cat", "dog"}, canonicals
    print("test_distinct_coco_categories_never_cross_merge OK")


def test_empty_pool():
    uc = UnionCanonicalizer()
    assert uc.canonicalize_pool([]) == []
    raws = build_raw_mentions(ram_tags=["", "   ", "123"], detr_tags=[], vlm_objects=[])
    assert uc.canonicalize_pool(raws) == []
    print("test_empty_pool OK")


def test_singularize_does_not_corrupt_already_singular_words():
    # Real RAM tags from the audit: TextBlob's singularize() turned these
    # into non-words ("gras", "dres") even though they were already
    # singular and are real, common caption words.
    for w in ["grass", "dress", "glass", "bus", "tennis", "class"]:
        assert singularize(w) == w, (w, singularize(w))
    print("test_singularize_does_not_corrupt_already_singular_words OK")


def test_singularize_safe_to_apply_twice():
    # Regression for the specific compounding bug: candidate_pool.py feeds
    # text_objects.py's (already singularized) output into this module's
    # singularize() a second time. "glasses" -> "glass" (text_objects) ->
    # must NOT become "glas" here.
    once = singularize("glasses")
    twice = singularize(once)
    assert once == twice, (once, twice)
    print("test_singularize_safe_to_apply_twice OK")


def test_bus_from_vlm_only_is_kept_as_coco_category():
    # Before the fix, a VLM-only mention of "bus" was corrupted to "bu" by
    # text_objects.py, which does not match the COCO synonym table and is
    # not a recognized WordNet physical-entity noun, so it was silently
    # dropped from the candidate pool entirely -- a real MSCOCO category
    # vanishing whenever RAM/DETR didn't independently also tag it.
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(ram_tags=[], detr_tags=[], vlm_objects=["bus"])
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    assert cands[0].canonical == "bus", cands
    assert cands[0].is_coco_category is True
    print("test_bus_from_vlm_only_is_kept_as_coco_category OK")


def test_grass_from_ram_keeps_correct_canonical_label():
    # Real RAM tag "grass" was previously canonicalized to "gras". It's not
    # a COCO category, but it IS a real WordNet physical object, so it
    # survives the physical-object filter either way -- the bug was purely
    # in the (mis-)spelling of its canonical label.
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(ram_tags=["grass"], detr_tags=[], vlm_objects=[])
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    assert cands[0].canonical == "grass", cands
    print("test_grass_from_ram_keeps_correct_canonical_label OK")


def test_curl_from_ram_is_filtered_as_non_object():
    # Real RAM++ tag "curl" (emitted for a curled-up cat's posture) has a
    # WordNet noun sense ("a ringlet of hair") that IS a physical_entity, so
    # `_has_physical_noun_synset` alone lets it through. Confirmed via a
    # 500-real-caption audit: across all three candidate_pool_cache.jsonl
    # runs, "curl" was the only RAM++ tag to survive the physical-object
    # filter while never denoting a real, distinct scene object -- it needs
    # the explicit blocklist entry, same as "comfort"/"fill".
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(ram_tags=["cat", "curl"], detr_tags=[], vlm_objects=[])
    cands = uc.canonicalize_pool(raws)
    canonicals = {c.canonical for c in cands}
    assert "curl" not in canonicals, canonicals
    assert "cat" in canonicals, canonicals
    print("test_curl_from_ram_is_filtered_as_non_object OK")


def test_provenance_tracking():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["dog"],
        detr_tags=[],
        vlm_objects=["a dog", "a dog"],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1
    assert cands[0].sources == {"ram", "vlm"}
    print("test_provenance_tracking OK")


if __name__ == "__main__":
    test_basic_clean_and_singularize()
    test_coco_canonical_direct_and_head_fallback()
    test_coco_synonyms_merge_across_sources()
    test_wordnet_merge_for_noncoco_words()
    test_head_noun_merge()
    test_distinct_coco_categories_never_cross_merge()
    test_singularize_does_not_corrupt_already_singular_words()
    test_singularize_safe_to_apply_twice()
    test_bus_from_vlm_only_is_kept_as_coco_category()
    test_grass_from_ram_keeps_correct_canonical_label()
    test_curl_from_ram_is_filtered_as_non_object()
    test_empty_pool()
    test_provenance_tracking()
    print("\nALL synonyms.py TESTS PASSED")
