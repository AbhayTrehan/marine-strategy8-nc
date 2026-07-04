"""
singularize_utils.py
=====================

Centralized, dictionary-aware singularization, used by both
`text_objects.py` (candidate noun extraction from the VLM's own caption)
and `synonyms.py` (candidate canonicalization). Both previously used
TextBlob's `.singularize()` independently, which does blind suffix-
stripping: it removes a trailing "s" whenever a heuristic rule fires,
without ever checking whether the result is an actual word.

This badly mangles common nouns that are already singular but happen to
end in "s": "bus" -> "bu", "tennis" -> "tenni", "grass" -> "gras",
"dress" -> "dres", "glass" -> "glas", "class" -> "clas". Several of these
are real MSCOCO categories ("bus") or common caption words ("grass",
"dress"), so the corruption silently breaks COCO-category matching and
canonicalization for those words whenever they are only mentioned in the
VLM's own free-text caption (RAM/DETR tags happen to not trigger this as
often since their vocabulary rarely ends in a bare "s", but the same
buggy function was applied to them too and can misfire, e.g. RAM tag
"grass" -> "gras").

Because BOTH text_objects.py and synonyms.py called TextBlob's
`.singularize()` independently (candidate_pool.py's pipeline runs
text_objects.py's extraction first, then feeds the result into
synonyms.py's canonicalization, which singularizes *again*), the bug can
also compound: "glasses" -> (text_objects, correct) "glass" ->
(synonyms, buggy) "glas".

Fix: use WordNet's `morphy` lookup instead of suffix-stripping heuristics.
`morphy` only returns a lemma when it can verify it against WordNet's noun
index (falling back through its built-in exception list for irregular
plurals like "mice" -> "mouse", "feet" -> "foot"), so it never invents a
non-word, and it correctly leaves already-singular words unchanged
because there is no plural rule mapping them to a shorter (non-existent)
form. When morphy doesn't recognize a word at all (e.g. it's not an
English noun, or singularization genuinely doesn't apply), we fall back
to returning the word unchanged rather than guessing.

Verified against a battery of ~20 known-good plurals and ~18
already-singular edge cases (see tests/test_singularize_utils.py) with
zero regressions, and confirmed idempotent (applying it twice never
differs from applying it once) -- so it's now safe for text_objects.py
and synonyms.py to each call it independently without compounding.

Known, accepted residual limitation: WordNet's morphy does not special-
case every irregular English plural (e.g. "men" -> "man", "people" ->
"person" are not corrected by morphy itself). This is NOT a corruption
bug like the TextBlob behavior above -- the word is left as a real,
correctly-spelled English word ("men" stays "men") rather than turned
into a non-word ("bu") -- it just occasionally misses a merge opportunity.
In this codebase specifically, "people" is a non-issue because it already
appears as a literal key in the curated MSCOCO synonym table (see
synonyms.py::load_coco_synonym_map), so it still canonicalizes to
"person" correctly regardless of this function.
"""

from __future__ import annotations

from functools import lru_cache

from nltk.corpus import wordnet as wn


@lru_cache(maxsize=8192)
def robust_singularize_word(word: str) -> str:
    """Singularize a single token.

    Returns `word` unchanged if it is empty, not purely alphabetic, or if
    WordNet's morphy lookup does not recognize it as a noun form -- i.e.
    we only ever change a word when we can verify the result against
    WordNet's dictionary, never via a blind suffix strip.
    """
    if not word or not word.isalpha():
        return word
    lemma = wn.morphy(word.lower(), wn.NOUN)
    return lemma if lemma is not None else word


@lru_cache(maxsize=8192)
def robust_singularize_phrase(phrase: str) -> str:
    """Singularize only the LAST token of a (possibly multi-word) phrase.

    Matches the existing convention at both call sites: for a compound
    like "birthday cakes" we want "birthday cake" (only the head noun
    singularized), not also mangling the modifier "birthday".
    """
    if not phrase:
        return phrase
    parts = phrase.split(" ")
    parts[-1] = robust_singularize_word(parts[-1])
    return " ".join(parts)
