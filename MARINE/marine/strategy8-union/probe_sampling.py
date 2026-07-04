"""
probe_sampling.py
==================

Implements Strategy 8-U-NC's Section 3.1: guaranteed-absent probe pool
sampling. For a given image, this builds the probe set P = {p_1, ..., p_K}
that null_calibration.py's one-class null model is fit from.

The three filters, applied in order, per the spec:

  1. Candidate exclusion. Remove from the vocabulary V every word already
     in O_init, together with its synonyms, hypernyms, and hyponyms, "using
     the same synonym list applied for CHAIR label alignment" -- i.e. the
     curated COCO synonym table this codebase already loads in synonyms.py
     (for COCO-category candidates, exclude the WHOLE synonym group the
     candidate belongs to), PLUS a WordNet synonym/hypernym/hyponym
     expansion (for non-COCO candidates, and as a second layer for COCO
     ones too, since a COCO-category candidate can still have WordNet
     hypernyms/hyponyms not in the curated table).
  2. Low-confidence exclusion. Remove any remaining word whose zero-shot
     detector score against the REAL image already exceeds a permissive
     threshold tau_low -- probes must be absent, not merely undetected at
     high confidence. This step needs the real image + the same detector
     feature_extractors.py uses, so it is deliberately expressed here as a
     pluggable `low_conf_score_fn` rather than a concrete model call (the
     real call site is build_probe_pool.py, run where GPU + images exist).
  3. Distractor bias + uniform fill. Preferentially sample words that
     co-occur frequently with objects already in O_init (in the POPE
     adversarial-split sense; see cooccurrence.py for the real
     COCO-annotation-backed scorer), then fill the remainder of P by
     uniform sampling from whatever's left of V.

K is fixed and image-independent (Section 3.1: "K is fixed... e.g.
K in [50, 100]").
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from nltk.corpus import wordnet as wn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from synonyms import basic_clean, load_coco_synonym_map  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: candidate exclusion (synonyms, hypernyms, hyponyms)
# ---------------------------------------------------------------------------
def _coco_synonym_group_of(word: str, synonyms_map: Dict[str, str]) -> Set[str]:
    """If `word` is (or maps to) a COCO category, return every curated
    synonym-table entry that maps to that SAME category (the "whole
    synonym group", e.g. querying with "puppy" or "dog" both return
    {"dog", "puppy", "beagle", "pup", ...}). Empty set if `word` isn't a
    recognized COCO-category word at all."""
    cleaned = basic_clean(word)
    canonical = synonyms_map.get(cleaned)
    if canonical is None:
        return set()
    return {w for w, c in synonyms_map.items() if c == canonical} | {canonical}


def _wordnet_synonyms_hypernyms_hyponyms(word: str) -> Set[str]:
    """WordNet expansion for one word: every noun-synset lemma name for the
    word itself (synonyms), plus the lemma names of its DIRECT hypernyms
    and DIRECT hyponyms (one level each way -- the paper says "synonyms,
    hypernyms, and hyponyms" without specifying closure depth; one level
    is the natural reading and avoids the expansion set ballooning to
    include very distant, unrelated concepts via deep closure)."""
    if not word:
        return set()
    key = word.replace(" ", "_")
    out: Set[str] = set()
    for syn in wn.synsets(key, pos=wn.NOUN):
        for lemma in syn.lemmas():
            out.add(lemma.name().replace("_", " "))
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                out.add(lemma.name().replace("_", " "))
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                out.add(lemma.name().replace("_", " "))
    return out


def build_exclusion_set(
    candidate_words: Sequence[str],
    synonyms_map: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """Section 3.1, filter 1: the full set of words to remove from V for
    this image, given its candidate pool O_init.

    Returns a set of lowercased, whitespace-cleaned strings ready to be
    compared against `basic_clean(vocab_word)`.
    """
    synonyms_map = synonyms_map if synonyms_map is not None else load_coco_synonym_map()
    exclusion: Set[str] = set()
    for w in candidate_words:
        cleaned = basic_clean(w)
        if not cleaned:
            continue
        exclusion.add(cleaned)
        exclusion |= {basic_clean(x) for x in _coco_synonym_group_of(w, synonyms_map)}
        exclusion |= {basic_clean(x) for x in _wordnet_synonyms_hypernyms_hyponyms(cleaned)}
    exclusion.discard("")
    return exclusion


# ---------------------------------------------------------------------------
# Step 2: low-confidence exclusion
# ---------------------------------------------------------------------------
def filter_low_confidence(
    words: Sequence[str],
    low_conf_score_fn: Callable[[Sequence[str]], Sequence[float]],
    tau_low: float = 0.3,
    batch_size: int = 256,
) -> List[str]:
    """Section 3.1, filter 2: drop any word `w` for which
    low_conf_score_fn([w, ...]) > tau_low -- a permissive threshold, so
    only words with a real, if weak, detection signal are excluded.

    `low_conf_score_fn` is called in batches (not one word at a time) for
    efficiency, matching feature_extractors.py's OwlViTScorer.score_batch
    convention (one real forward pass per batch against the actual image).
    In production this is a thin wrapper around
    `OwlViTScorer.score_batch(image, words)` that returns just the s_det
    component; here it's a pluggable callable so the selection LOGIC can
    be unit-tested without a real model/image.
    """
    if not words:
        return []
    kept: List[str] = []
    for i in range(0, len(words), batch_size):
        batch = list(words[i:i + batch_size])
        scores = list(low_conf_score_fn(batch))
        if len(scores) != len(batch):
            raise ValueError(
                f"low_conf_score_fn returned {len(scores)} scores for {len(batch)} words"
            )
        kept.extend(w for w, s in zip(batch, scores) if s <= tau_low)
    return kept


# ---------------------------------------------------------------------------
# Step 3: distractor-biased sampling + uniform fill
# ---------------------------------------------------------------------------
def _weighted_sample_without_replacement(
    words: Sequence[str], weights: Sequence[float], n: int, rng: np.random.Generator
) -> List[str]:
    n = min(n, len(words))
    if n <= 0:
        return []
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        # no usable signal in this subset -- fall back to uniform
        idx = rng.choice(len(words), size=n, replace=False)
    else:
        p = w / w.sum()
        idx = rng.choice(len(words), size=n, replace=False, p=p)
    return [words[i] for i in idx]


def sample_probe_pool(
    vocabulary: Sequence[str],
    candidate_words: Sequence[str],
    K: int,
    low_conf_score_fn: Callable[[Sequence[str]], Sequence[float]],
    distractor_scorer: Optional[Callable[[str], float]] = None,
    tau_low: float = 0.3,
    synonyms_map: Optional[Dict[str, str]] = None,
    rng: Optional[np.random.Generator] = None,
    min_K: Optional[int] = None,
) -> List[str]:
    """Runs all three filters of Section 3.1 and returns up to K probe
    words for one image.

    Args:
        vocabulary: the fixed object vocabulary V (COCO-80 + RAM++ tag
            list; see load_default_vocabulary).
        candidate_words: this image's O_init canonical object names.
        K: number of probes to return (Section 3.1: e.g. K in [50, 100]).
        low_conf_score_fn: batched s_det scorer against the REAL image
            (filter 2); in production, OwlViTScorer.score_batch bound to
            this image.
        distractor_scorer: optional callable word -> co-occurrence score
            (filter 3); higher = more likely to preferentially sample. If
            None, filter 3 degenerates to pure uniform sampling (a
            legitimate ablation, not an error -- distractor bias is a bonus
            on top of a valid probe pool, not a correctness requirement).
        tau_low: permissive low-confidence threshold (Section 3.1: e.g. 0.3).
        synonyms_map: curated COCO synonym table (defaults to
            synonyms.load_coco_synonym_map()).
        rng: numpy Generator for reproducibility; defaults to a fresh one.
        min_K: if given, and fewer than K (but at least min_K) vocabulary
            words survive filters 1-2, silently use ALL survivors instead
            of raising -- i.e. degrade K for just this image rather than
            failing it outright. This matters mainly for a COCO-80-only
            vocabulary (no --ram_tag_list_path given), where a handful of
            images may not have quite K=80 survivors even though the
            method is otherwise working fine; with the full RAM++ tag
            list (thousands of words) this essentially never triggers.
            If None (default), the original strict behavior applies: raise
            ValueError whenever fewer than K words survive.

    Returns:
        A list of probe words: exactly K unless min_K triggered a
        graceful degradation, in which case it's between min_K and K.
        Raises ValueError if the vocabulary doesn't have enough survivors
        even for min_K (or for K, when min_K is None).
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if min_K is not None and not (0 < min_K <= K):
        raise ValueError(f"min_K must be in (0, K], got min_K={min_K}, K={K}")
    rng = rng if rng is not None else np.random.default_rng()

    # --- Filter 1: candidate exclusion ------------------------------------
    exclusion = build_exclusion_set(candidate_words, synonyms_map=synonyms_map)
    seen: Set[str] = set()
    survivors: List[str] = []
    for w in vocabulary:
        cw = basic_clean(w)
        if not cw or cw in exclusion or cw in seen:
            continue
        seen.add(cw)
        survivors.append(w)

    # --- Filter 2: low-confidence exclusion -------------------------------
    survivors = filter_low_confidence(survivors, low_conf_score_fn, tau_low=tau_low)

    effective_K = K
    if len(survivors) < K:
        floor = min_K if min_K is not None else K
        if len(survivors) < floor:
            raise ValueError(
                f"Only {len(survivors)} vocabulary words survived candidate+"
                f"low-confidence exclusion, need at least "
                f"{floor}{'' if min_K is None else f' (min_K, target K={K})'}. "
                f"Enlarge the vocabulary (e.g. include the full RAM++ tag "
                f"list) or raise tau_low."
            )
        effective_K = len(survivors)  # min_K given and satisfied -> degrade gracefully

    # --- Filter 3: distractor bias + uniform fill -------------------------
    if distractor_scorer is not None:
        scored = [(w, float(distractor_scorer(w))) for w in survivors]
    else:
        scored = [(w, 0.0) for w in survivors]

    distractor_words = [w for w, s in scored if s > 0.0]
    distractor_weights = [s for w, s in scored if s > 0.0]
    n_distractor = min(len(distractor_words), effective_K)
    chosen_distractors = _weighted_sample_without_replacement(
        distractor_words, distractor_weights, n_distractor, rng
    )

    chosen_set = set(chosen_distractors)
    fill_pool = [w for w in survivors if w not in chosen_set]
    n_fill = effective_K - len(chosen_distractors)
    fill_choice = (
        list(rng.choice(np.array(fill_pool, dtype=object), size=n_fill, replace=False))
        if n_fill > 0
        else []
    )

    probes = list(chosen_distractors) + [str(w) for w in fill_choice]
    rng.shuffle(probes)
    return probes


# ---------------------------------------------------------------------------
# Vocabulary construction: COCO-80 + RAM++ tag list
# ---------------------------------------------------------------------------
def coco_80_categories(synonyms_map: Optional[Dict[str, str]] = None) -> List[str]:
    """The 80 canonical MSCOCO category names, derived from the SAME
    curated synonym table synonyms.py/eval_chair.py already use (rather
    than a second hard-coded copy that could drift out of sync)."""
    synonyms_map = synonyms_map if synonyms_map is not None else load_coco_synonym_map()
    return sorted(set(synonyms_map.values()))


def load_ram_tag_list(path: str) -> List[str]:
    """Loads the RAM++ tag list from the `ram` package's bundled data file
    (present on the server where the `ram` package is installed, e.g.
    `.../site-packages/ram/data/ram_tag_list.txt`, one tag per line). Not
    available in a sandbox without the real `ram` package -- callers
    should catch FileNotFoundError and fall back to a smaller vocabulary
    if this path doesn't exist yet."""
    with open(path) as f:
        tags = [line.strip() for line in f if line.strip()]
    return tags


def load_default_vocabulary(
    ram_tag_list_path: Optional[str] = None,
    extra_words: Optional[Sequence[str]] = None,
    synonyms_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Builds V = COCO-80 union RAM++ tag list (Section 3.1), plus any
    extra real, already-observed vocabulary the caller wants folded in
    (e.g. words seen across data/marine_qa/guidance/*.json, as a
    real-data-grounded fallback/supplement when the full RAM++ tag list
    file isn't available in a given environment)."""
    vocab: List[str] = []
    seen: Set[str] = set()

    def _add(words: Sequence[str]):
        for w in words:
            cw = basic_clean(w)
            if cw and cw not in seen:
                seen.add(cw)
                vocab.append(w)

    _add(coco_80_categories(synonyms_map=synonyms_map))
    if ram_tag_list_path is not None:
        _add(load_ram_tag_list(ram_tag_list_path))
    if extra_words is not None:
        _add(extra_words)
    return vocab
