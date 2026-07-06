"""
clip_distractor_scorer.py
==========================

Training-free replacement for cooccurrence.py's corpus-derived co-occurrence
table. Instead of mining "which COCO categories tend to appear together"
from 120K ground-truth annotated images (which makes the method corpus-
dependent and limited to the 80 COCO categories), this module uses CLIP's
OWN pretrained text encoder — already loaded by the pipeline for s_clip —
to measure semantic relatedness between vocabulary words and the current
image's candidate objects, purely from CLIP's frozen weights with zero
additional fitting.

How it works:
  1. ONCE (at startup, ~30 sec): embed the entire vocabulary V through
     CLIP's text encoder → a (|V|, D) matrix of L2-normalized vectors,
     saved to disk as a .npy cache for instant reuse on subsequent runs.
  2. PER IMAGE (microseconds): given this image's candidate words, look
     up their precomputed text embeddings, compute cosine similarity
     between each vocabulary word and the closest candidate, return that
     as the "distractor score" — higher means more semantically plausible
     as a hard-negative probe for this scene.

This is the same signal cooccurrence.py tried to provide ("chair" is a
harder probe for a desk scene than "giraffe" is), but:
  - Training-free: CLIP is a fixed pretrained model already used elsewhere
    in this exact pipeline (for s_clip). No corpus statistics are fit,
    nothing is persisted across images.
  - Generalizes beyond COCO-80: works for any word in V, not just the 80
    categories with ground-truth annotations.
  - Domain-agnostic: if the pipeline is ever applied to non-COCO images,
    the semantic relatedness signal still works (CLIP's text encoder has
    broad conceptual coverage), unlike COCO-specific co-occurrence counts.

This module also provides the CLIP-based shortlisting function used to
keep the expensive OWL-ViT low-confidence check bounded when V is large
(~4500 words with RAM++ tags): rank vocabulary survivors by semantic
relevance, keep a fixed-size shortlist for OWL-ViT, so that enlarging V
doesn't proportionally increase inference cost.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary embedding cache (precomputed ONCE, reused forever)
# ---------------------------------------------------------------------------
def precompute_vocabulary_embeddings(
    clip_scorer,
    vocabulary: Sequence[str],
    cache_path: Optional[str] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed every word in `vocabulary` through CLIP's text encoder, once.

    Args:
        clip_scorer: a feature_extractors.ClipScorer instance (already
            loaded by the pipeline for s_clip — we reuse it, not load
            another copy).
        vocabulary: the full vocabulary V (COCO-80 + RAM++ tags).
        cache_path: if given AND the file already exists, skip computation
            and load from disk. If given and doesn't exist, compute and
            save. If None, always compute (no disk I/O).
        batch_size: how many words to embed per CLIP forward pass.

    Returns:
        (|V|, D) float32 numpy array of L2-normalized text embeddings,
        in the same order as `vocabulary`.
    """
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path)
        if data.shape[0] == len(vocabulary):
            return data
        # vocabulary changed since last cache — recompute
        print(f"[CLIP distractor] Cache {cache_path} has {data.shape[0]} rows "
              f"but vocabulary has {len(vocabulary)} words — recomputing")

    import torch

    all_embeddings = []
    for i in range(0, len(vocabulary), batch_size):
        batch = list(vocabulary[i:i + batch_size])
        emb = clip_scorer._text_embeddings(batch)  # (batch, D), L2-normalized
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"[CLIP distractor] Cached {embeddings.shape} embeddings → {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# Per-image semantic scoring (pure numpy, microseconds)
# ---------------------------------------------------------------------------
def score_vocabulary_by_semantic_relevance(
    candidate_words: Sequence[str],
    vocabulary: Sequence[str],
    vocab_embeddings: np.ndarray,
) -> np.ndarray:
    """For each vocabulary word, compute its maximum cosine similarity to
    any of this image's candidate words. Returns a (|V|,) array of scores.

    Candidate words that aren't in the vocabulary (e.g. non-COCO RAM++
    tags that were in O_init but aren't in V) are looked up by exact
    string match; if not found, they're silently skipped (the remaining
    candidates still provide signal). If NO candidate has a vocabulary
    match (unlikely but possible for very unusual images), all scores are
    0.0 and the caller falls back to uniform sampling — same as
    cooccurrence.py's fallback when no co-occurrence signal existed.
    """
    vocab_idx = {w: i for i, w in enumerate(vocabulary)}
    candidate_indices = [vocab_idx[w] for w in candidate_words if w in vocab_idx]

    if not candidate_indices:
        return np.zeros(len(vocabulary), dtype=float)

    cand_embs = vocab_embeddings[candidate_indices]  # (n_cand, D)
    # cosine similarity: both are already L2-normalized
    sim_matrix = vocab_embeddings @ cand_embs.T      # (|V|, n_cand)
    scores = sim_matrix.max(axis=1)                   # (|V|,)
    return scores.astype(float)


class ClipSemanticDistractorScorer:
    """Drop-in replacement for cooccurrence.CooccurrenceScorer in
    probe_sampling.sample_probe_pool's `distractor_scorer` argument.

    Binds precomputed vocabulary embeddings to a specific image's candidate
    pool, exposing the single-argument `score(word) -> float` callable
    that sample_probe_pool expects.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        vocab_embeddings: np.ndarray,
        candidate_words: Sequence[str],
    ):
        self.vocabulary = list(vocabulary)
        self._scores = score_vocabulary_by_semantic_relevance(
            candidate_words, self.vocabulary, vocab_embeddings
        )
        self._word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}

    def score(self, word: str) -> float:
        idx = self._word_to_idx.get(word)
        if idx is None:
            return 0.0
        return float(self._scores[idx])

    def __call__(self, word: str) -> float:
        return self.score(word)


# ---------------------------------------------------------------------------
# CLIP-based shortlisting (keeps OWL-ViT cost bounded when V is large)
# ---------------------------------------------------------------------------
def shortlist_by_semantic_relevance(
    survivors: Sequence[str],
    vocabulary: Sequence[str],
    vocab_embeddings: np.ndarray,
    candidate_words: Sequence[str],
    max_shortlist: int = 200,
    hard_fraction: float = 0.75,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Given vocabulary survivors after Filter 1 (exclusion), narrow them
    down to at most `max_shortlist` words before the expensive OWL-ViT
    Filter 2 runs, so that OWL-ViT cost stays ~constant regardless of
    vocabulary size.

    The shortlist is composed of:
      - top `hard_fraction * max_shortlist` words by semantic relevance
        to the current image's candidates (the "hard" probes)
      - the remaining slots filled by uniform random draws from the rest
        (diversity — ensures the null model isn't exclusively calibrated
        against the hardest possible probes, which could make it TOO
        conservative)

    If len(survivors) <= max_shortlist already, returns survivors unchanged
    (no shortlisting needed — this is the COCO-80-only case).
    """
    if len(survivors) <= max_shortlist:
        return list(survivors)

    rng = rng if rng is not None else np.random.default_rng()

    scores = score_vocabulary_by_semantic_relevance(
        candidate_words, vocabulary, vocab_embeddings
    )
    vocab_idx = {w: i for i, w in enumerate(vocabulary)}
    survivor_scores = [(w, scores[vocab_idx[w]] if w in vocab_idx else 0.0) for w in survivors]
    survivor_scores.sort(key=lambda x: -x[1])

    n_hard = min(int(max_shortlist * hard_fraction), len(survivor_scores))
    hard_picks = [w for w, _ in survivor_scores[:n_hard]]

    remaining = [w for w, _ in survivor_scores[n_hard:]]
    n_random = min(max_shortlist - n_hard, len(remaining))
    if n_random > 0:
        random_picks = list(rng.choice(np.array(remaining, dtype=object), size=n_random, replace=False))
    else:
        random_picks = []

    return hard_picks + [str(w) for w in random_picks]
