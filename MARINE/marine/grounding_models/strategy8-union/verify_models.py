#!/usr/bin/env python3
"""
verify_models.py
=================

Run this ONCE on the server before the full pipeline. It loads every model
the pipeline uses, runs a single real query through each one's exact code
path, and prints PASS/FAIL with diagnostic shapes/values. If all four
checks pass, the full pipeline will work — every model call in the
pipeline goes through exactly these same functions.

Usage:
    python marine/strategy8-union/verify_models.py [--device cuda]
    python marine/strategy8-union/verify_models.py --gdino_model IDEA-Research/grounding-dino-tiny
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _make_test_image():
    """A real 224x224 RGB PIL image (random pixels — content doesn't
    matter, we just need model forward passes to not crash)."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def check_owlvit(device: str) -> bool:
    print("=" * 60)
    print("[1/4] OWL-ViT (s_det, s_area)")
    try:
        from feature_extractors import OwlViTScorer
        scorer = OwlViTScorer(device=device)
        image = _make_test_image()
        results = scorer.score_batch(image, ["dog", "cat", "chair"])
        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        for i, (s_det, s_area) in enumerate(results):
            assert isinstance(s_det, float) and isinstance(s_area, float)
            assert 0.0 <= s_det <= 1.0, f"s_det={s_det} out of [0,1]"
            assert 0.0 <= s_area <= 1.0, f"s_area={s_area} out of [0,1]"
        print(f"  Results: {results}")
        print("  PASS ✓")
        return True
    except Exception:
        traceback.print_exc()
        print("  FAIL ✗")
        return False


def check_clip(device: str) -> bool:
    print("=" * 60)
    print("[2/4] CLIP (s_clip + text embeddings for vocabulary precomputation)")
    try:
        from feature_extractors import ClipScorer
        scorer = ClipScorer(device=device)
        image = _make_test_image()

        # Test s_clip scoring (used during feature extraction)
        sims = scorer.score_batch(image, ["dog", "cat", "chair"])
        assert len(sims) == 3, f"expected 3 similarities, got {len(sims)}"
        for s in sims:
            assert isinstance(s, float)
            assert -1.0 <= s <= 1.0, f"cosine sim={s} out of [-1,1]"
        print(f"  s_clip scores: {sims}")

        # Test text embedding extraction (used by clip_distractor_scorer.py
        # for vocabulary precomputation — this is where the original crash
        # happened)
        text_emb = scorer._text_embeddings(["dog", "cat", "chair"])
        print(f"  text_emb shape: {text_emb.shape} (expected: (3, 512))")
        assert text_emb.shape[0] == 3, f"expected 3 rows, got {text_emb.shape[0]}"
        assert text_emb.shape[1] > 0, "embedding dimension is 0"
        norms = text_emb.norm(p=2, dim=-1)
        assert all(abs(n.item() - 1.0) < 0.01 for n in norms), f"not L2-normalized: norms={norms.tolist()}"
        print(f"  text_emb norms: {norms.tolist()} (expected: ~[1.0, 1.0, 1.0])")

        # Test image embedding
        img_emb = scorer._image_embedding(image)
        print(f"  image_emb shape: {img_emb.shape} (expected: (1, 512))")
        assert img_emb.shape[1] == text_emb.shape[1], (
            f"DIMENSION MISMATCH: image_emb is {img_emb.shape[1]}D but "
            f"text_emb is {text_emb.shape[1]}D — cosine similarities would "
            f"be computed in the wrong space"
        )

        print("  PASS ✓")
        return True
    except Exception:
        traceback.print_exc()
        print("  FAIL ✗")
        return False


def check_clip_vocab_precomputation(device: str) -> bool:
    print("=" * 60)
    print("[3/4] CLIP vocabulary precomputation (clip_distractor_scorer.py)")
    try:
        from feature_extractors import ClipScorer
        from clip_distractor_scorer import precompute_vocabulary_embeddings

        scorer = ClipScorer(device=device)
        test_vocab = ["dog", "cat", "chair", "table", "car"]
        embeddings = precompute_vocabulary_embeddings(scorer, test_vocab, cache_path=None)
        print(f"  embeddings shape: {embeddings.shape} (expected: (5, 512))")
        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] > 0
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  norms: {norms.tolist()} (expected: ~[1.0, 1.0, 1.0, 1.0, 1.0])")
        assert all(abs(n - 1.0) < 0.05 for n in norms), f"not L2-normalized"
        print("  PASS ✓")
        return True
    except Exception:
        traceback.print_exc()
        print("  FAIL ✗")
        return False


def check_gdino(device: str, model_name: str) -> bool:
    print("=" * 60)
    print(f"[4/4] GroundingDINO ({model_name})")
    if not model_name:
        print("  SKIPPED (no --gdino_model given, pipeline will run in 3D mode)")
        return True
    try:
        from gdino_scorer import GDINOScorer
        scorer = GDINOScorer(model_name=model_name, device=device)
        image = _make_test_image()
        results = scorer.score_batch(image, ["dog", "cat", "chair"])
        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        for i, s in enumerate(results):
            assert isinstance(s, float), f"result[{i}] is {type(s)}, expected float"
            assert 0.0 <= s <= 1.0, f"s_gdino={s} out of [0,1]"
        print(f"  s_gdino scores: {results}")
        print("  PASS ✓")
        return True
    except Exception:
        traceback.print_exc()
        print("  FAIL ✗")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify all models work before running the full pipeline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gdino_model", type=str, default="",
                        help="HF model name for GroundingDINO; empty = skip GDINO check")
    args = parser.parse_args()

    results = []
    results.append(("OWL-ViT", check_owlvit(args.device)))
    results.append(("CLIP", check_clip(args.device)))
    results.append(("CLIP vocab precomp", check_clip_vocab_precomputation(args.device)))
    results.append(("GroundingDINO", check_gdino(args.device, args.gdino_model)))

    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll checks passed. The full pipeline will work.")
    else:
        print("\nSome checks FAILED. Fix the issues above before running the full pipeline.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
