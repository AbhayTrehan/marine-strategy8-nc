"""
build_probe_pool.py
====================

Strategy 8-U-NC probe pool construction: for every image in an existing
candidate_pool_cache.jsonl, samples guaranteed-absent probes and extracts
their FULL feature vectors (s_det, s_clip, s_area, s_gdino — 4D).

Key changes from the original 3D version:
  - Loads the full RAM++ vocabulary (~4500 words), not just COCO-80
  - Uses CLIP semantic shortlisting (clip_distractor_scorer.py) to keep
    OWL-ViT cost bounded when vocabulary is large
  - Queries GroundingDINO for s_gdino (4th feature) for every probe
  - Replaces the corpus-fit co-occurrence table with the training-free
    CLIP semantic distractor scorer

Usage (on the server):
    python marine/strategy8-union/build_probe_pool.py \\
        --candidate_pool_cache ./output/llava2/strategy8_union_nc/candidate_pool_cache_4d.jsonl \\
        --image_folder ./data/coco/val2014 \\
        --ram_tag_list_path $(python -c "import ram,os;print(os.path.join(os.path.dirname(ram.__file__),'data','ram_tag_list.txt'))") \\
        --K 80 --tau_low 0.3 \\
        --output_file ./output/llava2/strategy8_union_nc/probe_pool_cache_4d.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from clip_distractor_scorer import (  # noqa: E402
    ClipSemanticDistractorScorer,
    precompute_vocabulary_embeddings,
    shortlist_by_semantic_relevance,
)
from probe_sampling import load_default_vocabulary, sample_probe_pool  # noqa: E402
from synonyms import load_coco_synonym_map  # noqa: E402


def build_probe_pool_cache(
    candidate_pool_cache: Dict[str, dict],
    image_dir: str,
    feature_extractor,
    gdino_scorer,
    vocabulary: List[str],
    vocab_embeddings,
    K: int,
    tau_low: float,
    output_path: str,
    seed: int = 242,
    min_K: Optional[int] = None,
    max_owlvit_candidates: int = 200,
    image_filter: Optional[List[str]] = None,
) -> None:
    import numpy as np

    synonyms_map = load_coco_synonym_map()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if image_filter is not None:
        image_filter_set = set(image_filter)
        items = [(img, rec) for img, rec in candidate_pool_cache.items() if img in image_filter_set]
        missing = image_filter_set - set(candidate_pool_cache.keys())
        if missing:
            print(f"[Strategy8-U-NC][Probe pool] WARNING: {len(missing)} of "
                  f"{len(image_filter_set)} requested images are NOT in the "
                  f"candidate cache and will be skipped entirely (never "
                  f"reach probe building, fitting, OR the histogram): "
                  f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        items = list(candidate_pool_cache.items())

    n_done = 0
    n_skipped = 0
    n_degraded = 0
    n_total = len(items)
    with open(output_path, "w") as out_f:
        for img_file, rec in items:
            candidate_words = [c["canonical"] for c in rec["candidates"]]
            image_path = os.path.join(image_dir, img_file)
            image = Image.open(image_path).convert("RGB")

            def low_conf_score_fn(words, _image=image):
                det_area = feature_extractor.owlvit.score_batch(_image, words)
                return [s_det for s_det, _s_area in det_area]

            # CLIP-based distractor scorer (training-free, replaces co-occurrence table)
            distractor_scorer = ClipSemanticDistractorScorer(
                vocabulary, vocab_embeddings, candidate_words
            )

            # CLIP-based shortlisting (keeps OWL-ViT cost bounded)
            def shortlist_fn(survivors, _cands=candidate_words, _rng_seed=(seed, img_file)):
                _rng = np.random.default_rng(hash(_rng_seed) & 0xFFFFFFFF)
                return shortlist_by_semantic_relevance(
                    survivors, vocabulary, vocab_embeddings, _cands,
                    max_shortlist=max_owlvit_candidates, rng=_rng,
                )

            rng = np.random.default_rng(hash((seed, img_file)) & 0xFFFFFFFF)
            try:
                probe_words = sample_probe_pool(
                    vocabulary=vocabulary,
                    candidate_words=candidate_words,
                    K=K,
                    low_conf_score_fn=low_conf_score_fn,
                    distractor_scorer=distractor_scorer,
                    tau_low=tau_low,
                    synonyms_map=synonyms_map,
                    rng=rng,
                    min_K=min_K,
                    shortlist_fn=shortlist_fn,
                )
            except ValueError as e:
                print(f"[Strategy8-U-NC][Probe pool] WARNING: {img_file} skipped ({e})")
                n_skipped += 1
                n_done += 1
                continue

            if len(probe_words) < K:
                print(f"[Strategy8-U-NC][Probe pool] NOTE: {img_file} got "
                      f"{len(probe_words)}/{K} probes")
                n_degraded += 1

            # Extract 3D features (s_det, s_clip, s_area) via existing pipeline
            feats_3d = feature_extractor.extract(image, probe_words)

            # Extract s_gdino (4th feature) via GroundingDINO -- omitted
            # entirely (not a fake 0.0) when no GDINO scorer is configured,
            # so fit_null_calibration.py's 3D-vs-4D auto-detection (which
            # checks key PRESENCE, not value) correctly falls back to 3D
            # rather than treating a constant placeholder as real signal.
            if gdino_scorer is not None:
                gdino_scores = gdino_scorer.score_batch(image, probe_words)
            else:
                gdino_scores = None

            probes_out = []
            for i, w in enumerate(probe_words):
                s_det, s_clip, s_area = feats_3d.get(w, (0.0, 0.0, 0.0))
                probe_record = {
                    "word": w,
                    "s_det": s_det,
                    "s_clip": s_clip,
                    "s_area": s_area,
                }
                if gdino_scores is not None:
                    probe_record["s_gdino"] = gdino_scores[i]
                probes_out.append(probe_record)

            out_f.write(json.dumps({
                "image": img_file,
                "K": len(probes_out),
                "K_requested": K,
                "tau_low": tau_low,
                "probes": probes_out,
            }) + "\n")
            out_f.flush()

            n_done += 1
            if n_done % 25 == 0 or n_done == n_total:
                print(f"[Strategy8-U-NC][Probe pool] {n_done}/{n_total} images processed")

    print(f"[Strategy8-U-NC][Probe pool] Done. {n_total - n_skipped}/{n_total} written "
          f"({n_skipped} skipped, {n_degraded} degraded). → {output_path}")


def load_probe_pool_cache(path: str) -> Dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image"]] = rec
    return out


def main():
    parser = argparse.ArgumentParser(description="Strategy8-U-NC: 4D probe pool construction")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--ram_tag_list_path", type=str, default=None)
    parser.add_argument("--vocab_embeddings_cache", type=str, default=None,
                        help="path to .npy cache of CLIP text embeddings for the vocabulary; "
                             "computed once and reused on subsequent runs")
    parser.add_argument("--owlvit_model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--gdino_model", type=str, default=None,
                        help="HF model name/path (e.g. IDEA-Research/grounding-dino-tiny), "
                             "via transformers' AutoModelForZeroShotObjectDetection; "
                             "omit to skip s_gdino (3D mode)")
    parser.add_argument("--K", type=int, default=80)
    parser.add_argument("--min_K", type=int, default=30)
    parser.add_argument("--tau_low", type=float, default=0.3)
    parser.add_argument("--max_owlvit_candidates", type=int, default=200,
                        help="max words sent to OWL-ViT per image (CLIP shortlisting reduces V to this)")
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_list_file", type=str, default=None,
                        help="JSON list of image filenames to process; if omitted, all images "
                             "in the candidate cache are processed")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    from candidate_pool import load_candidate_pool_cache
    from feature_extractors import FeatureExtractor

    cache = load_candidate_pool_cache(args.candidate_pool_cache)

    vocabulary = load_default_vocabulary(ram_tag_list_path=args.ram_tag_list_path)
    print(f"[Strategy8-U-NC][Probe pool] Vocabulary V: {len(vocabulary)} words")

    feature_extractor = FeatureExtractor(args.owlvit_model, args.clip_model, device=args.device)

    # Precompute CLIP text embeddings for the entire vocabulary (once)
    embed_cache = args.vocab_embeddings_cache
    if embed_cache is None:
        embed_cache = os.path.join(
            os.path.dirname(os.path.abspath(args.output_file)),
            "vocab_clip_embeddings.npy"
        )
    vocab_embeddings = precompute_vocabulary_embeddings(
        feature_extractor.clip, vocabulary, cache_path=embed_cache
    )

    # GroundingDINO (optional — graceful fallback to 3D if not configured)
    gdino_scorer = None
    if args.gdino_model:
        from gdino_scorer import GDINOScorer
        gdino_scorer = GDINOScorer(model_name=args.gdino_model, device=args.device)
        print(f"[Strategy8-U-NC][Probe pool] GroundingDINO ({args.gdino_model}) loaded → 4D features")
    else:
        print("[Strategy8-U-NC][Probe pool] No --gdino_model given → 3D features (s_gdino=0)")

    image_filter = None
    if args.image_list_file:
        with open(args.image_list_file) as f:
            image_filter = json.load(f)
        print(f"[Strategy8-U-NC][Probe pool] Filtering to {len(image_filter)} images from {args.image_list_file}")

    build_probe_pool_cache(
        candidate_pool_cache=cache,
        image_dir=args.image_folder,
        feature_extractor=feature_extractor,
        gdino_scorer=gdino_scorer,
        vocabulary=vocabulary,
        vocab_embeddings=vocab_embeddings,
        K=args.K,
        tau_low=args.tau_low,
        output_path=args.output_file,
        seed=args.seed,
        min_K=(args.min_K if args.min_K > 0 else None),
        max_owlvit_candidates=args.max_owlvit_candidates,
        image_filter=image_filter,
    )


if __name__ == "__main__":
    main()
