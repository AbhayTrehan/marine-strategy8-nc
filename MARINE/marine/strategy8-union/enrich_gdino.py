"""
enrich_gdino.py
================

Adds the 4th feature dimension (s_gdino) to an existing
candidate_pool_cache.jsonl WITHOUT modifying candidate_pool.py or any
other Strategy 8-U shared code. Reads the existing 3-feature cache,
queries GroundingDINO for every candidate word in every image, and writes
a new enriched cache with s_gdino appended to each candidate record.

Usage (on the server):
    python marine/strategy8-union/enrich_gdino.py \\
        --candidate_pool_cache ./output/llava2/strategy8_union/candidate_pool_cache.jsonl \\
        --image_folder ./data/coco/val2014 \\
        --gdino_model IDEA-Research/grounding-dino-tiny \\
        --output_file ./output/llava2/strategy8_union_nc/candidate_pool_cache_4d.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def enrich_candidate_cache_with_gdino(
    candidate_pool_cache: Dict[str, dict],
    image_dir: str,
    gdino_scorer,
    output_path: str,
) -> None:
    """Reads the existing cache, adds s_gdino to each candidate, writes
    an enriched version."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    n_done = 0
    n_total = len(candidate_pool_cache)
    with open(output_path, "w") as out_f:
        for img_file, rec in candidate_pool_cache.items():
            candidates = rec["candidates"]
            words = [c["canonical"] for c in candidates]

            image_path = os.path.join(image_dir, img_file)
            if words:
                image = Image.open(image_path).convert("RGB")
                gdino_scores = gdino_scorer.score_batch(image, words)
            else:
                gdino_scores = []

            enriched_candidates = []
            for c, s_gdino in zip(candidates, gdino_scores):
                enriched = dict(c)
                enriched["s_gdino"] = float(s_gdino)
                enriched_candidates.append(enriched)

            enriched_rec = dict(rec)
            enriched_rec["candidates"] = enriched_candidates
            out_f.write(json.dumps(enriched_rec) + "\n")
            out_f.flush()

            n_done += 1
            if n_done % 25 == 0 or n_done == n_total:
                print(f"[Strategy8-U-NC][GDINO enrich] {n_done}/{n_total} images processed")

    print(f"[Strategy8-U-NC][GDINO enrich] Done. Enriched cache → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Add GroundingDINO scores to candidate_pool_cache")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-tiny",
                        help="HF model name/path (via transformers' AutoModelForZeroShotObjectDetection)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    from candidate_pool import load_candidate_pool_cache
    from gdino_scorer import GDINOScorer

    cache = load_candidate_pool_cache(args.candidate_pool_cache)
    scorer = GDINOScorer(model_name=args.gdino_model, device=args.device)

    enrich_candidate_cache_with_gdino(cache, args.image_folder, scorer, args.output_file)


if __name__ == "__main__":
    main()
