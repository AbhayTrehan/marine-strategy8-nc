"""
diagnose_false_verifications.py
=================================

Diagnostic: for the hallucinated (ground-truth-negative) candidates that
got wrongly verified (p <= epsilon) in a sanity-check run, print their
full raw feature vector alongside summary statistics of that image's
probe population, so we can see WHICH dimension(s) are driving the false
verification -- in particular, whether GDINO (s_gdino) is elevated too
(meaning it's not adding independent signal on these cases) or whether
it's specifically s_det/s_clip carrying the false signal while s_gdino
correctly stays low (meaning GDINO IS helping, just not enough to flip
the verdict given how the other 3 dimensions combine).

Usage:
    python marine/strategy8-union/diagnose_false_verifications.py \\
        --candidate_pool_cache ./output/llava2/strategy8_union_nc/candidate_pool_cache_4d.jsonl \\
        --sort_results_file ./output/llava2/strategy8_union_nc/sort_results.json \\
        --coco_annotations_path ./data/coco/annotations \\
        --image_list_file ./output/llava2/strategy8_union_nc/sanity_check_images.json \\
        --epsilon 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from chair_histogram_nc import build_ground_truth_labels, label_candidate
from synonyms import load_coco_synonym_map


def main():
    parser = argparse.ArgumentParser(description="Inspect wrongly-verified hallucinated candidates")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--sort_results_file", type=str, required=True)
    parser.add_argument("--coco_annotations_path", type=str, required=True)
    parser.add_argument("--chair_cache_path", type=str, default="./data/coco/chair_cache.pkl")
    parser.add_argument("--image_list_file", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()

    from candidate_pool import load_candidate_pool_cache

    candidate_pool_cache = load_candidate_pool_cache(args.candidate_pool_cache)
    with open(args.sort_results_file) as f:
        sort_results = json.load(f)
    with open(args.image_list_file) as f:
        image_list = json.load(f)

    ground_truth = build_ground_truth_labels(
        image_list, args.coco_annotations_path, chair_cache_path=args.chair_cache_path
    )
    synonyms_map = load_coco_synonym_map()

    n_shown = 0
    for img in image_list:
        pool = candidate_pool_cache.get(img)
        sort_result = sort_results.get(img)
        if pool is None or sort_result is None:
            continue

        gt_objects = ground_truth.get(img, set())
        cand_by_name = {c["canonical"]: c for c in pool["candidates"]}

        # Probe summary stats for this image (for comparison)
        probe_feats = [
            (p.get("s_det", 0), p.get("s_clip", 0), p.get("s_area", 0), p.get("s_gdino"))
            for p in pool.get("probes", [])
        ] if "probes" in pool else None

        for name, p, d in zip(
            sort_result["candidate_names"],
            sort_result["candidate_p_values"],
            sort_result["candidate_signed_distances"],
        ):
            cand = cand_by_name.get(name)
            if cand is None:
                continue
            verdict = label_candidate(name, cand.get("is_coco_category", False), gt_objects, synonyms_map)
            if verdict is None or verdict is True:
                continue  # only interested in TRUE hallucinated (verdict=False)
            if p > args.epsilon:
                continue  # only interested in WRONGLY VERIFIED (p <= epsilon)

            n_shown += 1
            print(f"\n=== {img} :: '{name}' (HALLUCINATED, wrongly verified) ===")
            print(f"  p-value: {p:.4f}   signed distance D(w): {d:.3f}")
            print(f"  raw features: s_det={cand.get('s_det'):.4f}  s_clip={cand.get('s_clip'):.4f}  "
                  f"s_area={cand.get('s_area'):.4f}"
                  + (f"  s_gdino={cand.get('s_gdino'):.4f}" if "s_gdino" in cand else "  (no s_gdino)"))
            print(f"  ground truth objects present: {sorted(gt_objects)}")
            print(f"  sources: {cand.get('sources')}, raw mentions: {cand.get('raw_mentions')}")

    print(f"\n\nTotal wrongly-verified hallucinated candidates shown: {n_shown}")


if __name__ == "__main__":
    main()
