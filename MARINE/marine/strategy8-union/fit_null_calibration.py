"""
fit_null_calibration.py
========================

Strategy 8-U-NC's counterpart to fit_gmm.py + build_question_file.py: given
an existing candidate_pool_cache.jsonl (Step A, unchanged) and a
probe_pool_cache.jsonl (build_probe_pool.py, new), this module

  1. runs null_calibration.sort_one_image for EVERY image (this is pure
     numpy -- no GPU/model calls needed at all, since both caches already
     hold real, precomputed s_det/s_clip/s_area features), producing a
     ConformalSortResult per image, and
  2. classifies an image's candidates into O_pos/O_neg at any given
     epsilon (Eq. 13-14) and builds the tri-state question file exactly
     like build_question_file.py does for the GMM sorter -- so Phase II
     (prompts.py, tristate_logits.py) needs no changes at all.

Because step 1 needs no model calls, re-classifying at a NEW epsilon (e.g.
sweeping epsilon in {0.05, 0.1, 0.2} for the sanity-check report) is nearly
free: sort once, call .split(epsilon) as many times as needed.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

from null_calibration import ConformalSortResult, sort_one_image
from prompts import build_tristate_prompts


# ---------------------------------------------------------------------------
# Step 1: sort every image (pure numpy, no model calls)
# ---------------------------------------------------------------------------
def sort_all_images(
    candidate_pool_cache: Dict[str, dict],
    probe_pool_cache: Dict[str, dict],
    shrinkage: Optional[float] = None,
    image_filter: Optional[List[str]] = None,
) -> Dict[str, ConformalSortResult]:
    """Runs sort_one_image for every image present in BOTH caches (an
    image missing from probe_pool_cache -- e.g. build_probe_pool.py hasn't
    been run for it yet -- is silently skipped with a printed warning
    rather than crashing the whole batch)."""
    image_filter_set = set(image_filter) if image_filter is not None else None
    results: Dict[str, ConformalSortResult] = {}

    for img, cand_rec in candidate_pool_cache.items():
        if image_filter_set is not None and img not in image_filter_set:
            continue
        probe_rec = probe_pool_cache.get(img)
        if probe_rec is None:
            print(f"[Strategy8-U-NC][Fit] WARNING: no probe pool cached for {img}, skipping")
            continue

        candidate_features = {
            c["canonical"]: (c["s_det"], c["s_clip"], c["s_area"]) for c in cand_rec["candidates"]
        }
        probe_features = {p["word"]: (p["s_det"], p["s_clip"], p["s_area"]) for p in probe_rec["probes"]}

        try:
            results[img] = sort_one_image(candidate_features, probe_features, shrinkage=shrinkage)
        except ValueError as e:
            print(f"[Strategy8-U-NC][Fit] WARNING: {img} skipped ({e})")

    return results


def save_sort_results(results: Dict[str, ConformalSortResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump({img: r.to_dict() for img, r in results.items()}, f)


def load_sort_results(path: str) -> Dict[str, ConformalSortResult]:
    with open(path) as f:
        raw = json.load(f)
    return {img: ConformalSortResult.from_dict(d) for img, d in raw.items()}


# ---------------------------------------------------------------------------
# Step 2: classify at a given epsilon + build the question file
# (mirrors build_question_file.py::classify_image_candidates / build_question_file)
# ---------------------------------------------------------------------------
def classify_image_candidates_nc(
    sort_result: ConformalSortResult, epsilon: float
) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Eq. 13-14 applied to one image's already-computed conformal sort.
    Returns (o_pos, o_neg, {candidate_name: p_value})."""
    o_pos, o_neg = sort_result.split(epsilon)
    p_values = dict(zip(sort_result.candidate_names, sort_result.candidate_p_values))
    return o_pos, o_neg, p_values


def build_question_file_nc(
    question_path: str,
    sort_results: Dict[str, ConformalSortResult],
    epsilon: float,
    image_filter: Optional[List[str]] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """NC counterpart of build_question_file.py::build_question_file.
    Returns (strategy8_nc_questions, per_image_classification) in the
    IDENTICAL output schema build_question_file.py uses, so dataset.py,
    generate.py, and report.py all work unchanged."""
    try:
        with open(question_path) as f:
            questions = json.load(f)
    except json.JSONDecodeError:
        with open(question_path) as f:
            questions = [json.loads(line) for line in f]

    image_filter_set = set(image_filter) if image_filter is not None else None
    per_image_classification: Dict[str, dict] = {}
    out_questions: List[dict] = []

    for q in questions:
        img = q["image"]
        if image_filter_set is not None and img not in image_filter_set:
            continue

        if img not in per_image_classification:
            sort_result = sort_results.get(img)
            if sort_result is None:
                o_pos, o_neg, p_values = [], [], {}
            else:
                o_pos, o_neg, p_values = classify_image_candidates_nc(sort_result, epsilon)
            per_image_classification[img] = {
                "o_pos": o_pos,
                "o_neg": o_neg,
                "p_values": p_values,
                "epsilon": epsilon,
            }

        cls = per_image_classification[img]

        if "conversations" in q:
            query = q["conversations"][0]["value"]
            qid = q.get("id", q.get("question_id"))
        else:
            query = q["text"]
            qid = q.get("question_id", q.get("id"))

        c_ung, c_pos, c_neg = build_tristate_prompts(query, cls["o_pos"], cls["o_neg"])

        out_questions.append({
            "id": qid,
            "image": img,
            "conversations": [
                {"from": "human", "value": query},
                {"from": "gpt", "value": ""},
                {"from": "guidance_pos", "value": c_pos},
                {"from": "guidance_neg", "value": c_neg},
            ],
        })

    return out_questions, per_image_classification


def main():
    from candidate_pool import load_candidate_pool_cache
    from build_probe_pool import load_probe_pool_cache

    parser = argparse.ArgumentParser(description="Strategy8-U-NC: fit the null-calibrated conformal sorter")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--probe_pool_cache", type=str, required=True)
    parser.add_argument("--shrinkage", type=float, default=None,
                        help="fixed Ledoit-Wolf lambda; omit to use the analytic formula")
    parser.add_argument("--sort_results_output", type=str, required=True,
                        help="where to cache the per-image ConformalSortResult (reusable across epsilons)")
    parser.add_argument("--question_file", type=str, default=None,
                        help="if given, also build a strategy8-nc question file at --epsilon")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--question_output_file", type=str, default=None)
    parser.add_argument("--classification_output_file", type=str, default=None)
    args = parser.parse_args()

    cand_cache = load_candidate_pool_cache(args.candidate_pool_cache)
    probe_cache = load_probe_pool_cache(args.probe_pool_cache)

    results = sort_all_images(cand_cache, probe_cache, shrinkage=args.shrinkage)
    save_sort_results(results, args.sort_results_output)
    print(f"[Strategy8-U-NC][Fit] Sorted {len(results)} images, cached to {args.sort_results_output}")

    if args.question_file:
        out_questions, per_image = build_question_file_nc(
            args.question_file, results, epsilon=args.epsilon,
        )
        with open(args.question_output_file, "w") as f:
            json.dump(out_questions, f, indent=2)
        print(f"[Strategy8-U-NC][Fit] Wrote {len(out_questions)} questions "
              f"({len(per_image)} images) at epsilon={args.epsilon}")
        if args.classification_output_file:
            with open(args.classification_output_file, "w") as f:
                json.dump(per_image, f, indent=2)


if __name__ == "__main__":
    main()
