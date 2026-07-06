"""
chair_histogram_nc.py
======================

The sanity-check plot: probe signed-distance/score histograms vs candidate
scores, colored by whether each candidate is REAL or HALLUCINATED per
real CHAIR ground truth (eval/eval_chair.py's own COCO instance+caption
annotations -- the exact same ground truth CHAIR itself is scored
against), across a sample of images, for a handful of epsilon values.

Two panels, per the reasoning below:

  Panel A -- signed distance D(w) (Eq. 11), pooled across the sampled
  images: probes (the null/"absent" reference population) vs candidates,
  candidates split into real/hallucinated by ground truth. This is the
  most direct visualization of whether the null model actually separates
  real objects from hallucinated ones in evidence space -- if it's
  working, real candidates should sit well to the right of (above) the
  probe distribution, and hallucinated candidates should overlap it.

  Panel B -- conformal p-value (Eq. 12), candidates only (probes don't
  have a meaningful p-value against themselves), again split by ground
  truth, with a vertical dashed line at EACH epsilon in the sweep. This is
  the decision-relevant view: at a given epsilon, everything to the left
  of its line is verified (O_pos); the fraction of the "hallucinated"
  (red) mass to the left of a given epsilon line IS the empirical false-
  verification rate at that epsilon -- directly checking the paper's
  conformal guarantee (Section 3.6) against real data, which is the whole
  point of this sanity check.

D(w) can't carry a single universal epsilon threshold line the way
p-values can (epsilon is a RANK-based threshold against each image's OWN
probe population, not a fixed distance), so epsilon lines only appear on
Panel B; Panel A only ever depends on the sampled images, never epsilon.

This module needs the REAL COCO ground-truth annotations (eval_chair.py's
CHAIR class, at MARINE/data/coco/annotations) and REAL candidate/probe
data from candidate_pool.py + build_probe_pool.py + fit_null_calibration.py
-- none of that is available in a sandbox, so the data-EXTRACTION half
(build_ground_truth_labels, main()) must run on the server. The
data-PREPARATION and PLOTTING logic below it is pure and fully unit-
tested with a small synthetic ground-truth lookup (not real image data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from synonyms import basic_clean, load_coco_synonym_map  # noqa: E402


# ---------------------------------------------------------------------------
# Real ground-truth extraction (needs real COCO annotations -- server only)
# ---------------------------------------------------------------------------
def build_ground_truth_labels(
    images: Sequence[str], coco_annotations_path: str, chair_cache_path: Optional[str] = None
) -> Dict[str, Set[str]]:
    """Real ground truth per image, reusing eval/eval_chair.py's OWN CHAIR
    class verbatim (same synonym table, same combine_coco_instances /
    combine_coco_captions logic CHAIR itself is scored with) so "real vs
    hallucinated" here means EXACTLY what CHAIR means by it -- no second,
    possibly-inconsistent ground-truth definition.

    Returns {image_filename: {coco_node_word, ...}} for every image in
    `images` (looked up by the COCO numeric image id embedded in the
    filename, matching eval_chair.py::load_captions' own convention).
    """
    _EVAL_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "eval"))
    if _EVAL_DIR not in sys.path:
        sys.path.insert(0, _EVAL_DIR)
    from eval_chair import CHAIR  # noqa: E402
    import pickle

    if chair_cache_path and os.path.exists(chair_cache_path):
        with open(chair_cache_path, "rb") as f:
            evaluator = pickle.load(f)
    else:
        evaluator = CHAIR(coco_annotations_path)
        if chair_cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(chair_cache_path)) or ".", exist_ok=True)
            with open(chair_cache_path, "wb") as f:
                pickle.dump(evaluator, f)

    out: Dict[str, Set[str]] = {}
    for img in images:
        if "COCO" in img:
            imid = int(img.split("_")[-1].split(".")[0])
        else:
            imid = img
        out[img] = set(evaluator.imid_to_objects.get(imid, set()))
    return out


# ---------------------------------------------------------------------------
# Pure labeling + data-prep logic (fully unit-testable)
# ---------------------------------------------------------------------------
def label_candidate(
    canonical: str, is_coco_category: bool, gt_objects: Set[str], synonyms_map: Optional[Dict[str, str]] = None
) -> Optional[bool]:
    """Returns True (real), False (hallucinated), or None (not judgeable --
    CHAIR's ground truth only covers the 80 COCO categories, so a non-COCO
    candidate, e.g. a RAM++-only word with no COCO synonym-table mapping,
    simply cannot be scored against it one way or the other).
    """
    if not is_coco_category:
        return None
    synonyms_map = synonyms_map if synonyms_map is not None else load_coco_synonym_map()
    node_word = synonyms_map.get(basic_clean(canonical), basic_clean(canonical))
    return node_word in gt_objects


@dataclass
class HistogramData:
    """Everything plot_probe_vs_candidate_histograms needs, pre-extracted
    so the plotting function itself has zero I/O and is trivially testable."""

    probe_distances: List[float] = field(default_factory=list)
    real_candidate_distances: List[float] = field(default_factory=list)
    hallucinated_candidate_distances: List[float] = field(default_factory=list)
    real_candidate_p_values: List[float] = field(default_factory=list)
    hallucinated_candidate_p_values: List[float] = field(default_factory=list)
    n_images: int = 0
    n_judged: int = 0    # candidates with a COCO ground-truth verdict
    n_unjudged: int = 0  # non-COCO candidates, excluded from the plot


def collect_histogram_data(
    image_list: Sequence[str],
    candidate_pool_cache: Dict[str, dict],
    sort_results: Dict[str, dict],
    ground_truth: Dict[str, Set[str]],
    synonyms_map: Optional[Dict[str, str]] = None,
) -> HistogramData:
    """Pools probe distances and (real/hallucinated-labeled) candidate
    distances + p-values across `image_list`. Candidates whose
    `is_coco_category` is False are skipped (see label_candidate) --
    they're neither real nor hallucinated in CHAIR's own terms, so
    including them would silently misrepresent the ground-truth check.
    """
    synonyms_map = synonyms_map if synonyms_map is not None else load_coco_synonym_map()
    data = HistogramData()

    for img in image_list:
        pool = candidate_pool_cache.get(img)
        sort_result = sort_results.get(img)
        if pool is None or sort_result is None:
            continue
        data.n_images += 1

        data.probe_distances.extend(
            d for d in sort_result["probe_signed_distances"] if d != float("-inf")
        )

        gt_objects = ground_truth.get(img, set())
        cand_by_name = {c["canonical"]: c for c in pool["candidates"]}
        for name, p, d in zip(
            sort_result["candidate_names"],
            sort_result["candidate_p_values"],
            sort_result["candidate_signed_distances"],
        ):
            cand = cand_by_name.get(name)
            if cand is None:
                continue
            verdict = label_candidate(name, cand.get("is_coco_category", False), gt_objects, synonyms_map)
            if verdict is None:
                data.n_unjudged += 1
                continue
            data.n_judged += 1
            d_finite = d if d != float("-inf") else float("nan")
            if verdict:
                data.real_candidate_distances.append(d_finite)
                data.real_candidate_p_values.append(p)
            else:
                data.hallucinated_candidate_distances.append(d_finite)
                data.hallucinated_candidate_p_values.append(p)

    return data


# ---------------------------------------------------------------------------
# Plotting (pure, given HistogramData -- no I/O beyond writing the PNG)
# ---------------------------------------------------------------------------
def plot_probe_vs_candidate_histograms(
    data: HistogramData, epsilons: Sequence[float], output_path: str, title_suffix: str = ""
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epsilons = sorted(epsilons)
    fig, (ax_dist, ax_pval) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: signed distance D(w) ---
    finite_probe = [d for d in data.probe_distances if np.isfinite(d)]
    finite_real = [d for d in data.real_candidate_distances if np.isfinite(d)]
    finite_hall = [d for d in data.hallucinated_candidate_distances if np.isfinite(d)]

    all_finite = finite_probe + finite_real + finite_hall
    bins = np.linspace(min(all_finite, default=0.0), max(all_finite, default=1.0), 40) \
        if all_finite else np.linspace(0, 1, 40)

    if finite_probe:
        ax_dist.hist(finite_probe, bins=bins, density=True, alpha=0.4, color="gray",
                     label=f"probes (n={len(finite_probe)})")
    if finite_hall:
        ax_dist.hist(finite_hall, bins=bins, density=True, alpha=0.55, color="crimson",
                     label=f"hallucinated candidates (n={len(finite_hall)})")
    if finite_real:
        ax_dist.hist(finite_real, bins=bins, density=True, alpha=0.55, color="seagreen",
                     label=f"real candidates (n={len(finite_real)})")
    n_neg_inf_real = sum(1 for d in data.real_candidate_distances if d != d)  # NaN marker for -inf
    n_neg_inf_hall = sum(1 for d in data.hallucinated_candidate_distances if d != d)
    ax_dist.set_xlabel("Signed Mahalanobis distance D(w)  (Eq. 11)")
    ax_dist.set_ylabel("density")
    note = ""
    if n_neg_inf_real or n_neg_inf_hall:
        note = (f"\n(excludes D(w)=-inf: {n_neg_inf_real} real, {n_neg_inf_hall} "
                f"hallucinated candidates rejected by the one-sided test)")
    ax_dist.set_title(f"Probe vs. candidate evidence (D(w)){note}", fontsize=10)
    ax_dist.legend(fontsize=9)

    # --- Panel B: conformal p-values, with epsilon lines ---
    p_bins = np.linspace(0.0, 1.0, 41)
    if data.hallucinated_candidate_p_values:
        ax_pval.hist(data.hallucinated_candidate_p_values, bins=p_bins, alpha=0.55, color="crimson",
                     label=f"hallucinated candidates (n={len(data.hallucinated_candidate_p_values)})")
    if data.real_candidate_p_values:
        ax_pval.hist(data.real_candidate_p_values, bins=p_bins, alpha=0.55, color="seagreen",
                     label=f"real candidates (n={len(data.real_candidate_p_values)})")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(epsilons), 1)))
    for eps, color in zip(epsilons, colors):
        ax_pval.axvline(eps, color=color, linestyle="--", linewidth=2, label=f"epsilon={eps:g}")
    ax_pval.set_xlabel("conformal p-value (Eq. 12)")
    ax_pval.set_ylabel("count")
    ax_pval.set_title("Candidate p-values vs. epsilon (Eq. 13-14 decision boundary)", fontsize=10)
    ax_pval.legend(fontsize=9)

    fig.suptitle(
        f"Strategy 8-U-NC sanity check{title_suffix}\n"
        f"{data.n_images} images, {data.n_judged} COCO-judgeable candidates "
        f"({data.n_unjudged} non-COCO candidates excluded, not judgeable against CHAIR ground truth)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def empirical_false_verification_rates(data: HistogramData, epsilons: Sequence[float]) -> Dict[float, float]:
    """The direct, numeric check of Section 3.6's conformal guarantee
    against REAL data: at each epsilon, what fraction of HALLUCINATED
    (ground-truth-negative) candidates were nonetheless verified
    (p <= epsilon)? The guarantee (Pr[p(o_i) <= eps] <= eps for a truly
    absent object) predicts this should not greatly exceed epsilon."""
    out = {}
    n_hall = len(data.hallucinated_candidate_p_values)
    for eps in epsilons:
        if n_hall == 0:
            out[eps] = float("nan")
            continue
        n_false_verified = sum(1 for p in data.hallucinated_candidate_p_values if p <= eps)
        out[eps] = n_false_verified / n_hall
    return out


def main():
    parser = argparse.ArgumentParser(description="Strategy8-U-NC sanity check: probe vs candidate histograms")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--sort_results_file", type=str, required=True,
                        help="output of fit_null_calibration.py's --sort_results_output")
    parser.add_argument("--coco_annotations_path", type=str, required=True)
    parser.add_argument("--chair_cache_path", type=str, default="./data/coco/chair_cache.pkl")
    parser.add_argument("--image_list_file", type=str, required=True,
                        help="JSON list of ~50 image filenames to include in the sanity check")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--output_file", type=str, required=True)
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

    data = collect_histogram_data(image_list, candidate_pool_cache, sort_results, ground_truth)
    out_path = plot_probe_vs_candidate_histograms(data, args.epsilons, args.output_file)

    rates = empirical_false_verification_rates(data, args.epsilons)
    print(f"[Strategy8-U-NC][Sanity check] Wrote {out_path}")
    print(f"[Strategy8-U-NC][Sanity check] {data.n_images} images, {data.n_judged} judged candidates "
          f"({data.n_unjudged} non-COCO, excluded)")
    for eps, rate in rates.items():
        print(f"[Strategy8-U-NC][Sanity check] epsilon={eps:g}: empirical false-verification rate "
              f"among hallucinated candidates = {rate:.3f} (guarantee target: <= {eps:g})")


if __name__ == "__main__":
    main()
