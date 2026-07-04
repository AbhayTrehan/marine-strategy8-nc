"""
build_probe_pool.py
====================

Strategy 8-U-NC's counterpart to candidate_pool.py: for every image already
present in an existing candidate_pool_cache.jsonl (Strategy 8-U's Step A --
UNCHANGED, reused as-is), this script samples that image's guaranteed-absent
probe pool P (probe_sampling.py, Section 3.1) and extracts each probe's real
3D feature vector using the EXACT SAME feature pipeline as the candidates
(feature_extractors.py::FeatureExtractor -- Section 3.2's "using exactly the
same feature pipeline regardless of whether w is a candidate or a probe").

This is a REAL script: it needs the real image files (MARINE/data/coco/val2014)
and the real OWL-ViT/CLIP models (GPU), exactly like candidate_pool.py, and
cannot be executed in a sandbox without that access -- run it on the same
server/environment candidate_pool.py already ran on.

Output schema (one JSON object per line, mirroring candidate_pool.py):
{
  "image": "COCO_val2014_000000144305.jpg",
  "K": 80,
  "tau_low": 0.3,
  "probes": [
    {"word": "chair", "s_det": 0.05, "s_clip": 0.18, "s_area": 0.0},
    ...
  ]
}

Usage (on the server):
    python marine/strategy8-union/build_probe_pool.py \\
        --candidate_pool_cache ./output/llava2/strategy8_union/candidate_pool_cache.jsonl \\
        --image_folder ./data/coco/val2014 \\
        --ram_tag_list_path <path to ram package's ram_tag_list.txt> \\
        --cooccurrence_table ./data/coco/cooccurrence_table.json \\
        --coco_annotations_path ./data/coco/annotations \\
        --K 80 --tau_low 0.3 \\
        --output_file ./output/llava2/strategy8_union_nc/probe_pool_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cooccurrence import CooccurrenceScorer, load_cooccurrence_table  # noqa: E402
from probe_sampling import load_default_vocabulary, sample_probe_pool  # noqa: E402
from synonyms import load_coco_synonym_map  # noqa: E402


def build_probe_pool_cache(
    candidate_pool_cache: Dict[str, dict],
    image_dir: str,
    feature_extractor,
    vocabulary: List[str],
    K: int,
    tau_low: float,
    output_path: str,
    cooccurrence_table: Optional[Dict[str, Dict[str, int]]] = None,
    seed: int = 242,
    min_K: Optional[int] = None,
) -> None:
    import numpy as np

    synonyms_map = load_coco_synonym_map()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    n_done = 0
    n_skipped = 0
    n_degraded = 0
    n_total = len(candidate_pool_cache)
    with open(output_path, "w") as out_f:
        for img_file, rec in candidate_pool_cache.items():
            candidate_words = [c["canonical"] for c in rec["candidates"]]
            image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")

            # Filter 2 needs a REAL, image-bound s_det scorer: one OWL-ViT
            # forward pass per candidate probe-word batch, against THIS image.
            def low_conf_score_fn(words, _image=image):
                det_area = feature_extractor.owlvit.score_batch(_image, words)
                return [s_det for s_det, _s_area in det_area]

            distractor_scorer = None
            if cooccurrence_table is not None:
                distractor_scorer = CooccurrenceScorer(cooccurrence_table, candidate_words)

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
                )
            except ValueError as e:
                # A single image with a too-small vocabulary survivor set
                # (e.g. COCO-80-only vocabulary combined with an
                # unusually broad candidate pool) must not kill the whole
                # 50-image batch -- skip it, log why, keep going. This
                # mirrors fit_null_calibration.py's own per-image
                # warn-and-skip pattern for the same reason.
                print(f"[Strategy8-U-NC][Probe pool] WARNING: {img_file} skipped ({e})")
                n_skipped += 1
                n_done += 1
                continue

            if len(probe_words) < K:
                print(f"[Strategy8-U-NC][Probe pool] NOTE: {img_file} got only "
                      f"{len(probe_words)}/{K} probes (min_K={min_K} allowed degrading)")
                n_degraded += 1

            feats = feature_extractor.extract(image, probe_words)
            probes_out = []
            for w in probe_words:
                s_det, s_clip, s_area = feats.get(w, (0.0, 0.0, 0.0))
                probes_out.append({"word": w, "s_det": s_det, "s_clip": s_clip, "s_area": s_area})

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

    print(f"[Strategy8-U-NC][Probe pool] Done. {n_total - n_skipped}/{n_total} images written "
          f"({n_skipped} skipped, {n_degraded} degraded below K={K}). Cache written to {output_path}")


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
    parser = argparse.ArgumentParser(description="Strategy8-U-NC: probe pool sampling + feature extraction")
    parser.add_argument("--candidate_pool_cache", type=str, required=True,
                        help="existing candidate_pool_cache.jsonl from candidate_pool.py (Step A, unchanged)")
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--ram_tag_list_path", type=str, default=None,
                        help="path to the `ram` package's bundled ram_tag_list.txt; "
                             "if omitted, vocabulary V falls back to COCO-80 only")
    parser.add_argument("--cooccurrence_table", type=str, default=None,
                        help="JSON produced by cooccurrence.build_cooccurrence_table "
                             "(needs data/coco/annotations); omit to disable distractor bias")
    parser.add_argument("--owlvit_model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--K", type=int, default=80, help="probe pool size (paper: 50-100)")
    parser.add_argument("--min_K", type=int, default=30,
                        help="if fewer than K vocabulary words survive filters 1-2 for a given "
                             "image, use whatever survived down to this floor instead of "
                             "failing that image outright (must stay >= 4 for null_calibration's "
                             "own covariance-stability floor; default 30 keeps the null model "
                             "reasonably well-conditioned even when degraded). Pass "
                             "--min_K 0 to disable and hard-fail on ANY image with fewer than K "
                             "survivors instead.")
    parser.add_argument("--tau_low", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    from candidate_pool import load_candidate_pool_cache
    from feature_extractors import FeatureExtractor

    cache = load_candidate_pool_cache(args.candidate_pool_cache)

    vocabulary = load_default_vocabulary(ram_tag_list_path=args.ram_tag_list_path)
    print(f"[Strategy8-U-NC][Probe pool] Vocabulary V size: {len(vocabulary)} words"
          + (" (COCO-80 only -- pass --ram_tag_list_path for the full paper-spec vocabulary)"
             if args.ram_tag_list_path is None else ""))

    cooccurrence_table = None
    if args.cooccurrence_table:
        cooccurrence_table = load_cooccurrence_table(args.cooccurrence_table)

    feature_extractor = FeatureExtractor(args.owlvit_model, args.clip_model, device=args.device)

    build_probe_pool_cache(
        candidate_pool_cache=cache,
        image_dir=args.image_folder,
        feature_extractor=feature_extractor,
        vocabulary=vocabulary,
        K=args.K,
        tau_low=args.tau_low,
        output_path=args.output_file,
        cooccurrence_table=cooccurrence_table,
        seed=args.seed,
        min_K=(args.min_K if args.min_K > 0 else None),
    )


if __name__ == "__main__":
    main()
