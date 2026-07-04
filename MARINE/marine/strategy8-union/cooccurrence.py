"""
cooccurrence.py
================

Supports Strategy 8-U-NC's Section 3.1, filter 3 ("distractor bias"):
"Preferentially sample words that co-occur frequently with the objects
already in O_init (in the sense used to construct POPE's adversarial
split), since these are precisely the language-prior traps a null model
must be calibrated against."

POPE's adversarial split (Li et al., 2023b) ranks, for each COCO category,
the OTHER COCO categories most frequently appearing in the SAME image
(mined from the real ground-truth instance segmentation annotations), and
samples its "hard negative" probe objects preferentially from that
co-occurrence-ranked list. This module:

  1. `build_cooccurrence_table` -- computes the REAL co-occurrence counts
     from COCO's instance annotations (the exact same
     instances_val2014.json / instances_train2014.json files
     eval_chair.py already needs, at MARINE/data/coco/annotations). This
     needs the real annotation files and is meant to be run once, on the
     server, its output cached to JSON.
  2. `CooccurrenceScorer` -- a thin, pure wrapper around an already-built
     table, exposing the `distractor_scorer(word) -> float` callable that
     probe_sampling.sample_probe_pool expects. This half has no I/O and is
     fully unit-testable with a small synthetic table.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from synonyms import basic_clean  # noqa: E402


# ---------------------------------------------------------------------------
# Real table construction (needs real COCO instance annotations; run on
# the server, same requirement as eval/eval_chair.py's combine_coco_instances)
# ---------------------------------------------------------------------------
def build_cooccurrence_table(instance_annotation_paths: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Builds {category_a: {category_b: co-occurrence count}} from one or
    more COCO `instances_*.json` files (e.g.
    [".../instances_val2014.json", ".../instances_train2014.json"], mirroring
    eval_chair.py's combine_coco_instances which pools both splits).

    Co-occurrence here means: category_a and category_b both have at least
    one annotated instance in the SAME image. This is symmetric
    (table[a][b] == table[b][a]) and excludes self-pairs (a category never
    "co-occurs" with itself in this table).
    """
    image_to_categories: Dict[int, set] = defaultdict(set)
    cat_id_to_name: Dict[int, str] = {}

    for path in instance_annotation_paths:
        with open(path) as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            cat_id_to_name[cat["id"]] = cat["name"]
        for ann in data.get("annotations", []):
            name = cat_id_to_name.get(ann["category_id"])
            if name is not None:
                image_to_categories[ann["image_id"]].add(name)

    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for categories in image_to_categories.values():
        cats = sorted(categories)
        for i in range(len(cats)):
            for j in range(len(cats)):
                if i == j:
                    continue
                table[cats[i]][cats[j]] += 1

    return {k: dict(v) for k, v in table.items()}


def save_cooccurrence_table(table: Dict[str, Dict[str, int]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, indent=2)


def load_cooccurrence_table(path: str) -> Dict[str, Dict[str, int]]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pure scoring logic (fully unit-testable, no I/O)
# ---------------------------------------------------------------------------
def score_from_table(
    word: str,
    present_objects: Iterable[str],
    table: Dict[str, Dict[str, int]],
    aggregation: str = "max",
) -> float:
    """Section 3.1, filter 3's per-probe-candidate distractor score: how
    strongly `word` co-occurs with the objects already present in this
    image's candidate pool (`present_objects`).

    Returns 0.0 if `word` has no entry in the table at all, or if none of
    `present_objects` co-occur with it (this is the correct behavior for
    non-COCO vocabulary words, e.g. RAM++-only tags, since the table is
    necessarily built only over the 80 COCO categories that have ground-
    truth instance annotations -- those words simply fall through to the
    uniform-fill portion of sample_probe_pool, exactly as the paper's
    "the remainder of P is filled by uniform sampling" describes).

    aggregation: 'max' (default, the single strongest co-occurring present
    object drives the score) or 'sum' (cumulative signal across all
    present objects this word co-occurs with).
    """
    cw = basic_clean(word)
    row = None
    for key, r in table.items():
        if basic_clean(key) == cw:
            row = r
            break
    if row is None:
        return 0.0

    scores = []
    for obj in present_objects:
        co = basic_clean(obj)
        for key, count in row.items():
            if basic_clean(key) == co:
                scores.append(float(count))
                break
    if not scores:
        return 0.0
    if aggregation == "sum":
        return float(sum(scores))
    return float(max(scores))


class CooccurrenceScorer:
    """Binds a co-occurrence table to a specific image's candidate pool,
    exposing the single-argument `score(word) -> float` callable
    probe_sampling.sample_probe_pool's `distractor_scorer` expects."""

    def __init__(self, table: Dict[str, Dict[str, int]], present_objects: Sequence[str],
                 aggregation: str = "max"):
        self.table = table
        self.present_objects = list(present_objects)
        self.aggregation = aggregation

    def score(self, word: str) -> float:
        return score_from_table(word, self.present_objects, self.table, aggregation=self.aggregation)

    def __call__(self, word: str) -> float:
        return self.score(word)
