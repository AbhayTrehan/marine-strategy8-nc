# Strategy 8-U-NC: Null-Calibrated Conformal Sorter -- sanity-check pipeline

Implementation of `strategy8_nc.pdf` on top of the existing Strategy 8-U
codebase in this directory. **Phase II (prompts.py, tristate_logits.py) is
reused completely unchanged** -- the spec is explicit that only Phase I
(the offline sorter) changes; everything downstream of `O_pos`/`O_neg`
(prompt construction, tri-state decoding) is identical. In fact
`tristate_logits.py` already implements the corrected additive decoding
equation (`z_final = z_ung + alpha*(z_pos - z_neg)`) that `strategy8_nc.pdf`
Section 4.3 specifies, so no changes were needed there at all.

## What's new here

| File | Role | Needs real images/GPU to produce real numbers? |
|---|---|---|
| `probe_sampling.py` | Section 3.1: guaranteed-absent probe pool sampling (candidate exclusion via COCO-synonym-group + WordNet synonym/hypernym/hyponym expansion; low-confidence exclusion; distractor-biased + uniform-fill sampling). | No (pure logic, pluggable score functions) -- but the score functions it's called with, in production, do. |
| `cooccurrence.py` | Builds the real POPE-style object co-occurrence table from COCO instance annotations (distractor bias), plus a pure scorer. | Table-building: **yes** (needs `data/coco/annotations`). Scoring given a table: no. |
| `null_calibration.py` | Sections 3.2-3.7: the statistical core -- probe-only normalization, one-class Gaussian null with Ledoit-Wolf shrinkage, signed Mahalanobis distance (one-sided projection test), conformal p-values, hard split at epsilon. | No -- pure numpy over already-extracted `(s_det, s_clip, s_area)` triples. |
| `build_probe_pool.py` | Real script: for every image in an existing `candidate_pool_cache.jsonl`, samples probes and extracts their real features via the SAME `feature_extractors.FeatureExtractor` Strategy 8-U already uses. | **Yes** -- real images + real OWL-ViT/CLIP (GPU), exactly like `candidate_pool.py`. |
| `fit_null_calibration.py` | Runs `null_calibration.sort_one_image` for every image (pure numpy, no model calls needed once both caches exist), and builds the tri-state question file at a given epsilon -- identical output schema to `build_question_file.py`, so `dataset.py`/`generate.py`/`report.py` all work unchanged. | No, once `build_probe_pool.py` has run. |
| `chair_histogram_nc.py` | The sanity-check plot: probe vs. candidate score histograms (signed distance + conformal p-value panels), colored real/hallucinated per **real CHAIR ground truth** (reuses `eval/eval_chair.py::CHAIR` verbatim), across a sample of images, for a sweep of epsilon values. Also reports the empirical false-verification rate per epsilon as a direct numeric check of Section 3.6's conformal guarantee against real data. | Ground-truth extraction: **yes** (`data/coco/annotations`). Data-prep/plotting: no. |
| `report_nc.py` | The visual HTML report: per image, candidate pool + probe pool summary + conformal p-values/signed distances + O_pos/O_neg at every epsilon in the sweep + guidance prompts + final response. Deliberately has **no ground-truth annotations** (it's a sanity check of the method's own behavior, not an accuracy eval -- see `chair_histogram_nc.py` for the accuracy-against-ground-truth view). | Needs real image files to embed thumbnails; the data tables work from cached JSON either way. |

All six have full unit test coverage in `tests/` using synthetic data
(clearly documented as such in each test file's docstring) -- these verify
the math and wiring are correct, but are **not** a substitute for running
the real pipeline. In particular, `tests/test_null_calibration.py`
includes a Monte Carlo check that the conformal validity guarantee
(Section 3.6) actually holds for this implementation on synthetic Gaussian
data, which is the core theoretical property the whole sanity check exists
to validate.

## Two related bugs fixed in the shared candidate-pool code (Strategy 8-U side, also used by NC)

Found and fixed via an audit of the real `candidate_pool_cache.jsonl` (500
real COCO images, real LLaVA/RAM++/DETR/OWL-ViT/CLIP output):

* `singularize_utils.py` -- replaced TextBlob's blind suffix-stripping
  singularizer (which mangled "bus"->"bu", "tennis"->"tenni", etc.) with a
  WordNet-`morphy`-backed one that never invents a non-word.
* `synonyms.py::_NON_OBJECT_BLOCKLIST` -- added `"curl"` (a real RAM++
  posture tag that WordNet's physical-entity check incorrectly passes).
* **Known, deliberately NOT fixed yet** (separate follow-up, by request):
  a broader class of RAM++ action-tags (`catch`, `beat`, `swing`, `stand`,
  `shake`, `service`, `ride`, `drive`, `park`, `coach`, `line`, `back`, ...)
  pass the same WordNet physical-entity check for the same reason `curl`
  did. `is_likely_physical_object` checks ANY WordNet noun sense, not just
  the dominant one -- see `synonyms.py`'s module docstring for the two
  fix options discussed (per-word blocklist vs. first-sense-only check).

## Running this for real (on the server, with `MARINE/data/coco/{val2014,annotations}` and GPU)

```bash
# 0. Strategy 8-U's Step A must already exist (candidate_pool_cache.jsonl) --
#    reuse the existing one; text_objects.py/synonyms.py's bug fixes mean
#    re-running candidate_pool.py (if you choose to) will produce a
#    slightly cleaner O_init than the cached version, but the cached
#    version is still usable as-is.

# 1. (once) build the real object co-occurrence table for distractor bias
python marine/strategy8-union/cooccurrence.py \
    --instances_json ./data/coco/annotations/instances_val2014.json \
                      ./data/coco/annotations/instances_train2014.json \
    --output_file ./data/coco/cooccurrence_table.json

# 2. sample + score the probe pool for every image (needs GPU + real images)
python marine/strategy8-union/build_probe_pool.py \
    --candidate_pool_cache ./output/llava2/strategy8_union/candidate_pool_cache.jsonl \
    --image_folder ./data/coco/val2014 \
    --ram_tag_list_path <path to ram package's ram_tag_list.txt> \
    --cooccurrence_table ./data/coco/cooccurrence_table.json \
    --K 80 --tau_low 0.3 \
    --output_file ./output/llava2/strategy8_union_nc/probe_pool_cache.jsonl

# 3. fit the null-calibrated sorter for every image (pure numpy, fast)
python marine/strategy8-union/fit_null_calibration.py \
    --candidate_pool_cache ./output/llava2/strategy8_union/candidate_pool_cache.jsonl \
    --probe_pool_cache ./output/llava2/strategy8_union_nc/probe_pool_cache.jsonl \
    --sort_results_output ./output/llava2/strategy8_union_nc/sort_results.json

# 4. pick ~50 images for the sanity check (reusing the existing, already-
#    real 100-image report split so features are already cached for them)
python -c "
import json
split = json.load(open('./output/llava2/strategy8_union/split.json'))
json.dump(sorted(split['report_images'])[:50], open('./output/llava2/strategy8_union_nc/sanity_check_images.json', 'w'))
"

# 5. the sanity-check histogram (needs real COCO annotations)
python marine/strategy8-union/chair_histogram_nc.py \
    --candidate_pool_cache ./output/llava2/strategy8_union/candidate_pool_cache.jsonl \
    --sort_results_file ./output/llava2/strategy8_union_nc/sort_results.json \
    --coco_annotations_path ./data/coco/annotations \
    --image_list_file ./output/llava2/strategy8_union_nc/sanity_check_images.json \
    --epsilons 0.05 0.1 0.2 \
    --output_file ./output/llava2/strategy8_union_nc/report/sanity_check_histogram.png

# 6. the visual HTML report (no ground truth) -- see report_nc.generate_report_nc;
#    a thin CLI wrapper mirroring build_question_file.py's __main__ can be added
#    the same way once you've picked which question_file/answers_file to render
#    against (e.g. build one with fit_null_calibration.py --question_file ...
#    --epsilon 0.1, generate answers with generate.py exactly like Strategy 8-U,
#    then call report_nc.generate_report_nc with those paths).
```

## Design choices worth knowing about

* **Per-image, not pooled.** Unlike `gmm.py`'s `GlobalGMM` (which had to
  pool many images to get a stable fit, since ~10-25 candidates isn't
  enough for a 2-component 3D covariance), the null model here only ever
  needs the probe pool, whose size `K` is free and image-independent. So
  it's fit fresh per image and thrown away, exactly as the paper specifies
  (Section 3.4/6), with none of Strategy 8-U's global-pooling compromise.
* **Vocabulary V.** `probe_sampling.load_default_vocabulary` builds
  COCO-80 (derived from the SAME curated synonym table `synonyms.py`
  already uses, not a second hard-coded copy) plus, optionally, the real
  RAM++ tag list if you pass `--ram_tag_list_path` pointing at the `ram`
  package's bundled tag file. Without it, V is COCO-80 only, which still
  works but is smaller than the paper's intended vocabulary -- pass the
  real path on the server for the full-fidelity version.
* **Distractor bias is an optional enhancement, not a correctness
  requirement.** If no co-occurrence table is supplied,
  `sample_probe_pool` degrades gracefully to pure uniform sampling over
  the surviving vocabulary -- still a valid probe pool, just without the
  "hard negative" emphasis Section 3.1 describes.
* **Shrinkage.** Defaults to the analytic Ledoit-Wolf formula
  (`shrinkage=None`); pass a fixed `--shrinkage` value (e.g. 0.1, the
  paper's suggested constant) to `fit_null_calibration.py` for the
  fixed-lambda variant instead.
