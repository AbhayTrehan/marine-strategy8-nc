#!/usr/bin/env bash
# Strategy 8-U-NC sanity check: probe pool -> null-calibrated conformal
# sorter -> probe-vs-candidate histograms (real CHAIR ground truth) ->
# visual report (no ground truth).
#
# Prerequisites (run once, already done for Strategy 8-U):
#   - ./output/llava2/strategy8_union/candidate_pool_cache.jsonl (Step A)
#   - ./output/llava2/strategy8_union/split.json (image split; reused here
#     for a real, already-cached 50-image sanity-check sample)
#   - ./data/coco/val2014 (real images) and ./data/coco/annotations (real
#     COCO instance + caption annotations)
#
# This is a NEW script -- it does not modify eval_strategy8_union.sh or
# anything else. Mirrors that script's conventions (PYTHONPATH for the
# externally-cloned LLaVA repo, MODEL_VERSION/SEED variables).
#
# Usage:
#   bash scripts/run_strategy8_nc_sanity_check.sh
#   N_IMAGES=30 EPSILONS="0.05 0.2" bash scripts/run_strategy8_nc_sanity_check.sh
#
# All tunable parameters below are plain shell variables with env-var
# overrides (there is no positional/flag argument parsing in this script --
# set them via the environment as shown above, or just edit the values below).

set -e

export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

RAM_TAG_LIST_PATH="${RAM_TAG_LIST_PATH:-}"   # e.g. $(python -c "import ram,os;print(os.path.join(os.path.dirname(ram.__file__),'data','ram_tag_list.txt'))")
K="${K:-80}"
TAU_LOW="${TAU_LOW:-0.3}"
SHRINKAGE="${SHRINKAGE:-}"   # empty -> analytic Ledoit-Wolf; set e.g. SHRINKAGE=0.1 to override
N_IMAGES="${N_IMAGES:-50}"
EPSILONS="${EPSILONS:-0.05 0.1 0.2}"
SEED="${SEED:-242}"

CANDIDATE_POOL_CACHE=./output/llava2/strategy8_union/candidate_pool_cache.jsonl
SPLIT_FILE=./output/llava2/strategy8_union/split.json
COCO_IMAGE_DIR=./data/coco/val2014
COCO_ANNOTATIONS_PATH=./data/coco/annotations
COOCCURRENCE_TABLE=./data/coco/cooccurrence_table.json

OUTPUT_DIR=./output/llava2/strategy8_union_nc
mkdir -p "$OUTPUT_DIR/report"

echo "[1/5] Selecting $N_IMAGES sanity-check images from the existing real report split..."
python -c "
import json
split = json.load(open('$SPLIT_FILE'))
images = sorted(split['report_images'])[:$N_IMAGES]
json.dump(images, open('$OUTPUT_DIR/sanity_check_images.json', 'w'), indent=2)
print(f'Selected {len(images)} images -> $OUTPUT_DIR/sanity_check_images.json')
"

if [ ! -f "$COOCCURRENCE_TABLE" ]; then
  echo "[2/5] Building the real object co-occurrence table (distractor bias)..."
  python ./marine/strategy8-union/cooccurrence.py \
      --instances_json "$COCO_ANNOTATIONS_PATH/instances_val2014.json" \
                        "$COCO_ANNOTATIONS_PATH/instances_train2014.json" \
      --output_file "$COOCCURRENCE_TABLE"
else
  echo "[2/5] Reusing existing co-occurrence table at $COOCCURRENCE_TABLE"
fi

echo "[3/5] Sampling + scoring the probe pool (needs GPU + real images)..."
RAM_TAG_ARG=""
if [ -n "$RAM_TAG_LIST_PATH" ]; then
  RAM_TAG_ARG="--ram_tag_list_path $RAM_TAG_LIST_PATH"
fi
python ./marine/strategy8-union/build_probe_pool.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --image_folder "$COCO_IMAGE_DIR" \
    $RAM_TAG_ARG \
    --cooccurrence_table "$COOCCURRENCE_TABLE" \
    --K $K --tau_low $TAU_LOW --seed $SEED \
    --output_file "$OUTPUT_DIR/probe_pool_cache.jsonl"

echo "[4/5] Fitting the null-calibrated conformal sorter (pure numpy, fast)..."
SHRINKAGE_ARG=""
if [ -n "$SHRINKAGE" ]; then
  SHRINKAGE_ARG="--shrinkage $SHRINKAGE"
fi
python ./marine/strategy8-union/fit_null_calibration.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --probe_pool_cache "$OUTPUT_DIR/probe_pool_cache.jsonl" \
    $SHRINKAGE_ARG \
    --sort_results_output "$OUTPUT_DIR/sort_results.json"

echo "[5/5] Building the sanity-check histogram (real CHAIR ground truth)..."
python ./marine/strategy8-union/chair_histogram_nc.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --sort_results_file "$OUTPUT_DIR/sort_results.json" \
    --coco_annotations_path "$COCO_ANNOTATIONS_PATH" \
    --image_list_file "$OUTPUT_DIR/sanity_check_images.json" \
    --epsilons $EPSILONS \
    --output_file "$OUTPUT_DIR/report/sanity_check_histogram.png"

echo "Done. See:"
echo "  $OUTPUT_DIR/sort_results.json                    (per-image conformal sort, reusable across epsilons)"
echo "  $OUTPUT_DIR/report/sanity_check_histogram.png    (probe vs candidate scores, real CHAIR ground truth)"
echo ""
echo "For the no-ground-truth visual HTML report (report_nc.generate_report_nc), first generate"
echo "captions with a question file built via fit_null_calibration.py --question_file ... --epsilon <eps>"
echo "(same as Strategy 8-U's generate.py step), then call generate_report_nc with those paths."
