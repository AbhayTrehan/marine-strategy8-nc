#!/usr/bin/env bash
# Strategy 8-U-NC sanity check: 4D features, CLIP semantic probes,
# GroundingDINO as 4th signal.
#
# Usage:
#   bash scripts/run_strategy8_nc_sanity_check.sh
#   N_IMAGES=100 EPSILONS="0.05 0.1 0.2" bash scripts/run_strategy8_nc_sanity_check.sh

set -e

# --- Tunable parameters (override via environment) -----------------------
RAM_TAG_LIST_PATH="${RAM_TAG_LIST_PATH:-$(python -c "import ram,os;print(os.path.join(os.path.dirname(ram.__file__),'data','ram_tag_list.txt'))" 2>/dev/null || echo "")}"
K="${K:-80}"
TAU_LOW="${TAU_LOW:-0.3}"
SHRINKAGE="${SHRINKAGE:-}"
N_IMAGES="${N_IMAGES:-50}"
EPSILONS="${EPSILONS:-0.05 0.1 0.2}"
SEED="${SEED:-242}"
MAX_OWLVIT="${MAX_OWLVIT:-200}"

# GroundingDINO (HF model name; empty = skip, run 3D only)
GDINO_MODEL="${GDINO_MODEL:-}"

# --- Paths ---------------------------------------------------------------
ORIGINAL_CANDIDATE_CACHE=./output/llava2/strategy8_union/candidate_pool_cache.jsonl
SPLIT_FILE=./output/llava2/strategy8_union/split.json
COCO_IMAGE_DIR=./data/coco/val2014
COCO_ANNOTATIONS_PATH=./data/coco/annotations

OUTPUT_DIR=./output/llava2/strategy8_union_nc
mkdir -p "$OUTPUT_DIR/report"

IMAGE_LIST="$OUTPUT_DIR/sanity_check_images.json"

# =========================================================================
echo "[1/6] Selecting $N_IMAGES sanity-check images..."
python -c "
import json
split = json.load(open('$SPLIT_FILE'))
images = sorted(split['report_images'])[:$N_IMAGES]
json.dump(images, open('$IMAGE_LIST', 'w'), indent=2)
print(f'Selected {len(images)} images')
"

# =========================================================================
# Determine the candidate cache to use (enriched with GDINO, or original)
CANDIDATE_POOL_CACHE="$ORIGINAL_CANDIDATE_CACHE"

if [ -n "$GDINO_MODEL" ]; then
    echo "[2/6] Enriching candidates with GroundingDINO (only selected images)..."
    # Always regenerate to avoid stale partial files from interrupted runs
    python ./marine/strategy8-union/enrich_gdino.py \
        --candidate_pool_cache "$ORIGINAL_CANDIDATE_CACHE" \
        --image_folder "$COCO_IMAGE_DIR" \
        --gdino_model "$GDINO_MODEL" \
        --image_list_file "$IMAGE_LIST" \
        --output_file "$OUTPUT_DIR/candidate_pool_cache_4d.jsonl"
    CANDIDATE_POOL_CACHE="$OUTPUT_DIR/candidate_pool_cache_4d.jsonl"
else
    echo "[2/6] No GDINO_MODEL set -> running in 3D mode (no s_gdino)"
fi

# =========================================================================
echo "[3/6] Sampling + scoring the probe pool (only selected images)..."
RAM_TAG_ARG=""
if [ -n "$RAM_TAG_LIST_PATH" ]; then
    RAM_TAG_ARG="--ram_tag_list_path $RAM_TAG_LIST_PATH"
fi
GDINO_ARGS=""
if [ -n "$GDINO_MODEL" ]; then
    GDINO_ARGS="--gdino_model $GDINO_MODEL"
fi

python ./marine/strategy8-union/build_probe_pool.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --image_folder "$COCO_IMAGE_DIR" \
    $RAM_TAG_ARG \
    $GDINO_ARGS \
    --K $K --tau_low $TAU_LOW --seed $SEED \
    --max_owlvit_candidates $MAX_OWLVIT \
    --min_K 30 \
    --vocab_embeddings_cache "$OUTPUT_DIR/vocab_clip_embeddings.npy" \
    --image_list_file "$IMAGE_LIST" \
    --output_file "$OUTPUT_DIR/probe_pool_cache.jsonl"

# =========================================================================
echo "[4/6] Fitting the null-calibrated conformal sorter..."
SHRINKAGE_ARG=""
if [ -n "$SHRINKAGE" ]; then
    SHRINKAGE_ARG="--shrinkage $SHRINKAGE"
fi
python ./marine/strategy8-union/fit_null_calibration.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --probe_pool_cache "$OUTPUT_DIR/probe_pool_cache.jsonl" \
    $SHRINKAGE_ARG \
    --sort_results_output "$OUTPUT_DIR/sort_results.json"

# =========================================================================
echo "[5/6] Building the sanity-check histogram..."
python ./marine/strategy8-union/chair_histogram_nc.py \
    --candidate_pool_cache "$CANDIDATE_POOL_CACHE" \
    --sort_results_file "$OUTPUT_DIR/sort_results.json" \
    --coco_annotations_path "$COCO_ANNOTATIONS_PATH" \
    --image_list_file "$IMAGE_LIST" \
    --epsilons $EPSILONS \
    --output_file "$OUTPUT_DIR/report/sanity_check_histogram.png"

# =========================================================================
echo "[6/6] Done."
echo "  Sort results:  $OUTPUT_DIR/sort_results.json"
echo "  Histogram:     $OUTPUT_DIR/report/sanity_check_histogram.png"
echo ""
echo "Next steps:"
echo "  1. Check the histogram and false-verification rates printed above"
echo "  2. To run on more images: N_IMAGES=200 bash scripts/run_strategy8_nc_sanity_check.sh"
