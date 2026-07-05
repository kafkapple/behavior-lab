#!/bin/bash
# Full pipeline: MAMMAL fitting → DualPM dataset → Training → Inference
#
# This script demonstrates the complete workflow from raw mouse video
# to 3D point cloud generation.

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

MAMMAL_DIR="/home/joon/dev/MAMMAL_mouse"
DUALPM_DIR="/home/joon/dev/DualPM_Paper"

# Input data
DATASET_NAME="markerless_mouse_1_nerf"
START_FRAME=0
END_FRAME=100

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FITTING_RESULT="${DATASET_NAME}_${TIMESTAMP}"
DUALPM_DATASET="/home/joon/data/dualpm_mouse_${TIMESTAMP}"
WEIGHTS_DIR="${DUALPM_DIR}/weights/mouse_${TIMESTAMP}"

# Training settings
TRAIN_STEPS=50000
RESOLUTION=160

# ============================================================
# Step 1: Run MAMMAL mesh fitting
# ============================================================

echo "============================================================"
echo "Step 1: Running MAMMAL mesh fitting"
echo "============================================================"

cd "$MAMMAL_DIR"

# Activate MAMMAL environment
# conda activate mammal_stable

python fitter_articulation.py \
    dataset=default_markerless \
    fitter.start_frame=$START_FRAME \
    fitter.end_frame=$END_FRAME \
    fitter.with_render=true

# Get the actual fitting result folder name
FITTING_RESULT=$(ls -t results/fitting/ | head -1)
echo "Fitting result: $FITTING_RESULT"

# ============================================================
# Step 2: Convert to DualPM dataset
# ============================================================

echo ""
echo "============================================================"
echo "Step 2: Converting to DualPM dataset"
echo "============================================================"

cd "${DUALPM_DIR}/tools/mammal_converter"

python convert.py \
    --mammal_dir "$MAMMAL_DIR" \
    --fitting_result "$FITTING_RESULT" \
    --output_dir "$DUALPM_DATASET" \
    --resolution $RESOLUTION

# ============================================================
# Step 3: (Optional) Extract DINOv2 features
# ============================================================

echo ""
echo "============================================================"
echo "Step 3: Extracting DINOv2 features"
echo "============================================================"

# Check if feature extraction is available
FEATURE_DIR="/home/joon/dev/dualpm_features"
if [ -d "$FEATURE_DIR" ]; then
    cd "$FEATURE_DIR"
    python extract.py \
        --input_dir "${DUALPM_DATASET}/renders" \
        --output_dir "${DUALPM_DATASET}/features" \
        --resolution $RESOLUTION
else
    echo "Feature extraction not available, skipping..."
    echo "Features will be extracted on-the-fly during training."
fi

# ============================================================
# Step 4: Train DualPM model
# ============================================================

echo ""
echo "============================================================"
echo "Step 4: Training DualPM model"
echo "============================================================"

cd "$DUALPM_DIR"

python scripts/train.py \
    dataset_root="$DUALPM_DATASET" \
    train_config.save_path="$WEIGHTS_DIR" \
    train_config.steps=$TRAIN_STEPS \
    resolution=$RESOLUTION

# ============================================================
# Step 5: Inference on new data
# ============================================================

echo ""
echo "============================================================"
echo "Step 5: Running inference"
echo "============================================================"

# Use the trained model on test data
# (Assumes you have preprocessed test data with features)

TEST_FEAT_DIR="/home/joon/data/dualpm_mouse/feat"
TEST_MASK_DIR="/home/joon/data/dualpm_mouse/mask"
INFERENCE_OUTPUT="/home/joon/data/dualpm_mouse/results_${TIMESTAMP}"

if [ -d "$TEST_FEAT_DIR" ]; then
    python scripts/infer.py \
        weights_path="${WEIGHTS_DIR}/weights_${TRAIN_STEPS}.pth" \
        feat_dir="$TEST_FEAT_DIR" \
        mask_dir="$TEST_MASK_DIR" \
        output_dir="$INFERENCE_OUTPUT"

    echo ""
    echo "============================================================"
    echo "Pipeline Complete!"
    echo "============================================================"
    echo "Results:"
    echo "  - Training weights: $WEIGHTS_DIR"
    echo "  - Point clouds: $INFERENCE_OUTPUT"
    echo ""
    echo "Point cloud files (.ply) can be viewed with:"
    echo "  - MeshLab"
    echo "  - CloudCompare"
    echo "  - Open3D viewer"
else
    echo "Test data not found at $TEST_FEAT_DIR"
    echo "Run preprocessing first, then run inference manually."
fi
