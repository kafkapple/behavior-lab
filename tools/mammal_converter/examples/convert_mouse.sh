#!/bin/bash
# Example: Convert MAMMAL_mouse fitting results to DualPM dataset

# Configuration
MAMMAL_DIR="/home/joon/dev/MAMMAL_mouse"
FITTING_RESULT="markerless_mouse_1_nerf_20251126_234835"
OUTPUT_DIR="/home/joon/data/dualpm_mouse_train"

# Optional settings
RESOLUTION=160
TRAIN_SPLIT=0.9
CAMERA_VIEW=0

# Run conversion
cd "$(dirname "$0")/.."

python convert.py \
    --mammal_dir "$MAMMAL_DIR" \
    --fitting_result "$FITTING_RESULT" \
    --output_dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --train_split $TRAIN_SPLIT \
    --camera_view $CAMERA_VIEW

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. (Optional) Extract DINOv2 features:"
echo "   cd /home/joon/dev/dualpm_features"
echo "   python extract.py --input_dir $OUTPUT_DIR/renders --output_dir $OUTPUT_DIR/features"
echo ""
echo "2. Train DualPM model:"
echo "   cd /home/joon/dev/DualPM_Paper"
echo "   python scripts/train.py dataset_root=$OUTPUT_DIR train_config.save_path=./weights/mouse"
echo ""
echo "3. Inference on new mouse images:"
echo "   python scripts/infer.py weights_path=./weights/mouse/weights_100000.pth"
