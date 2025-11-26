#!/bin/bash
# Template: Convert custom Keras model to binary format

set -e

echo "Custom Model Conversion Template"
echo "================================="
echo

# ============================================
# CONFIGURATION - Edit these variables
# ============================================

# Input model (Keras H5 file)
INPUT_MODEL="path/to/your/model.h5"

# Output directory
OUTPUT_DIR="../output"

# Model input size (adjust to match your model)
IMG_WIDTH=416
IMG_HEIGHT=416
CHANNELS=3

# Output file names
OUTPUT_WEIGHTS="custom_weights.bin"
OUTPUT_BIAS="custom_bias.bin"

# ============================================
# Conversion pipeline
# ============================================

# Validate input
if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Model file not found: $INPUT_MODEL"
    echo ""
    echo "Please edit this script and set INPUT_MODEL to a valid .h5 file"
    echo ""
    echo "Example:"
    echo "  INPUT_MODEL=\"/path/to/my_trained_model.h5\""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Input model: $INPUT_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: ${IMG_WIDTH}x${IMG_HEIGHT}x${CHANNELS}"
echo

# Temporary files
TEMP_WEIGHTS="$OUTPUT_DIR/temp_model.weights"
TEMP_CFG="$OUTPUT_DIR/temp_model.cfg"

echo "Step 1: Converting Keras model to Darknet format..."
cd ../python

python h5_to_darknet.py \
    --input "$INPUT_MODEL" \
    --output-weights "$TEMP_WEIGHTS" \
    --output-cfg "$TEMP_CFG" \
    --img-width "$IMG_WIDTH" \
    --img-height "$IMG_HEIGHT" \
    --channels "$CHANNELS" \
    --verbose

if [ $? -ne 0 ]; then
    echo "Error: Python conversion failed"
    echo ""
    echo "Common issues:"
    echo "  - Model file is weights-only (see YOLOV3_GUIDE.md)"
    echo "  - Missing dependencies (run: pip install -r requirements.txt)"
    echo "  - Invalid model format"
    exit 1
fi

echo ""
echo "Step 2: Extracting weights to binary format..."
cd ../cpp

./weights_extractor \
    --cfg "$TEMP_CFG" \
    --weights "$TEMP_WEIGHTS" \
    --output-weights "$OUTPUT_DIR/$OUTPUT_WEIGHTS" \
    --output-bias "$OUTPUT_DIR/$OUTPUT_BIAS" \
    --verbose

if [ $? -ne 0 ]; then
    echo "Error: Weight extraction failed"
    exit 1
fi

echo ""
echo "================================="
echo "Conversion complete!"
echo "================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/$OUTPUT_WEIGHTS "$OUTPUT_DIR"/$OUTPUT_BIAS

# Summary
echo ""
echo "Summary:"
echo "  Model: $INPUT_MODEL"
echo "  Weights: $OUTPUT_DIR/$OUTPUT_WEIGHTS"
echo "  Biases: $OUTPUT_DIR/$OUTPUT_BIAS"
echo ""
echo "These binary files are ready for hardware deployment."
echo ""

# Optional: Keep or remove temporary files
read -p "Remove temporary Darknet files (.weights, .cfg)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$TEMP_WEIGHTS" "$TEMP_CFG"
    echo "Temporary files removed."
else
    echo "Temporary files kept:"
    echo "  $TEMP_WEIGHTS"
    echo "  $TEMP_CFG"
fi
