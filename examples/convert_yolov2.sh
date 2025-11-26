#!/bin/bash
# Example: Extract YOLOv2 weights to binary format

set -e

echo "YOLOv2 Weight Extraction Example"
echo "=================================="
echo

# Configuration
CFG_FILE="path/to/yolov2.cfg"
WEIGHTS_FILE="path/to/yolov2.weights"
OUTPUT_DIR="../output"

# Check if weights file exists
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Error: YOLOv2 weights file not found: $WEIGHTS_FILE"
    echo ""
    echo "Please download YOLOv2 weights:"
    echo "  wget https://pjreddie.com/media/files/yolov2.weights"
    echo ""
    echo "And YOLOv2 config:"
    echo "  wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg"
    echo ""
    echo "Then edit this script to point to the correct paths."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract weights using C++ extractor
echo "Extracting weights from YOLOv2 model..."
cd ../cpp

./weights_extractor \
    --cfg "$CFG_FILE" \
    --weights "$WEIGHTS_FILE" \
    --output-weights "$OUTPUT_DIR/yolov2_weights.bin" \
    --output-bias "$OUTPUT_DIR/yolov2_bias.bin" \
    --verbose

echo ""
echo "Extraction complete!"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/yolov2_*.bin

echo ""
echo "Files ready for hardware deployment."
