# Quick Start Guide

Get started with Neural Network Weight Extractor in 5 minutes.

## Prerequisites

- **Python 3.6+** with pip (optional, for H5 conversion)
- **C++11 compiler** (g++ or clang++)

## Installation

### 1. Build C++ Extractor

```bash
cd cpp
make
```

That's it! The C++ extractor has no external dependencies.

### 2. Install Python Dependencies (Optional)

Only needed if converting from Keras H5 models:

```bash
cd python
pip install -r requirements.txt
```

## Quick Examples

### Example 1: Extract from Darknet Weights (Fastest)

If you already have `.weights` and `.cfg` files:

```bash
cd cpp
./weights_extractor \
    --cfg /path/to/model.cfg \
    --weights /path/to/model.weights
```

**Output**: `weights.bin` and `bias.bin` in current directory.

**Time**: ~1 second for typical models.

### Example 2: Convert from Keras H5

For complete Keras models:

```bash
# Step 1: Convert H5 to Darknet format
cd python
python h5_to_darknet.py \
    --input /path/to/model.h5 \
    --output-weights model.weights \
    --output-cfg model.cfg

# Step 2: Extract to binary
cd ../cpp
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights
```

**Output**: `weights.bin` and `bias.bin`

**Time**: ~5 seconds total for typical models.

### Example 3: Custom Output Location

```bash
cd cpp
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --output-weights /path/to/output/weights.bin \
    --output-bias /path/to/output/bias.bin
```

## Understanding the Output

### weights.bin
- Contains all convolutional layer weights
- Batch normalization already folded in
- Format: 32-bit floats (IEEE 754, little-endian)
- Sequential: [layer0_weights, layer1_weights, ...]

### bias.bin
- Contains all convolutional layer biases
- Batch normalization already folded in
- Format: 32-bit floats (IEEE 754, little-endian)
- Sequential: [layer0_biases, layer1_biases, ...]

## Common Workflows

### Workflow 1: YOLOv2/v3 Models

```bash
# Download pre-trained weights
wget https://pjreddie.com/media/files/yolov2.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg

# Extract
cd cpp
./weights_extractor --cfg yolov2.cfg --weights yolov2.weights
```

**Result**: Binary files ready for hardware deployment.

### Workflow 2: Custom Keras Model

```bash
# Train and save model
python train.py  # Your training script
# Ensure it uses: model.save('my_model.h5')

# Convert to binary
cd python
python h5_to_darknet.py \
    --input my_model.h5 \
    --output-weights my_model.weights \
    --output-cfg my_model.cfg

cd ../cpp
./weights_extractor --cfg my_model.cfg --weights my_model.weights
```

**Result**: Optimized weights for your custom model.

### Workflow 3: Batch Processing

```bash
cd cpp

# Process multiple models
for model in model1 model2 model3; do
    ./weights_extractor \
        --cfg ../models/${model}.cfg \
        --weights ../models/${model}.weights \
        --output-weights ../output/${model}_w.bin \
        --output-bias ../output/${model}_b.bin
done
```

## Verification

Check extracted files:

```bash
# List output files
ls -lh *.bin

# Check file sizes (should match model size)
du -h weights.bin bias.bin

# View first few bytes (should be valid floats)
hexdump -C weights.bin | head
```

## Troubleshooting

### "Command not found: ./weights_extractor"

```bash
cd cpp
make clean && make
chmod +x weights_extractor
```

### "ImportError: No module named 'tensorflow'"

```bash
cd python
pip install tensorflow
# or
pip install -r requirements.txt
```

### "Cannot open weights file"

Check file path is correct:
```bash
ls -l /path/to/weights.weights
```

### Different file sizes than expected

This is normal! The tool:
- Extracts only convolutional layers
- Folds batch normalization (reduces size)
- Skips pooling, activation, and routing layers

Use `--verbose` flag to see what's being extracted:
```bash
./weights_extractor --cfg model.cfg --weights model.weights --verbose
```

## Next Steps

- **Read full documentation**: See [README.md](README.md)
- **Learn about BN folding**: See "What It Does" section in README
- **Handle special cases**: See [YOLOV3_GUIDE.md](YOLOV3_GUIDE.md) for weights-only H5 files
- **Integrate with your project**: Binary files are ready to use

## Example Output

Successful extraction looks like:

```
Darknet Weights Extractor
=========================

Parsing configuration file: yolov2.cfg
Found 23 convolutional layers

Reading weights file: yolov2.weights

Extracting and processing layers...
Processed layer 23/23

=========================
Extraction completed successfully!
=========================

Statistics:
  Layers processed: 23
  Total weights: 237,339
  Total biases: 10,761
  Total parameters: 248,100

Output files:
  Weights: weights.bin
  Biases:  bias.bin
```

## Performance Expectations

**Extraction Times:**
- Small models (5-20 layers): <0.5 seconds
- Medium models (20-50 layers): 0.5-1 second
- Large models (50-100+ layers): 1-3 seconds

**Memory Usage:**
- Minimal - streams data during processing
- Peak usage ≈ largest layer size × 2

## Getting Help

- **Quick questions**: Check [README.md](README.md) FAQ section
- **Special cases**: See [YOLOV3_GUIDE.md](YOLOV3_GUIDE.md)
- **Issues**: Open a GitHub issue with:
  - Your command
  - Error message
  - Model details (architecture, size)
  - Operating system

---

**Ready to extract!** Your binary weight files will be optimized for hardware deployment. 🚀
