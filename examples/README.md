# Usage Examples

This directory contains example scripts demonstrating how to use the Neural Network Weight Extractor.

## Available Examples

### convert_yolov2.sh

Extracts weights from YOLOv2 model to binary format.

**Prerequisites:**
- Download YOLOv2 weights and config files
- Edit script to point to correct file paths

**Usage:**
```bash
chmod +x convert_yolov2.sh
./convert_yolov2.sh
```

### convert_custom.sh

Template for converting your own Keras models.

**Usage:**
1. Edit the script configuration section:
   ```bash
   INPUT_MODEL="path/to/your/model.h5"
   IMG_WIDTH=416
   IMG_HEIGHT=416
   CHANNELS=3
   ```

2. Run the script:
   ```bash
   chmod +x convert_custom.sh
   ./convert_custom.sh
   ```

## Quick Commands

### Extract from Darknet Weights

```bash
cd ../cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

### Convert from Keras H5

```bash
# Step 1: H5 to Darknet
cd ../python
python h5_to_darknet.py \
    --input model.h5 \
    --output-weights model.weights \
    --output-cfg model.cfg

# Step 2: Darknet to Binary
cd ../cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

## Model Files

Example models can be downloaded from:

**YOLOv2:**
- Weights: https://pjreddie.com/media/files/yolov2.weights
- Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg

**YOLOv3:**
- Weights: https://pjreddie.com/media/files/yolov3.weights
- Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

## Output Location

By default, scripts create an `output/` directory for results:
```
examples/
├── output/
│   ├── model_weights.bin
│   ├── model_bias.bin
│   └── ...
```

## Tips

1. **Check file paths** before running scripts
2. **Use verbose mode** (`--verbose`) for detailed output
3. **Verify output sizes** match expectations
4. **Keep temporary files** for debugging if needed

## Troubleshooting

### "Model file not found"

Edit the script and update the `INPUT_MODEL` or `WEIGHTS_FILE` path.

### "Conversion failed"

Check:
- Model file is valid and readable
- Python dependencies installed (`pip install -r requirements.txt`)
- For H5 files: must contain full model (not just weights)

### "Permission denied"

Make scripts executable:
```bash
chmod +x *.sh
```

## Creating Your Own Examples

Copy and modify `convert_custom.sh`:

```bash
cp convert_custom.sh convert_mymodel.sh
# Edit convert_mymodel.sh with your model details
./convert_mymodel.sh
```

## More Information

- Main documentation: [../README.md](../README.md)
- Quick start: [../QUICKSTART.md](../QUICKSTART.md)
- Weights-only files: [../YOLOV3_GUIDE.md](../YOLOV3_GUIDE.md)

