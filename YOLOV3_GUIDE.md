# Working with Weights-Only Model Files

## Understanding Weights-Only H5 Files

Some Keras H5 files contain **only weights**, not the complete model architecture. This is common in older training workflows where models were saved with:

```python
model.save_weights('trained_weights.h5')  # Only weights
```

Instead of:

```python
model.save('complete_model.h5')  # Full model + weights
```

**Why this matters**: To load weights-only files, the model architecture must be reconstructed first.

## Detecting Weights-Only Files

Try to convert the file. If you see this error:

```
Note: Could not load as complete model: ...
This H5 file appears to contain only weights (not the full model).
```

Then you have a weights-only file.

## Solutions

### Solution 1: Re-save as Complete Model (Recommended)

If you have access to the original training code:

```python
# Load model with architecture
from your_model_definition import build_model

model = build_model(input_shape=(416, 416, 3), num_classes=80)
model.load_weights('trained_weights.h5')

# Save complete model
model.save('complete_model.h5')
```

Then use our converter:

```bash
python h5_to_darknet.py \
    --input complete_model.h5 \
    --output-weights model.weights \
    --output-cfg model.cfg
```

### Solution 2: Use Existing .weights Files

If `.weights` files already exist (from previous conversions):

```bash
# Skip Python step, go directly to binary extraction
cd cpp
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    --output-weights output_weights.bin \
    --output-bias output_bias.bin
```

### Solution 3: Manual Architecture Reconstruction

Create a script to rebuild the model:

```python
from tensorflow import keras

# Define architecture
def build_yolov3(input_shape=(608, 608, 3), num_classes=80):
    # ... your model architecture ...
    return model

# Load weights
model = build_yolov3()
model.load_weights('weights_only.h5')

# Save complete model
model.save('complete_model.h5')
```

## YOLOv3 Specific Notes

YOLOv3 models from popular repositories often use weights-only saving for efficiency:

**Common Pattern:**
```python
# In training script
model.save_weights(log_dir + 'trained_weights_final.h5')
```

**To convert these:**

1. **Locate model definition file** (usually `model.py` or `yolo3/model.py`)
2. **Rebuild model** using the definition
3. **Load weights** into the model
4. **Save complete model**
5. **Convert** using our tool

Example for typical YOLOv3 structure:

```python
from yolo3.model import yolo_body
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Rebuild model
num_anchors = 9
num_classes = 80
input_shape = (608, 608, 3)

yolo_model = yolo_body(
    Input(shape=(None, None, 3)), 
    num_anchors // 3, 
    num_classes
)

# Load weights
yolo_model.load_weights('trained_weights.h5')

# Save complete
yolo_model.save('complete_yolov3.h5')
```

## Testing Your Solution

After converting, verify the output:

```bash
# Check file sizes
ls -lh *.weights *.cfg

# Extract with verbose mode
cd cpp
./weights_extractor --cfg model.cfg --weights model.weights --verbose

# Verify output
ls -lh *.bin
```

Expected output for YOLOv3:
- 69 convolutional layers
- ~60M parameters
- weights.bin: ~240 MB
- bias.bin: ~350 KB

## Best Practices for Future Models

When saving your own models:

```python
# Always save complete model for easier deployment
model.save('my_model_complete.h5')

# If you also need weights-only (for transfer learning):
model.save_weights('my_model_weights_only.h5')

# Keep both files with clear naming
```

## Common Issues

### "Model architecture missing"

**Cause**: H5 file is weights-only

**Fix**: Rebuild architecture, then load weights

### "Layer size mismatch"

**Cause**: Wrong architecture for the weights

**Fix**: Ensure architecture exactly matches training configuration

### "Cannot find model definition"

**Cause**: Original training code not available

**Fix**: 
1. Check model repository for architecture files
2. Look for published papers describing the architecture
3. Use pre-converted `.weights` files if available

## Summary

**Weights-Only Files:**
- Common in older workflows
- Require architecture reconstruction
- Can be converted after proper model building

**Complete Model Files:**
- Contain architecture + weights
- Can be directly converted
- Preferred for deployment

**Recommended Approach:**
- Always save complete models for deployment
- Keep weights-only for training/research if needed
- Document your model architecture clearly

---

For standard model conversion, see [QUICKSTART.md](QUICKSTART.md).
