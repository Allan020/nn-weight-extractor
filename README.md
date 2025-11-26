# Neural Network Weight Extractor

A standalone tool for extracting and converting neural network weights from Keras/TensorFlow models to binary format with batch normalization folding. Designed for hardware acceleration workflows, embedded systems, and custom inference engines.

## Overview

This tool provides a clean two-step pipeline for converting trained neural network models into optimized binary weight and bias files:

```
Keras/TensorFlow Model → Darknet Format → Binary Files (weights.bin + bias.bin)
```

The tool performs batch normalization folding during extraction, eliminating BN operations and reducing computational requirements for inference - ideal for FPGA, ASIC, and embedded deployments.

## Key Features

- **Multiple Input Formats**: Keras H5 models, Darknet .weights files
- **Batch Normalization Folding**: Automatically folds BN into convolutional layers
- **Fast C++ Extractor**: Zero external dependencies, processes 50M parameters in <1 second
- **Architecture Support**: YOLOv2, YOLOv3, custom CNN architectures
- **Production Ready**: Comprehensive error handling, validation, and logging
- **Hardware Optimized**: Binary output format ready for hardware accelerators

## Installation

### Python Converter (Optional - for H5 models)

```bash
cd python
pip install -r requirements.txt
```

Requirements: Python 3.6+, TensorFlow/Keras, NumPy

### C++ Extractor

```bash
cd cpp
make
```

Requirements: C++11 compiler (g++, clang++)

## Quick Start

### Extract from Darknet Weights

If you already have Darknet .weights and .cfg files:

```bash
cd cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

Output: `weights.bin` and `bias.bin` in current directory.

### Convert from Keras H5

For complete Keras models (saved with `model.save()`):

```bash
# Step 1: Convert to Darknet format
cd python
python h5_to_darknet.py \
    --input model.h5 \
    --output-weights model.weights \
    --output-cfg model.cfg

# Step 2: Extract to binary
cd ../cpp
./weights_extractor --cfg model.cfg --weights model.weights
```

## Usage

### Python H5 Converter

```bash
python h5_to_darknet.py \
    --input model.h5 \
    --output-weights output.weights \
    --output-cfg output.cfg \
    [--img-size 416] \
    [--channels 3] \
    [--verbose]
```

**Options:**
- `--input`: Input Keras H5 model file
- `--output-weights`: Output Darknet weights file
- `--output-cfg`: Output Darknet configuration file
- `--img-size`: Input image size (default: from model)
- `--img-width/--img-height`: Specify width and height separately
- `--channels`: Number of input channels (default: 3)
- `--verbose`: Enable detailed output

**Note**: The H5 file must contain the complete model architecture (saved with `model.save()`, not `model.save_weights()`). For weights-only files, see the [YOLOv3 Guide](YOLOV3_GUIDE.md).

### C++ Binary Extractor

```bash
./weights_extractor \
    --cfg model.cfg \
    --weights model.weights \
    [--output-weights weights.bin] \
    [--output-bias bias.bin] \
    [--verbose]
```

**Options:**
- `--cfg`: Darknet configuration file
- `--weights`: Darknet weights file
- `--output-weights`: Output weights binary (default: weights.bin)
- `--output-bias`: Output bias binary (default: bias.bin)
- `--verbose`: Enable detailed layer-by-layer output

## What It Does

This tool extracts neural network weights and biases, then:

1. **Extracts weights** from Keras H5 or Darknet .weights formats
2. **Folds batch normalization** into convolutional layers for efficiency:
   ```
   new_weight = weight × (scale / √(variance + ε))
   new_bias = bias - mean × (scale / √(variance + ε))
   ```
3. **Converts to binary format** (32-bit floats, little-endian)
4. **Generates configuration** describing network architecture

### Why Batch Normalization Folding?

Folding BN into conv layers:
- ✓ Eliminates BN computations during inference
- ✓ Reduces memory bandwidth requirements  
- ✓ Maintains numerical accuracy
- ✓ Simplifies hardware implementation
- ✓ Reduces latency and power consumption

Perfect for FPGAs, ASICs, microcontrollers, and custom accelerators.

## Output Format

### weights.bin
- Binary file of 32-bit floats (IEEE 754, little-endian)
- Sequential convolutional layer weights (BN-folded)
- Format: `[layer0_weights, layer1_weights, ...]`

### bias.bin  
- Binary file of 32-bit floats (IEEE 754, little-endian)
- Sequential convolutional layer biases (BN-folded)
- Format: `[layer0_biases, layer1_biases, ...]`

## Examples

### YOLOv2 Extraction

```bash
cd cpp
./weights_extractor \
    --cfg yolov2.cfg \
    --weights yolov2.weights \
    --output-weights yolov2_w.bin \
    --output-bias yolov2_b.bin
```

### Custom Keras Model

```bash
cd python
python h5_to_darknet.py \
    --input my_model.h5 \
    --output-weights my_model.weights \
    --output-cfg my_model.cfg \
    --img-size 512 \
    --verbose

cd ../cpp
./weights_extractor \
    --cfg my_model.cfg \
    --weights my_model.weights
```

### Batch Processing

```bash
# Extract multiple models
cd cpp
for model in yolov2 yolov3 custom; do
    ./weights_extractor \
        --cfg ../models/${model}.cfg \
        --weights ../models/${model}.weights \
        --output-weights ../output/${model}_weights.bin \
        --output-bias ../output/${model}_bias.bin
done
```

## Supported Architectures

- **YOLOv2**: Fully tested and validated
- **YOLOv3**: Fully supported (see [YOLOv3 Guide](YOLOV3_GUIDE.md))
- **Custom CNNs**: Any Keras model with Conv2D + BatchNormalization
- **Non-standard architectures**: Grouped layers, custom patterns supported

The tool uses intelligent name-based matching to handle various layer arrangements.

## Project Structure

```
weight_converter_tool/
├── python/                    # Python H5 converter
│   ├── h5_to_darknet.py      # Main converter script
│   ├── model_parsers/        # Framework-specific parsers
│   ├── darknet_writer.py     # Darknet format writer
│   ├── cfg_generator.py      # Configuration generator
│   └── requirements.txt      # Python dependencies
│
├── cpp/                       # C++ binary extractor
│   ├── src/                  # Source files
│   ├── Makefile              # Build system
│   └── weights_extractor     # Compiled binary
│
├── examples/                  # Usage examples
│   ├── convert_yolov2.sh
│   └── convert_custom.sh
│
├── docs/                      # Additional documentation
├── README.md                  # This file
└── QUICKSTART.md             # Quick start guide
```

## Performance

**Python Converter:**
- Small models (<100 layers): <1 second
- Large models (YOLOv3, 106 layers): 2-5 seconds

**C++ Extractor:**
- YOLOv2 (50M params): <1 second
- YOLOv3 (62M params): ~2 seconds
- Memory efficient: Streams data, minimal RAM usage

## Troubleshooting

### H5 Loading Fails

**Issue**: Model cannot be loaded from H5 file

**Solution**: The H5 file may contain only weights (not full model). Options:
1. Re-save with `model.save()` instead of `model.save_weights()`
2. Use the original training script to build the architecture first
3. If you have existing .weights files, skip Python step and use C++ extractor directly

See [YOLOv3 Guide](YOLOV3_GUIDE.md) for details.

### Size Mismatches

**Issue**: Output file sizes don't match expectations

**Solution**: The tool extracts only convolutional layers with BN folding applied. This is expected and correct. Use `--verbose` to see layer-by-layer details.

### Compilation Errors

**Issue**: C++ compilation fails

**Solution**: 
```bash
cd cpp
make clean
make
```

Ensure C++11 or newer compiler is available.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

- Inspired by the [Darknet](https://github.com/pjreddie/darknet) framework
- YOLOv2/v3 architecture from Joseph Redmon's work
- Built for hardware acceleration research

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{weight_converter_tool,
  title = {Neural Network Weight Extractor},
  author = {Solomon Negussie Tesema},
  email = {solomon.negussie.tesema@gmail.com},
  year = {2025},
  url = {https://github.com/yourusername/weight_converter_tool}
}
```

## License

See [LICENSE](LICENSE.md)

## Support

- Documentation: See [docs/](docs/) folder
- Quick Start: [QUICKSTART.md](QUICKSTART.md)
- YOLOv3 Guide: [YOLOV3_GUIDE.md](YOLOV3_GUIDE.md)
- Issues: GitHub Issues page
- Discussions: GitHub Discussions

---

**Ready for**: FPGA implementation, ASIC design, embedded systems, custom accelerators, and edge AI deployment.
