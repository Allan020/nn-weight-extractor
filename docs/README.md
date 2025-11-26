# Documentation

Additional documentation for the Neural Network Weight Extractor.

## Available Documents

### Main Documentation
- **[README.md](../README.md)** - Complete project documentation
- **[QUICKSTART.md](../QUICKSTART.md)** - 5-minute quick start guide
- **[YOLOV3_GUIDE.md](../YOLOV3_GUIDE.md)** - Handling weights-only model files

### Contributing
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[LICENSE](../LICENSE)** - Project license

## Topics

### Getting Started
- [Installation](../README.md#installation)
- [Quick Examples](../QUICKSTART.md#quick-examples)
- [Basic Usage](../README.md#usage)

### Advanced Topics
- [Batch Normalization Folding](../README.md#why-batch-normalization-folding)
- [Output Format Specification](../README.md#output-format)
- [Supported Architectures](../README.md#supported-architectures)

### Special Cases
- [Weights-Only H5 Files](../YOLOV3_GUIDE.md)
- [Custom Architectures](../README.md#supported-architectures)
- [Grouped Layer Patterns](../YOLOV3_GUIDE.md)

### Development
- [Project Structure](../README.md#project-structure)
- [Performance](../README.md#performance)
- [Contributing](../CONTRIBUTING.md)

## API Reference

### Python Converter

```python
python h5_to_darknet.py [OPTIONS]

Required:
  --input PATH            Input Keras H5 model
  --output-weights PATH   Output Darknet weights
  --output-cfg PATH       Output Darknet config

Optional:
  --img-size INT          Input image size
  --img-width INT         Input width
  --img-height INT        Input height
  --channels INT          Input channels (default: 3)
  --verbose               Enable verbose output
```

### C++ Extractor

```bash
./weights_extractor [OPTIONS]

Required:
  --cfg PATH              Darknet config file
  --weights PATH          Darknet weights file

Optional:
  --output-weights PATH   Output weights binary
  --output-bias PATH      Output bias binary
  --verbose               Enable verbose output
```

## File Formats

### Input Formats
- **Keras H5** (.h5, .hdf5) - Complete models only
- **Darknet Weights** (.weights) - Binary format
- **Darknet Config** (.cfg) - Text format

### Output Formats
- **weights.bin** - 32-bit IEEE 754 floats, little-endian
- **bias.bin** - 32-bit IEEE 754 floats, little-endian

## Troubleshooting Guide

### Common Issues

**Model Loading Fails**
- Check H5 file contains complete model
- See [YOLOV3_GUIDE.md](../YOLOV3_GUIDE.md) for weights-only files

**Size Mismatches**
- Expected - only conv layers extracted
- BN folding reduces size
- Use `--verbose` for details

**Compilation Errors**
- Run `make clean && make`
- Ensure C++11 compiler available

## Examples Collection

See [examples/](../examples/) directory for:
- YOLOv2 extraction
- Custom model conversion
- Batch processing scripts

## Additional Resources

### External Documentation
- [Darknet Framework](https://pjreddie.com/darknet/)
- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)

### Related Projects
- [Darknet (Original)](https://github.com/pjreddie/darknet)
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Support

- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: [Your contact]

## Version History

See [CHANGELOG.md](CHANGELOG.md) (if available) for version history and updates.

