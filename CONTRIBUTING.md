# Contributing to Neural Network Weight Extractor

Thank you for considering contributing to this project!

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Provide clear description of the problem
- Include model details, error messages, and steps to reproduce
- Mention your environment (OS, Python version, compiler version)

### Suggesting Features

- Open an issue describing the feature
- Explain the use case and expected behavior
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed
4. **Test your changes**:
   - Test with different model architectures
   - Verify output correctness
   - Check for memory leaks (C++ code)
5. **Commit with clear messages**:
   ```
   git commit -m "Add support for PyTorch models"
   ```
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

### Code Style

**Python:**
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and modular

**C++:**
- Use C++11 standard
- Follow consistent naming (snake_case for functions/variables)
- Add header comments for functions
- Avoid unnecessary complexity

### Testing

- Test with YOLOv2, YOLOv3, and custom models
- Verify numerical accuracy of BN folding
- Check edge cases (layers without bias, grouped architectures)
- Test on different platforms (Linux, macOS, Windows)

### Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Update QUICKSTART.md if workflow changes
- Create guides for complex features

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/weight_converter_tool
cd weight_converter_tool

# Python environment
cd python
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Build C++ extractor
cd ../cpp
make
```

## Areas for Contribution

- **New input formats**: PyTorch, ONNX, TFLite
- **Optimization**: Multi-threading, SIMD operations
- **Quantization**: INT8, INT16 support
- **Validation**: Output verification tools
- **Documentation**: Tutorials, architecture guides
- **Testing**: Unit tests, integration tests
- **Examples**: More model architectures

## Questions?

Open an issue with the "question" label.

## Code of Conduct

Be respectful and constructive. This project welcomes contributions from everyone.

