# Installation Guide: VietOCR ONNX Plugin for Docling

Complete step-by-step installation guide for the VietOCR ONNX plugin.

## Prerequisites

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: 2+ CPU cores, 4GB+ RAM (tested on 2 cores, 6GB RAM)
- **Docling**: Version 2.0.0 or higher

## Installation Steps

### 1. Install Docling (if not already installed)

```bash
pip install docling
```

### 2. Install VietOCR ONNX Plugin

```bash
# Navigate to the plugin directory
cd docling_vietocr_onnx

# Install the plugin
pip install -e .
```

This will install the plugin and all required dependencies:
- onnxruntime
- numpy
- Pillow
- pydantic

### 3. Verify Installation

```bash
# Check if plugin is registered
python -c "from docling_vietocr_onnx import VietOcrOnnxModel; print('✓ Plugin installed')"
```

## Model Setup

You need ONNX models to use this plugin. Choose one of the methods below:

### Method 1: Convert Models Automatically (Recommended)

Use the provided conversion script:

```bash
# Install VietOCR (needed for conversion)
pip install vietocr torch onnx

# Run conversion script
python convert_vietocr_to_onnx.py --config vgg_transformer --output models/vietocr_onnx
```

This will:
1. Download VietOCR pretrained models
2. Convert to ONNX format
3. Save to `models/vietocr_onnx/`
4. Test the converted models

**Available configs:**
- `vgg_transformer` - Fastest, recommended for CPU
- `vgg_seq2seq` - Alternative architecture
- `resnet_transformer` - Slower but more accurate
- `resnet_seq2seq` - Most accurate, slowest

### Method 2: Manual Conversion

If you prefer manual control:

```bash
# Install dependencies
pip install vietocr torch onnx onnxruntime

# Clone conversion repository
git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git
cd ConvertVietOcr2Onnx

# Follow instructions in Converter.ipynb
jupyter notebook Converter.ipynb
```

Copy the generated files:
```bash
cp -r weight/ ../docling_vietocr_onnx/models/vietocr_onnx/
cp -r config/ ../docling_vietocr_onnx/models/vietocr_onnx/
```

### Method 3: Download Pre-converted Models

If someone has shared pre-converted models:

```bash
# Create directory structure
mkdir -p models/vietocr_onnx/weight
mkdir -p models/vietocr_onnx/config

# Download and place files
# - cnn.onnx → models/vietocr_onnx/weight/
# - encoder.onnx → models/vietocr_onnx/weight/
# - decoder.onnx → models/vietocr_onnx/weight/
# - vocab.txt → models/vietocr_onnx/config/
```

## Verification

### Test ONNX Models

```bash
pip install onnx
python -c "import onnx; onnx.checker.check_model('models/vietocr_onnx/weight/cnn.onnx'); print('✓ Models OK')"
```

### Test Plugin Integration

Create a test file `test_plugin.py`:

```python
from pathlib import Path
from docling_vietocr_onnx import VietOcrOnnxOptions

# Test model loading
options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
)

# Check if files exist
for attr in ['cnn_model_path', 'encoder_model_path', 'decoder_model_path', 'vocab_path']:
    path = Path(getattr(options, attr))
    if path.exists():
        print(f"✓ {attr}: {path}")
    else:
        print(f"✗ {attr}: NOT FOUND - {path}")
```

Run the test:
```bash
python test_plugin.py
```

All paths should show ✓.

### Full Integration Test

```python
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_vietocr_onnx import VietOcrOnnxOptions

# Configure
vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options
pipeline_options.allow_external_plugins = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print("✓ Converter created successfully!")
print("Plugin is ready to use!")
```

## Running Your First Conversion

```bash
# Use the optimized example
python example_optimized_vietnamese.py

# Or create your own script following the examples in README.md
```

## Troubleshooting

### Issue: "Model file not found"

**Solution**: Check your model paths are correct

```bash
ls -la models/vietocr_onnx/weight/
ls -la models/vietocr_onnx/config/
```

### Issue: "Plugin not detected"

**Solution**: Ensure external plugins are enabled

```python
pipeline_options.allow_external_plugins = True
```

### Issue: "ImportError: No module named 'docling_vietocr_onnx'"

**Solution**: Reinstall the plugin

```bash
cd docling_vietocr_onnx
pip install -e .
```

### Issue: "ONNX Runtime not found"

**Solution**: Install ONNX Runtime

```bash
pip install onnxruntime
```

For GPU support:
```bash
pip install onnxruntime-gpu
```

### Issue: Conversion fails with VietOCR errors

**Solution**: Install specific VietOCR version

```bash
pip install vietocr==0.3.9 torch==2.0.0
```

### Issue: Slow performance

**Solution**: Check configuration

1. Use `vgg_transformer` (fastest config)
2. Set `num_threads=2` (match your CPU cores)
3. Reduce `image_max_width` to 384 or 256
4. Use `TableFormerMode.FAST`

## Next Steps

1. **Read the documentation**: [README.md](README.md)
2. **Quick start guide**: [QUICKSTART.md](QUICKSTART.md)
3. **Run optimized example**: `example_optimized_vietnamese.py`
4. **Tune for your documents**: Adjust configuration parameters

## Directory Structure

After installation, your structure should look like:

```
docling_vietocr_onnx/
├── docling_vietocr_onnx/          # Plugin source code
│   ├── __init__.py
│   └── vietocr_onnx_model.py
├── models/                         # Model files (not in git)
│   └── vietocr_onnx/
│       ├── weight/
│       │   ├── cnn.onnx
│       │   ├── encoder.onnx
│       │   └── decoder.onnx
│       └── config/
│           └── vocab.txt
├── setup.py                        # Package setup
├── pyproject.toml                  # Modern package config
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── INSTALLATION.md                 # This file
├── example_optimized_vietnamese.py # Usage example
└── convert_vietocr_to_onnx.py     # Model conversion script
```

## Support

- **Docling Issues**: https://github.com/docling-project/docling/issues
- **VietOCR**: https://github.com/pbcquoc/vietocr
- **ONNX Runtime**: https://onnxruntime.ai/

Happy document processing! 🚀
