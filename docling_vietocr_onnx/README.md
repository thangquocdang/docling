# Docling VietOCR ONNX Plugin

High-performance Vietnamese OCR plugin for [Docling](https://github.com/docling-project/docling) using ONNX Runtime.

## Features

- **Fast Vietnamese Text Recognition**: 2-3x faster than PyTorch-based VietOCR
- **ONNX Runtime Optimization**: Leverages ONNX Runtime's graph optimizations
- **Low Memory Footprint**: Optimized for systems with limited RAM (works on 6GB RAM)
- **CPU Optimized**: Excellent performance on CPU-only systems
- **Seamless Integration**: Works as a drop-in replacement for other Docling OCR engines

## Installation

### Prerequisites

- Python 3.9+
- Docling 2.0.0+
- ONNX Runtime 1.16.0+

### Install from Source

```bash
# Clone or download the plugin
cd docling-vietocr-onnx

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Optional GPU Support

For GPU acceleration (requires CUDA):

```bash
pip install -e ".[gpu]"
```

## Model Setup

### Option 1: Use Pre-converted ONNX Models

Download pre-converted ONNX models from the VietOCR ONNX repository:

```bash
# Create model directory
mkdir -p models/vietocr_onnx
cd models/vietocr_onnx

# Download ONNX models (you need to obtain these from the ConvertVietOcr2Onnx repo)
# Structure should be:
# models/vietocr_onnx/
# ├── weight/
# │   ├── cnn.onnx
# │   ├── encoder.onnx
# │   └── decoder.onnx
# └── config/
#     └── vocab.txt
```

### Option 2: Convert PyTorch Models to ONNX

Use the [ConvertVietOcr2Onnx](https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx) repository:

1. Clone the conversion repository:
```bash
git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git
cd ConvertVietOcr2Onnx
```

2. Follow the conversion instructions in the Converter.ipynb notebook

3. Copy the generated ONNX models:
```bash
cp -r weight/ ../models/vietocr_onnx/
cp -r config/ ../models/vietocr_onnx/
```

## Usage

### Basic Usage

```python
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_vietocr_onnx import VietOcrOnnxOptions

# Configure VietOCR ONNX
vietocr_options = VietOcrOnnxOptions(
    # Path to ONNX models
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",

    # Inference settings
    num_threads=2,
    intra_op_num_threads=2,
    inter_op_num_threads=1,
)

# Setup pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options
pipeline_options.allow_external_plugins = True  # Enable external plugins

# Create converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# Convert document
result = converter.convert("vietnamese_document.pdf")
doc = result.document

# Export results
markdown = doc.export_to_markdown()
print(markdown)
```

### Optimized Configuration for Speed

```python
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import TableFormerMode, LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_MEDIUM
from docling_vietocr_onnx import VietOcrOnnxOptions

# Accelerator settings (2 cores, 6GB RAM)
accelerator_options = AcceleratorOptions(
    num_threads=2,
    device=AcceleratorDevice.CPU
)

# VietOCR ONNX - optimized for speed
vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
    num_threads=2,
    intra_op_num_threads=2,
    inter_op_num_threads=1,
    # Image size limits for speed
    image_height=32,
    image_max_width=512,
)

# Pipeline configuration
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = accelerator_options
pipeline_options.allow_external_plugins = True

# OCR settings
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options

# Table structure - FAST mode
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
pipeline_options.table_structure_options.do_cell_matching = False

# Layout - fastest model
pipeline_options.layout_options = LayoutOptions(
    model_spec=DOCLING_LAYOUT_EGRET_MEDIUM,
    create_orphan_clusters=False,
)

# Batch sizes for low RAM
pipeline_options.ocr_batch_size = 1
pipeline_options.layout_batch_size = 1
pipeline_options.table_batch_size = 1

# Higher threshold to skip more OCR (optional - trade accuracy for speed)
pipeline_options.bitmap_area_threshold = 0.1

# Create converter with optimized settings
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
```

### Using with CLI

Enable the external plugin when using the `docling` command:

```bash
# Show available external plugins
docling --show-external-plugins

# Run with VietOCR ONNX
docling --allow-external-plugins --ocr-engine=vietocr_onnx document.pdf
```

### Profiling and Debugging

```python
from docling.datamodel.settings import settings

# Enable detailed timing information
settings.debug.profile_pipeline_timings = True

# Enable OCR visualization (saves detection boxes as images)
settings.debug.visualize_ocr = True

# Run conversion
result = converter.convert("document.pdf")

# Print timing breakdown
print("Pipeline timings:")
for stage, timing in result.timings.items():
    print(f"  {stage}: {timing.times}s")
```

## Performance Comparison

Expected performance on 2-core CPU with 6GB RAM processing Vietnamese scanned PDFs with tables:

| Configuration | Time per Page | Notes |
|--------------|---------------|-------|
| Tesseract + TableFormer ACCURATE | ~34s | Baseline |
| RapidOCR + TableFormer FAST | ~20s | General optimization |
| **VietOCR ONNX + TableFormer FAST** | **~15-17s** | Best for Vietnamese |
| VietOCR ONNX + No Tables | ~10s | Maximum speed |

VietOCR ONNX advantages:
- **2-3x faster** than PyTorch VietOCR
- **Better Vietnamese accuracy** than general OCR engines
- **Lower memory usage** (~500MB vs 1.5GB for PyTorch)
- **Optimized for CPU** inference

## Configuration Options

### VietOcrOnnxOptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cnn_model_path` | str | "weight/cnn.onnx" | Path to CNN ONNX model |
| `encoder_model_path` | str | "weight/encoder.onnx" | Path to encoder ONNX model |
| `decoder_model_path` | str | "weight/decoder.onnx" | Path to decoder ONNX model |
| `vocab_path` | str | "config/vocab.txt" | Path to vocabulary file |
| `image_height` | int | 32 | Input image height |
| `image_min_width` | int | 32 | Minimum image width |
| `image_max_width` | int | 512 | Maximum image width |
| `max_seq_length` | int | 128 | Maximum output sequence length |
| `num_threads` | int | 2 | Total ONNX threads |
| `intra_op_num_threads` | int | 2 | Threads per operation |
| `inter_op_num_threads` | int | 1 | Threads across operations |

## Architecture

The VietOCR ONNX model consists of three components:

1. **CNN Encoder**: Extracts visual features from text images
2. **Transformer Encoder**: Encodes CNN features into context representations
3. **Transformer Decoder**: Generates Vietnamese text autoregressively

All three models run via ONNX Runtime for optimal performance.

## Troubleshooting

### Model Files Not Found

Ensure your model paths are correct:

```python
from pathlib import Path

# Use absolute paths
model_dir = Path("/absolute/path/to/models/vietocr_onnx")

vietocr_options = VietOcrOnnxOptions(
    cnn_model_path=str(model_dir / "weight/cnn.onnx"),
    encoder_model_path=str(model_dir / "weight/encoder.onnx"),
    decoder_model_path=str(model_dir / "weight/decoder.onnx"),
    vocab_path=str(model_dir / "config/vocab.txt"),
)
```

### Plugin Not Detected

Make sure external plugins are enabled:

```python
pipeline_options.allow_external_plugins = True
```

### Low Performance

- Reduce `image_max_width` to 384 or 256 for faster processing
- Set `do_cell_matching = False` in table options
- Increase `bitmap_area_threshold` to skip more OCR regions
- Use `TableFormerMode.FAST` instead of ACCURATE

### Memory Issues

- Set batch sizes to 1: `ocr_batch_size = 1`
- Reduce `max_seq_length` to 64 or 96
- Process fewer pages at once

## Development

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black docling_vietocr_onnx/

# Lint
flake8 docling_vietocr_onnx/
```

## License

MIT License

## Acknowledgments

- [Docling](https://github.com/docling-project/docling) - Document processing framework
- [VietOCR](https://github.com/pbcquoc/vietor) - Vietnamese OCR models
- [ConvertVietOcr2Onnx](https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx) - ONNX conversion tools
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine

## Support

For issues and questions:
- Docling issues: https://github.com/docling-project/docling/issues
- Plugin issues: Create an issue in this repository
