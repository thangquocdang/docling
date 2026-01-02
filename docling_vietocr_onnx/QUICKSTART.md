# Quick Start Guide: VietOCR ONNX for Docling

This guide will help you set up and use VietOCR ONNX models with Docling in under 10 minutes.

## Step 1: Install the Plugin

```bash
cd docling-vietocr-onnx
pip install -e .
```

## Step 2: Get ONNX Models

You have two options:

### Option A: Download Pre-converted Models (Recommended)

If someone has already converted the models, download them and place in this structure:

```
models/vietocr_onnx/
├── weight/
│   ├── cnn.onnx
│   ├── encoder.onnx
│   └── decoder.onnx
└── config/
    └── vocab.txt
```

### Option B: Convert Models Yourself

1. **Install VietOCR and conversion tools:**

```bash
pip install vietocr torch onnx
git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git
cd ConvertVietOcr2Onnx
```

2. **Download VietOCR pretrained model:**

```python
# In Python
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Choose a config (vgg_transformer is fastest)
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'

# Download the model (will be cached)
detector = Predictor(config)
```

The model will be downloaded to `~/.cache/vietocr/` or similar location.

3. **Convert to ONNX using the notebook:**

Open `Converter.ipynb` in Jupyter and follow the steps to convert:
- CNN model
- Encoder model
- Decoder model

Or run this Python script:

```python
import torch
import torch.onnx
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Load model
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'
predictor = Predictor(config)
model = predictor.model
model.eval()

# Create dummy input
img = torch.randn(1, 1, 32, 256)

# Export CNN
src = model.cnn(img)
torch.onnx.export(
    model.cnn, img, "weight/cnn.onnx",
    export_params=True, opset_version=12,
    input_names=['img'], output_names=['output'],
    dynamic_axes={'img': {3: 'length'}, 'output': {0: 'channel'}}
)

# Export Encoder
encoder_outputs, hidden = model.transformer.encoder(src)
torch.onnx.export(
    model.transformer.encoder, src, "weight/encoder.onnx",
    export_params=True, opset_version=11,
    input_names=['src'], output_names=['encoder_outputs', 'hidden'],
    dynamic_axes={'src': {0: "channel_input"}, 'encoder_outputs': {0: 'channel_output'}}
)

# Export Decoder (simplified - see full code in repo)
tgt = torch.tensor([[1]])  # SOS token
torch.onnx.export(
    model.transformer.decoder,
    (tgt, hidden, encoder_outputs),
    "weight/decoder.onnx",
    export_params=True, opset_version=11,
    input_names=['tgt', 'hidden', 'encoder_outputs'],
    output_names=['output', 'hidden_out', 'last'],
)

print("Models converted successfully!")
```

4. **Extract vocabulary:**

```python
# Save vocabulary to file
vocab = predictor.vocab
with open("config/vocab.txt", "w", encoding="utf-8") as f:
    for char in vocab.chars:
        f.write(char + "\n")
```

5. **Verify ONNX models:**

```bash
pip install onnx
python -c "import onnx; onnx.checker.check_model('weight/cnn.onnx')"
python -c "import onnx; onnx.checker.check_model('weight/encoder.onnx')"
python -c "import onnx; onnx.checker.check_model('weight/decoder.onnx')"
```

## Step 3: Test the Plugin

Create a simple test script:

```python
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_vietocr_onnx import VietOcrOnnxOptions

# Configure VietOCR ONNX
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

# Test with a Vietnamese PDF
result = converter.convert("test_vietnamese.pdf")
print(result.document.export_to_markdown())
```

## Step 4: Run Optimized Example

Use the provided optimized example:

```bash
# Edit paths in example_optimized_vietnamese.py
python example_optimized_vietnamese.py
```

## Expected Performance

On a 2-core CPU with 6GB RAM:

- **Baseline (Tesseract)**: ~34s per page
- **With VietOCR ONNX**: ~15-17s per page ✨
- **50% faster!** 🚀

## Troubleshooting

### "Model file not found"

Check your paths:
```python
from pathlib import Path
print(Path("models/vietocr_onnx/weight/cnn.onnx").exists())
```

Use absolute paths if needed:
```python
model_dir = Path("/full/path/to/models/vietocr_onnx")
```

### "Plugin not detected"

Make sure you:
1. Installed the plugin: `pip install -e .`
2. Enabled external plugins: `pipeline_options.allow_external_plugins = True`

Check if plugin is registered:
```bash
docling --show-external-plugins
```

### "ONNX Runtime error"

Install/upgrade ONNX Runtime:
```bash
pip install --upgrade onnxruntime
```

For GPU:
```bash
pip install onnxruntime-gpu
```

### Slow performance

- Lower `image_max_width` to 384 or 256
- Set `do_table_structure = False` if you don't have tables
- Increase `bitmap_area_threshold` to 0.2

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_optimized_vietnamese.py](example_optimized_vietnamese.py) for advanced usage
- Tune parameters for your specific documents

## Support

- Docling: https://github.com/docling-project/docling
- VietOCR: https://github.com/pbcquoc/vietocr
- ONNX Conversion: https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx
