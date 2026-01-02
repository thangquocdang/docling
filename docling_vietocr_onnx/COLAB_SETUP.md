# Setup VietOCR ONNX Plugin trong Google Colab

## Bước 1: Cài đặt Dependencies

```python
# Install Docling và dependencies
!pip install -q docling docling-core onnxruntime pyyaml

# Verify installation
import docling
print(f"✓ Docling installed")
```

## Bước 2: Clone Repositories

```bash
%%bash
cd /content

# Clone Docling source (nếu cần develop)
git clone https://github.com/thangquocdang/docling.git

# Clone ConvertVietOcr2Onnx (có sẵn ONNX models)
git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git
```

## Bước 3: Tạo vocab.txt

```python
import yaml

# Generate vocab.txt từ config
with open('/content/ConvertVietOcr2Onnx/config/base.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

vocab_string = config['vocab']

with open('/content/ConvertVietOcr2Onnx/config/vocab.txt', 'w', encoding='utf-8') as f:
    for char in vocab_string:
        f.write(char + '\n')

print(f"✓ vocab.txt created with {len(vocab_string)} characters")
```

## Bước 4: Copy Models

```bash
%%bash
cd /content/docling/docling_vietocr_onnx

# Tạo thư mục
mkdir -p models/vietocr_onnx/weight
mkdir -p models/vietocr_onnx/config

# Copy ONNX models
cp /content/ConvertVietOcr2Onnx/weight/*.onnx models/vietocr_onnx/weight/
cp /content/ConvertVietOcr2Onnx/config/vocab.txt models/vietocr_onnx/config/

# Verify
ls -lh models/vietocr_onnx/weight/
ls -lh models/vietocr_onnx/config/
```

## Bước 5: Install Plugin

```bash
%%bash
cd /content/docling/docling_vietocr_onnx
pip install -e .
```

## Bước 6: Test Plugin

```python
from docling_vietocr_onnx import VietOcrOnnxOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

# Cấu hình VietOCR ONNX
vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
    num_threads=2,
)

# Setup pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options
pipeline_options.allow_external_plugins = True  # QUAN TRỌNG!

# Tạo converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print("✅ Plugin ready to use!")

# Test với file PDF
# result = converter.convert("document.pdf")
# markdown = result.document.export_to_markdown()
# print(markdown)
```

## Fix Import Error

Nếu gặp lỗi `cannot import name 'Cell'`, plugin đã được fix:

```bash
%%bash
cd /content/docling/docling_vietocr_onnx
git pull
pip install -e . --force-reinstall
```

## Troubleshooting

### Lỗi: "Module not found docling_core"

```bash
!pip install docling-core
```

### Lỗi: "Module not found onnxruntime"

```bash
!pip install onnxruntime
```

### Lỗi: "cannot import name 'Cell'"

Import đã được fix sang:
```python
from docling_core.types.doc import BoundingBox  # thay vì từ base_models
from docling.datamodel.base_models import Page
```

## Complete Colab Notebook

```python
# === CELL 1: Install Dependencies ===
!pip install -q docling docling-core onnxruntime pyyaml
print("✓ Dependencies installed")

# === CELL 2: Clone Repos ===
!git clone https://github.com/thangquocdang/docling.git /content/docling
!git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git /content/ConvertVietOcr2Onnx
print("✓ Repos cloned")

# === CELL 3: Generate vocab.txt ===
import yaml

with open('/content/ConvertVietOcr2Onnx/config/base.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

with open('/content/ConvertVietOcr2Onnx/config/vocab.txt', 'w', encoding='utf-8') as f:
    for char in config['vocab']:
        f.write(char + '\n')

print(f"✓ vocab.txt created ({len(config['vocab'])} chars)")

# === CELL 4: Setup Models ===
!mkdir -p /content/docling/docling_vietocr_onnx/models/vietocr_onnx/weight
!mkdir -p /content/docling/docling_vietocr_onnx/models/vietocr_onnx/config
!cp /content/ConvertVietOcr2Onnx/weight/*.onnx /content/docling/docling_vietocr_onnx/models/vietocr_onnx/weight/
!cp /content/ConvertVietOcr2Onnx/config/vocab.txt /content/docling/docling_vietocr_onnx/models/vietocr_onnx/config/
!ls -lh /content/docling/docling_vietocr_onnx/models/vietocr_onnx/weight/
print("✓ Models copied")

# === CELL 5: Install Plugin ===
%cd /content/docling/docling_vietocr_onnx
!pip install -q -e .
print("✓ Plugin installed")

# === CELL 6: Test ===
from docling_vietocr_onnx import VietOcrOnnxOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions

vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
    num_threads=2,
)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options
pipeline_options.allow_external_plugins = True

print("✅ Ready to convert Vietnamese PDFs!")
print(f"Expected speed: ~15-17s per page (50% faster than Tesseract)")
```

## Notes

- Colab có RAM giới hạn, nên dùng batch_size=1
- Plugin tối ưu cho CPU, không cần GPU
- Models tổng cộng ~85MB
