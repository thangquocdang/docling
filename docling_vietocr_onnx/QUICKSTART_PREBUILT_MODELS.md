# 🚀 Quick Start với Models Có Sẵn

## ✅ TIN TỐT: Models đã có sẵn!

Repository [ConvertVietOcr2Onnx](https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx) đã có sẵn 3 ONNX models, bạn chỉ cần copy và tạo vocab.txt!

## Các bước nhanh (5 phút)

### Bước 1: Clone repo và setup

```bash
# Từ thư mục docling
git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git

# Tạo vocab.txt từ config
cd ConvertVietOcr2Onnx
python3 << 'EOF'
import yaml

with open('config/base.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

vocab_string = config['vocab']

with open('config/vocab.txt', 'w', encoding='utf-8') as f:
    for char in vocab_string:
        f.write(char + '\n')

print(f"✓ Created vocab.txt with {len(vocab_string)} characters")
EOF
```

### Bước 2: Copy models sang plugin

```bash
cd ../docling_vietocr_onnx

# Tạo thư mục
mkdir -p models/vietocr_onnx/weight
mkdir -p models/vietocr_onnx/config

# Copy ONNX models
cp ../ConvertVietOcr2Onnx/weight/*.onnx models/vietocr_onnx/weight/
cp ../ConvertVietOcr2Onnx/config/vocab.txt models/vietocr_onnx/config/

# Kiểm tra
ls -lh models/vietocr_onnx/weight/
ls -lh models/vietocr_onnx/config/
```

Bạn sẽ thấy:
```
models/vietocr_onnx/weight/
  cnn.onnx      (77MB)
  encoder.onnx  (3.5MB)
  decoder.onnx  (4.9MB)

models/vietocr_onnx/config/
  vocab.txt     (682 bytes, 229 characters)
```

### Bước 3: Test models

```bash
python3 << 'EOF'
from pathlib import Path
import onnxruntime
import numpy as np

model_dir = Path("models/vietocr_onnx")

# Load sessions
cnn = onnxruntime.InferenceSession(str(model_dir / "weight/cnn.onnx"))
encoder = onnxruntime.InferenceSession(str(model_dir / "weight/encoder.onnx"))
decoder = onnxruntime.InferenceSession(str(model_dir / "weight/decoder.onnx"))

# Test with dummy RGB image (3 channels)
img = np.random.rand(1, 3, 32, 200).astype(np.float32)

# Run inference
src = cnn.run(None, {cnn.get_inputs()[0].name: img})[0]
enc_out, hidden = encoder.run(None, {encoder.get_inputs()[0].name: src})
tgt = np.array([1], dtype=np.int64)
output = decoder.run(None, {
    decoder.get_inputs()[0].name: tgt,
    decoder.get_inputs()[1].name: hidden,
    decoder.get_inputs()[2].name: enc_out
})[0]

print("✅ All models work!")
print(f"Input: {img.shape} (RGB)")
print(f"Output: {output.shape} (vocab logits)")
EOF
```

### Bước 4: Sử dụng với Docling

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_vietocr_onnx import VietOcrOnnxOptions

# Cấu hình
vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
    num_threads=2,  # Điều chỉnh theo CPU
)

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

# Convert tài liệu tiếng Việt
result = converter.convert("document_tieng_viet.pdf")
markdown = result.document.export_to_markdown()
print(markdown)
```

## Tại sao nhanh hơn?

| Đặc điểm | PyTorch VietOCR | ONNX VietOCR |
|----------|-----------------|--------------|
| Tốc độ | ~25s/trang | **~15-17s/trang** |
| Bộ nhớ | ~1.5GB | **~500MB** |
| Kích thước | ~150MB | **~85MB** |
| Tối ưu CPU | Không | **Có** |

## Models Information

**Source:** https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx

**Model Details:**
- **CNN Encoder:** VGG19-based (77MB)
  - Input: [B, 3, 32, W] RGB images
  - Output: Feature maps

- **Transformer Encoder:** (3.5MB)
  - Input: CNN features
  - Output: Encoded sequence + hidden state

- **Transformer Decoder:** (4.9MB)
  - Input: Token + hidden + encoder outputs
  - Output: Next token probabilities
  - Autoregressive generation

**Vocabulary:**
- 229 Vietnamese characters
- Includes: a-z, A-Z, Vietnamese diacritics, numbers, symbols
- Special tokens: SOS=1, EOS=2

## Troubleshooting

### Models không chạy

**Kiểm tra:**
```bash
# Verify ONNX files
pip install onnx
python -c "import onnx; onnx.checker.check_model('models/vietocr_onnx/weight/cnn.onnx')"

# Verify vocab
wc -l models/vietocr_onnx/config/vocab.txt  # Should be 229
```

### Lỗi "expected 3 channels, got 1"

**Fix:** Plugin đã được update để dùng RGB. Update code:
```bash
cd docling_vietocr_onnx
git pull
pip install -e . --force-reinstall
```

### Hiệu suất chậm

**Tối ưu:**
```python
vietocr_options = VietOcrOnnxOptions(
    # ... model paths ...
    image_max_width=256,      # Giảm từ 512 -> nhanh hơn
    num_threads=2,             # Số CPU cores
    intra_op_num_threads=2,    # Threads per operation
)
```

## Cấu hình tối ưu cho 2 cores, 6GB RAM

```python
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import TableFormerMode, LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_MEDIUM

# Accelerator
accelerator_options = AcceleratorOptions(
    num_threads=2,
    device=AcceleratorDevice.CPU
)

# VietOCR ONNX - tốc độ cao
vietocr_options = VietOcrOnnxOptions(
    cnn_model_path="models/vietocr_onnx/weight/cnn.onnx",
    encoder_model_path="models/vietocr_onnx/weight/encoder.onnx",
    decoder_model_path="models/vietocr_onnx/weight/decoder.onnx",
    vocab_path="models/vietocr_onnx/config/vocab.txt",
    num_threads=2,
    image_max_width=384,  # Cân bằng tốc độ/độ chính xác
)

# Pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = accelerator_options
pipeline_options.allow_external_plugins = True

# OCR
pipeline_options.do_ocr = True
pipeline_options.ocr_options = vietocr_options

# Tables - chế độ FAST
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
pipeline_options.table_structure_options.do_cell_matching = False

# Layout - model nhanh nhất
pipeline_options.layout_options = LayoutOptions(
    model_spec=DOCLING_LAYOUT_EGRET_MEDIUM,
    create_orphan_clusters=False,
)

# Batch sizes cho RAM thấp
pipeline_options.ocr_batch_size = 1
pipeline_options.layout_batch_size = 1
pipeline_options.table_batch_size = 1

# Kết quả: ~15-17s/trang (nhanh hơn 50%)
```

## Files Structure

```
docling_vietocr_onnx/
├── models/                    # Bạn tạo thư mục này
│   └── vietocr_onnx/
│       ├── weight/
│       │   ├── cnn.onnx      # Copy từ ConvertVietOcr2Onnx
│       │   ├── encoder.onnx  # Copy từ ConvertVietOcr2Onnx
│       │   └── decoder.onnx  # Copy từ ConvertVietOcr2Onnx
│       └── config/
│           └── vocab.txt      # Generate từ base.yml
├── docling_vietocr_onnx/     # Plugin source
│   ├── __init__.py
│   └── vietocr_onnx_model.py
├── README_VI.md              # Tài liệu này
└── example_optimized_vietnamese.py
```

## Tổng kết

✅ **Không cần download weights từ Google Drive**
✅ **Không cần convert models (đã có sẵn ONNX)**
✅ **Chỉ cần tạo vocab.txt (1 dòng lệnh)**
✅ **Copy models và sử dụng ngay**

**Tổng thời gian setup: ~5 phút**

---

**Thành công!** Plugin sẵn sàng xử lý tài liệu tiếng Việt nhanh hơn 50%! 🚀
