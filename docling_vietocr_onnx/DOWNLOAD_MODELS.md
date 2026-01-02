# Downloading VietOCR Models

Since automatic download from Google Drive may not work due to network restrictions, here are manual methods to obtain the models.

## Option 1: Download Pretrained PyTorch Weights (Recommended)

### Step 1: Download the weights manually

Go to this Google Drive link:
```
https://drive.google.com/file/d/1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4/view
```

Or use `gdown` if you have access:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4 -O weight/transformerocr.pth
```

### Step 2: Place the downloaded file

Move the downloaded `transformerocr.pth` to:
```
docling_vietocr_onnx/weight/transformerocr.pth
```

### Step 3: Convert to ONNX

```bash
cd docling_vietocr_onnx
python convert_vietocr_official.py \
    --config ../ConvertVietOcr2Onnx/config/vgg-seq2seq.yml \
    --weight-path weight/transformerocr.pth \
    --output models/vietocr_onnx
```

This will create:
- `models/vietocr_onnx/weight/cnn.onnx`
- `models/vietocr_onnx/weight/encoder.onnx`
- `models/vietocr_onnx/weight/decoder.onnx`
- `models/vietocr_onnx/config/vocab.txt`

## Option 2: Use Pre-converted ONNX Models

If someone has already converted the models to ONNX, you can use them directly:

### Required Files

You need these 4 files:

```
models/vietocr_onnx/
├── weight/
│   ├── cnn.onnx          (~80MB)
│   ├── encoder.onnx      (~5MB)
│   └── decoder.onnx      (~30MB)
└── config/
    └── vocab.txt         (~2KB)
```

### Where to Get Pre-converted Models

1. **Ask someone who has already converted**: If a colleague has run the conversion, they can share the ONNX files
2. **Convert on another machine**: Run the conversion on a machine with internet access, then copy the files
3. **Use cloud storage**: Upload/download via Dropbox, Google Drive, or other services

## Option 3: Train Your Own Model

If you want to train a custom VietOCR model on your own data:

### Step 1: Clone VietOCR

```bash
git clone https://github.com/pbcquoc/vietocr.git
cd vietocr
pip install -e .
```

### Step 2: Prepare Your Dataset

Follow VietOCR documentation:
- https://github.com/pbcquoc/vietocr#training

### Step 3: Train the Model

```bash
python train.py --config your_config.yml
```

### Step 4: Convert to ONNX

Use the trained weights with our conversion script:

```bash
cd ../docling_vietocr_onnx
python convert_vietocr_official.py \
    --config path/to/your/config.yml \
    --weight-path path/to/your/weights.pth \
    --output models/vietocr_onnx
```

## Verification

After obtaining the models, verify they work:

```bash
# Check files exist
ls -lh models/vietocr_onnx/weight/
ls -lh models/vietocr_onnx/config/

# Test with Python
python -c "
import onnx
print('Checking CNN...', end=' ')
onnx.checker.check_model('models/vietocr_onnx/weight/cnn.onnx')
print('✓')

print('Checking Encoder...', end=' ')
onnx.checker.check_model('models/vietocr_onnx/weight/encoder.onnx')
print('✓')

print('Checking Decoder...', end=' ')
onnx.checker.check_model('models/vietocr_onnx/weight/decoder.onnx')
print('✓')

print('All models valid!')
"
```

## Troubleshooting

### "Failed to use proxy" when downloading

This means your network blocks Google Drive. Solutions:
- Download manually via web browser
- Use a different network
- Ask someone to download and share the file

### "Module not found" errors during conversion

Install required packages:
```bash
pip install torch torchvision onnx onnxruntime pyyaml
```

### Conversion fails with shape errors

Make sure you're using:
- Python 3.9+
- PyTorch 2.0+
- ONNX 1.14+

### Out of memory during conversion

The conversion process needs ~4GB RAM. If you have less:
- Close other applications
- Use a machine with more RAM
- Get pre-converted ONNX files instead

## Alternative: Use VietOCR PyTorch Plugin

If you cannot convert to ONNX, you can create a PyTorch-based plugin instead (slower but works):

```python
# Will be ~2-3x slower than ONNX but doesn't require conversion
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
predictor = Predictor(config)
```

See the main README.md for PyTorch plugin implementation details.

## Getting Help

If you're stuck:
1. Check the main [README.md](README.md)
2. See [INSTALLATION.md](INSTALLATION.md)
3. Open an issue on GitHub with details about your environment

## Model Sizes

Expected file sizes after conversion:

| File | Size |
|------|------|
| cnn.onnx | ~80MB |
| encoder.onnx | ~5MB |
| decoder.onnx | ~30MB |
| vocab.txt | ~2KB |
| **Total** | **~115MB** |

Original PyTorch weights: ~150MB

ONNX models are slightly smaller and much faster for inference!
