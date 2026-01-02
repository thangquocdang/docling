#!/bin/bash
# Setup script to download and convert VietOCR models to ONNX

set -e

echo "============================================"
echo "VietOCR ONNX Model Setup"
echo "============================================"

# Check if ConvertVietOcr2Onnx exists
if [ ! -d "../ConvertVietOcr2Onnx" ]; then
    echo "Error: ConvertVietOcr2Onnx repository not found"
    echo "Please run from docling root directory:"
    echo "  git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q gdown pyyaml

# Create weight directory
echo ""
echo "Creating directories..."
mkdir -p weight
mkdir -p models/vietocr_onnx

# Download pretrained weights
echo ""
echo "Downloading pretrained VietOCR weights..."
echo "File ID: 1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4"

if [ ! -f "weight/transformerocr.pth" ]; then
    gdown "https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4" -O weight/transformerocr.pth
    echo "✓ Weights downloaded"
else
    echo "✓ Weights already exist, skipping download"
fi

# Convert to ONNX
echo ""
echo "Converting models to ONNX format..."
python convert_vietocr_official.py \
    --config ../ConvertVietOcr2Onnx/config/vgg-seq2seq.yml \
    --weight-path weight/transformerocr.pth \
    --output models/vietocr_onnx

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "ONNX models saved to: models/vietocr_onnx/"
echo ""
echo "Files created:"
echo "  - models/vietocr_onnx/weight/cnn.onnx"
echo "  - models/vietocr_onnx/weight/encoder.onnx"
echo "  - models/vietocr_onnx/weight/decoder.onnx"
echo "  - models/vietocr_onnx/config/vocab.txt"
echo ""
echo "Next steps:"
echo "  1. python example_optimized_vietnamese.py"
echo "  2. Enjoy fast Vietnamese OCR! 🚀"
echo ""
