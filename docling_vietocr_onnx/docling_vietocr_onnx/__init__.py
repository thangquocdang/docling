"""Docling plugin for VietOCR ONNX models."""

from docling_vietocr_onnx.vietocr_onnx_model import (
    VietOcrOnnxModel,
    VietOcrOnnxOptions,
)

__version__ = "0.1.0"

__all__ = [
    "VietOcrOnnxModel",
    "VietOcrOnnxOptions",
    "ocr_engines",
]


def ocr_engines():
    """Plugin factory function for registering VietOCR ONNX with Docling.

    This function is called by Docling's plugin system to discover
    available OCR engines.

    Returns:
        dict: Dictionary containing list of OCR engine classes
    """
    return {
        "ocr_engines": [
            VietOcrOnnxModel,
        ]
    }
