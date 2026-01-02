"""Setup script for docling-vietocr-onnx plugin."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="docling-vietocr-onnx",
    version="0.1.0",
    author="VietOCR ONNX Plugin Contributors",
    description="VietOCR ONNX integration for Docling document processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/docling-project/docling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "docling>=2.0.0",
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "docling": [
            "vietocr_onnx = docling_vietocr_onnx:ocr_engines",
        ],
    },
)
