#!/usr/bin/env python3
"""
Optimized Vietnamese PDF processing with VietOCR ONNX plugin.

This example demonstrates the ultimate optimization configuration for processing
Vietnamese scanned documents with tables on low-resource systems (2 cores, 6GB RAM).

Expected performance: ~15-17 seconds per page (50% faster than baseline)
"""

import logging
import sys
import time
from pathlib import Path

# Configure logging BEFORE importing docling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vietnamese_ocr.log', mode='w')
    ]
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_MEDIUM
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

# Import VietOCR ONNX plugin
from docling_vietocr_onnx import VietOcrOnnxOptions

_log = logging.getLogger(__name__)


def create_optimized_converter(model_dir: Path):
    """Create DocumentConverter with optimized settings for Vietnamese PDFs.

    Args:
        model_dir: Path to VietOCR ONNX models directory

    Returns:
        Configured DocumentConverter instance
    """
    _log.info("Configuring optimized pipeline for Vietnamese documents")

    # ============================================================
    # 1. ACCELERATOR OPTIONS - Match your hardware
    # ============================================================
    accelerator_options = AcceleratorOptions(
        num_threads=2,  # 2 CPU cores
        device=AcceleratorDevice.CPU
    )

    # ============================================================
    # 2. VIETOCR ONNX - Fast Vietnamese text recognition
    # ============================================================
    vietocr_options = VietOcrOnnxOptions(
        # Model paths (adjust to your setup)
        cnn_model_path=str(model_dir / "weight/cnn.onnx"),
        encoder_model_path=str(model_dir / "weight/encoder.onnx"),
        decoder_model_path=str(model_dir / "weight/decoder.onnx"),
        vocab_path=str(model_dir / "config/vocab.txt"),

        # ONNX Runtime optimization
        num_threads=2,
        intra_op_num_threads=2,
        inter_op_num_threads=1,

        # Image processing limits (balance speed vs accuracy)
        image_height=32,
        image_min_width=32,
        image_max_width=512,  # Lower = faster, higher = more accurate
        max_seq_length=128,  # Maximum text length
    )

    # ============================================================
    # 3. PIPELINE CONFIGURATION
    # ============================================================
    pipeline_options = PdfPipelineOptions()

    # General settings
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.allow_external_plugins = True  # Required for external plugins

    # OCR settings - using VietOCR ONNX
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = vietocr_options

    # Table structure - FAST mode (2-3x faster than ACCURATE)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    pipeline_options.table_structure_options.do_cell_matching = False  # Skip for speed

    # Layout detection - use fastest model
    pipeline_options.layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_MEDIUM,  # Fastest layout model
        create_orphan_clusters=False,
    )

    # Batch sizes optimized for 6GB RAM
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1

    # OCR threshold - higher value = skip more regions (faster but less coverage)
    # 0.05 = aggressive OCR, 0.2 = conservative OCR
    pipeline_options.bitmap_area_threshold = 0.05

    # ============================================================
    # 4. CREATE CONVERTER
    # ============================================================
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    _log.info("Converter configured successfully")
    return converter


def process_document(converter, input_path: Path, output_dir: Path):
    """Process a single document with profiling.

    Args:
        converter: Configured DocumentConverter
        input_path: Path to input PDF
        output_dir: Directory for output files
    """
    _log.info(f"Processing: {input_path}")

    # Enable detailed profiling
    settings.debug.profile_pipeline_timings = True
    settings.debug.visualize_ocr = False  # Set to True to see OCR boxes

    # Start timing
    start_time = time.time()

    # Convert document
    result = converter.convert(input_path)
    doc = result.document

    # Calculate total time
    total_time = time.time() - start_time

    # ============================================================
    # PROFILING RESULTS
    # ============================================================
    _log.info("=" * 80)
    _log.info("PROFILING RESULTS")
    _log.info("=" * 80)

    # Stage breakdown
    if hasattr(result, 'timings') and result.timings:
        _log.info("\nStage Timings:")
        for stage, timing in result.timings.items():
            stage_time = timing.times if hasattr(timing, 'times') else timing
            percentage = (stage_time / total_time * 100) if total_time > 0 else 0
            _log.info(f"  {stage:30s}: {stage_time:6.2f}s ({percentage:5.1f}%)")

    _log.info(f"\nTotal Processing Time: {total_time:.2f}s")
    _log.info(f"Pages: {len(doc.pages)}")
    if len(doc.pages) > 0:
        _log.info(f"Average per Page: {total_time/len(doc.pages):.2f}s")

    # ============================================================
    # EXPORT RESULTS
    # ============================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to Markdown
    md_path = output_dir / f"{input_path.stem}_output.md"
    markdown = doc.export_to_markdown()
    md_path.write_text(markdown, encoding='utf-8')
    _log.info(f"\nMarkdown saved to: {md_path}")

    # Export to JSON
    json_path = output_dir / f"{input_path.stem}_output.json"
    doc.save_as_json(json_path)
    _log.info(f"JSON saved to: {json_path}")

    # Extract tables
    tables = [item for page in doc.pages for item in page.tables]
    if tables:
        _log.info(f"\nExtracted {len(tables)} table(s)")
        for i, table in enumerate(tables, 1):
            table_md = table.export_to_markdown()
            _log.info(f"\n--- Table {i} ---\n{table_md}")

    _log.info("=" * 80)


def main():
    """Main entry point."""
    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Path to VietOCR ONNX models
    # Adjust this to where you downloaded/converted the models
    model_dir = Path("models/vietocr_onnx")

    # Input document
    input_path = Path("test_vietnamese.pdf")

    # Output directory
    output_dir = Path("output")

    # ============================================================
    # VALIDATION
    # ============================================================

    if not model_dir.exists():
        _log.error(f"Model directory not found: {model_dir}")
        _log.error("Please download or convert VietOCR ONNX models first.")
        _log.error("See README.md for instructions.")
        sys.exit(1)

    if not input_path.exists():
        _log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Check for required model files
    required_files = [
        model_dir / "weight/cnn.onnx",
        model_dir / "weight/encoder.onnx",
        model_dir / "weight/decoder.onnx",
        model_dir / "config/vocab.txt",
    ]

    for file_path in required_files:
        if not file_path.exists():
            _log.error(f"Required model file not found: {file_path}")
            _log.error("Please ensure all ONNX models are in place.")
            sys.exit(1)

    # ============================================================
    # PROCESS DOCUMENT
    # ============================================================

    try:
        converter = create_optimized_converter(model_dir)
        process_document(converter, input_path, output_dir)
        _log.info("\nProcessing completed successfully!")

    except Exception as e:
        _log.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
