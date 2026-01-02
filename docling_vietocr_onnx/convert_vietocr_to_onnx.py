#!/usr/bin/env python3
"""
Convert VietOCR PyTorch models to ONNX format.

This script downloads a VietOCR model and converts it to ONNX format
for use with the docling-vietocr-onnx plugin.

Usage:
    python convert_vietocr_to_onnx.py --config vgg_transformer --output models/vietocr_onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.onnx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger(__name__)


def convert_vietocr_to_onnx(config_name: str, output_dir: Path):
    """Convert VietOCR model to ONNX format.

    Args:
        config_name: VietOCR config name (e.g., 'vgg_transformer', 'vgg_seq2seq')
        output_dir: Directory to save ONNX models
    """
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError:
        _log.error("VietOCR not installed. Install with: pip install vietocr")
        return False

    _log.info(f"Loading VietOCR model: {config_name}")

    # Load VietOCR model
    config = Cfg.load_config_from_name(config_name)
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False  # Simpler for ONNX

    predictor = Predictor(config)
    model = predictor.model
    model.eval()

    # Create output directories
    weight_dir = output_dir / "weight"
    config_dir = output_dir / "config"
    weight_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Converting models to ONNX...")

    # ========================================
    # 1. CONVERT CNN
    # ========================================
    _log.info("Converting CNN encoder...")

    # Create dummy input
    dummy_img = torch.randn(1, 1, 32, 256)

    with torch.no_grad():
        src = model.cnn(dummy_img)

    cnn_path = weight_dir / "cnn.onnx"
    torch.onnx.export(
        model.cnn,
        dummy_img,
        str(cnn_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        verbose=False,
        input_names=['img'],
        output_names=['output'],
        dynamic_axes={
            'img': {3: 'length'},
            'output': {0: 'channel'}
        }
    )

    _log.info(f"✓ CNN saved to: {cnn_path}")

    # Verify
    onnx_model = onnx.load(str(cnn_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ CNN model verified")

    # ========================================
    # 2. CONVERT ENCODER
    # ========================================
    _log.info("Converting Transformer encoder...")

    with torch.no_grad():
        encoder_outputs, hidden = model.transformer.encoder(src)

    encoder_path = weight_dir / "encoder.onnx"
    torch.onnx.export(
        model.transformer.encoder,
        src,
        str(encoder_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        verbose=False,
        input_names=['src'],
        output_names=['encoder_outputs', 'hidden'],
        dynamic_axes={
            'src': {0: 'channel_input'},
            'encoder_outputs': {0: 'channel_output'}
        }
    )

    _log.info(f"✓ Encoder saved to: {encoder_path}")

    # Verify
    onnx_model = onnx.load(str(encoder_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ Encoder model verified")

    # ========================================
    # 3. CONVERT DECODER
    # ========================================
    _log.info("Converting Transformer decoder...")

    # Create dummy decoder input
    tgt = torch.tensor([[1]], dtype=torch.long)  # SOS token

    decoder_path = weight_dir / "decoder.onnx"
    torch.onnx.export(
        model.transformer.decoder,
        (tgt, hidden, encoder_outputs),
        str(decoder_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        verbose=False,
        input_names=['tgt', 'hidden', 'encoder_outputs'],
        output_names=['output', 'hidden_out', 'last'],
        dynamic_axes={
            'encoder_outputs': {0: 'channel_input'},
            'last': {0: 'channel_output'}
        }
    )

    _log.info(f"✓ Decoder saved to: {decoder_path}")

    # Verify
    onnx_model = onnx.load(str(decoder_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ Decoder model verified")

    # ========================================
    # 4. SAVE VOCABULARY
    # ========================================
    _log.info("Saving vocabulary...")

    vocab = predictor.vocab
    vocab_path = config_dir / "vocab.txt"

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for char in vocab.chars:
            f.write(char + '\n')

    _log.info(f"✓ Vocabulary saved to: {vocab_path}")
    _log.info(f"  Total characters: {len(vocab.chars)}")

    # ========================================
    # 5. TEST ONNX MODELS
    # ========================================
    _log.info("\nTesting ONNX models with ONNX Runtime...")

    try:
        import onnxruntime

        # Load sessions
        cnn_session = onnxruntime.InferenceSession(str(cnn_path))
        encoder_session = onnxruntime.InferenceSession(str(encoder_path))
        decoder_session = onnxruntime.InferenceSession(str(decoder_path))

        # Test CNN
        cnn_input = {cnn_session.get_inputs()[0].name: dummy_img.numpy()}
        src_onnx = cnn_session.run(None, cnn_input)[0]
        _log.info("✓ CNN inference successful")

        # Test Encoder
        encoder_input = {encoder_session.get_inputs()[0].name: src_onnx}
        encoder_outputs_onnx, hidden_onnx = encoder_session.run(None, encoder_input)
        _log.info("✓ Encoder inference successful")

        # Test Decoder
        tgt_np = tgt.numpy().astype(np.int64)
        decoder_input = {
            decoder_session.get_inputs()[0].name: tgt_np,
            decoder_session.get_inputs()[1].name: hidden_onnx,
            decoder_session.get_inputs()[2].name: encoder_outputs_onnx
        }
        output_onnx = decoder_session.run(None, decoder_input)
        _log.info("✓ Decoder inference successful")

        _log.info("\n✅ All models converted and tested successfully!")

    except ImportError:
        _log.warning("ONNX Runtime not installed. Skipping inference test.")
        _log.warning("Install with: pip install onnxruntime")
        _log.info("\n✅ Models converted successfully (not tested)")

    # ========================================
    # SUMMARY
    # ========================================
    _log.info("\n" + "=" * 60)
    _log.info("CONVERSION SUMMARY")
    _log.info("=" * 60)
    _log.info(f"Config: {config_name}")
    _log.info(f"Output directory: {output_dir}")
    _log.info(f"\nGenerated files:")
    _log.info(f"  - {cnn_path}")
    _log.info(f"  - {encoder_path}")
    _log.info(f"  - {decoder_path}")
    _log.info(f"  - {vocab_path}")
    _log.info("\nNext steps:")
    _log.info(f"  1. Update your config to use these model paths")
    _log.info(f"  2. Run: python example_optimized_vietnamese.py")
    _log.info("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert VietOCR models to ONNX format"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='vgg_transformer',
        choices=[
            'vgg_transformer',
            'vgg_seq2seq',
            'resnet_transformer',
            'resnet_seq2seq',
        ],
        help='VietOCR config name (default: vgg_transformer)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/vietocr_onnx'),
        help='Output directory for ONNX models (default: models/vietocr_onnx)'
    )

    args = parser.parse_args()

    _log.info("VietOCR to ONNX Converter")
    _log.info("=" * 60)

    # Check dependencies
    try:
        import vietocr
        _log.info(f"VietOCR version: {vietocr.__version__}")
    except ImportError:
        _log.error("VietOCR not installed!")
        _log.error("Install with: pip install vietocr")
        return 1

    try:
        import onnx
        _log.info(f"ONNX version: {onnx.__version__}")
    except ImportError:
        _log.error("ONNX not installed!")
        _log.error("Install with: pip install onnx")
        return 1

    # Convert models
    success = convert_vietocr_to_onnx(args.config, args.output)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
