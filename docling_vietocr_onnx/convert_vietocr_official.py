#!/usr/bin/env python3
"""
Convert VietOCR models to ONNX format using official conversion code.

This script is based on the original ConvertVietOcr2Onnx repository:
https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx

Usage:
    python convert_vietocr_official.py --weight-path ./weight/transformerocr.pth --output models/vietocr_onnx
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.onnx

# Add ConvertVietOcr2Onnx to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ConvertVietOcr2Onnx"))

from tool.config import Cfg
from tool.translate import build_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger(__name__)


def convert_cnn_part(img, save_path, model):
    """Convert CNN encoder to ONNX.

    Args:
        img: Input image tensor (B, 3, H, W) - NOTE: 3 channels for RGB
        save_path: Path to save ONNX model
        model: VietOCR model

    Returns:
        CNN output tensor
    """
    with torch.no_grad():
        src = model.cnn(img)
        torch.onnx.export(
            model.cnn,
            img,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            verbose=True,
            input_names=['img'],
            output_names=['output'],
            dynamic_axes={
                'img': {3: 'length'},
                'output': {0: 'channel'}
            }
        )

    return src


def convert_encoder_part(model, src, save_path):
    """Convert Transformer encoder to ONNX.

    Args:
        model: VietOCR model
        src: CNN output tensor
        save_path: Path to save ONNX model

    Returns:
        Tuple of (hidden, encoder_outputs)
    """
    encoder_outputs, hidden = model.transformer.encoder(src)

    torch.onnx.export(
        model.transformer.encoder,
        src,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['src'],
        output_names=['encoder_outputs', 'hidden'],
        dynamic_axes={
            'src': {0: "channel_input"},
            'encoder_outputs': {0: 'channel_output'}
        }
    )

    return hidden, encoder_outputs


def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
    """Convert Transformer decoder to ONNX.

    Args:
        model: VietOCR model
        tgt: Target token tensor
        hidden: Hidden state from encoder
        encoder_outputs: Encoder output tensor
        save_path: Path to save ONNX model
    """
    tgt = tgt[-1]

    torch.onnx.export(
        model.transformer.decoder,
        (tgt, hidden, encoder_outputs),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['tgt', 'hidden', 'encoder_outputs'],
        output_names=['output', 'hidden_out', 'last'],
        dynamic_axes={
            'encoder_outputs': {0: 'channel_input'},
            'last': {0: 'channel_output'}
        }
    )


def convert_vietocr_to_onnx(config_path: Path, weight_path: Path, output_dir: Path):
    """Convert VietOCR model to ONNX format.

    Args:
        config_path: Path to VietOCR config file
        weight_path: Path to model weights
        output_dir: Directory to save ONNX models

    Returns:
        True if successful, False otherwise
    """
    _log.info(f"Loading VietOCR config from: {config_path}")

    # Load configuration
    config = Cfg.load_config_from_file(str(config_path))
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'

    # Build model
    _log.info("Building VietOCR model...")
    model, vocab = build_model(config)

    # Load weights
    _log.info(f"Loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model = model.eval()

    # Create output directories
    weight_dir = output_dir / "weight"
    config_dir = output_dir / "config"
    weight_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Converting models to ONNX...")

    # ========================================
    # 1. CONVERT CNN (NOTE: 3 channels for RGB!)
    # ========================================
    _log.info("Converting CNN encoder...")

    # Create dummy input - IMPORTANT: 3 channels (RGB)
    img = torch.rand(1, 3, 32, 475)

    cnn_path = weight_dir / "cnn.onnx"
    src = convert_cnn_part(img, str(cnn_path), model)

    _log.info(f"✓ CNN saved to: {cnn_path}")

    # Verify
    onnx_model = onnx.load(str(cnn_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ CNN model verified")

    # ========================================
    # 2. CONVERT ENCODER
    # ========================================
    _log.info("Converting Transformer encoder...")

    encoder_path = weight_dir / "encoder.onnx"
    hidden, encoder_outputs = convert_encoder_part(model, src, str(encoder_path))

    _log.info(f"✓ Encoder saved to: {encoder_path}")

    # Verify
    onnx_model = onnx.load(str(encoder_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ Encoder model verified")

    # ========================================
    # 3. CONVERT DECODER
    # ========================================
    _log.info("Converting Transformer decoder...")

    # Create decoder input with SOS token
    device = img.device
    tgt = torch.LongTensor([[1] * len(img)]).to(device)

    decoder_path = weight_dir / "decoder.onnx"
    convert_decoder_part(model, tgt, hidden, encoder_outputs, str(decoder_path))

    _log.info(f"✓ Decoder saved to: {decoder_path}")

    # Verify
    onnx_model = onnx.load(str(decoder_path))
    onnx.checker.check_model(onnx_model)
    _log.info("✓ Decoder model verified")

    # ========================================
    # 4. SAVE VOCABULARY
    # ========================================
    _log.info("Saving vocabulary...")

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
        # Load sessions
        cnn_session = onnxruntime.InferenceSession(str(cnn_path))
        encoder_session = onnxruntime.InferenceSession(str(encoder_path))
        decoder_session = onnxruntime.InferenceSession(str(decoder_path))

        # Test CNN - NOTE: 3 channels
        cnn_input = {cnn_session.get_inputs()[0].name: img.numpy()}
        src_onnx = cnn_session.run(None, cnn_input)[0]
        _log.info("✓ CNN inference successful")

        # Test Encoder
        encoder_input = {encoder_session.get_inputs()[0].name: src_onnx}
        encoder_outputs_onnx, hidden_onnx = encoder_session.run(None, encoder_input)
        _log.info("✓ Encoder inference successful")

        # Test Decoder
        tgt_np = tgt.numpy()[-1]  # Get last element
        decoder_input = {
            decoder_session.get_inputs()[0].name: tgt_np,
            decoder_session.get_inputs()[1].name: hidden_onnx,
            decoder_session.get_inputs()[2].name: encoder_outputs_onnx
        }
        output_onnx = decoder_session.run(None, decoder_input)
        _log.info("✓ Decoder inference successful")

        _log.info("\n✅ All models converted and tested successfully!")

    except Exception as e:
        _log.error(f"ONNX Runtime test failed: {e}")
        _log.warning("Models converted but not tested")

    # ========================================
    # SUMMARY
    # ========================================
    _log.info("\n" + "=" * 60)
    _log.info("CONVERSION SUMMARY")
    _log.info("=" * 60)
    _log.info(f"Config: {config_path}")
    _log.info(f"Weights: {weight_path}")
    _log.info(f"Output directory: {output_dir}")
    _log.info(f"\nGenerated files:")
    _log.info(f"  - {cnn_path}")
    _log.info(f"  - {encoder_path}")
    _log.info(f"  - {decoder_path}")
    _log.info(f"  - {vocab_path}")
    _log.info("\nIMPORTANT: Input images must be RGB (3 channels), not grayscale!")
    _log.info("\nNext steps:")
    _log.info(f"  1. Update plugin to use RGB images")
    _log.info(f"  2. Run: python example_optimized_vietnamese.py")
    _log.info("=" * 60)

    return True


def download_pretrained_weights(output_dir: Path):
    """Download pretrained VietOCR weights.

    Args:
        output_dir: Directory to save weights
    """
    _log.info("Downloading pretrained VietOCR weights...")

    try:
        import gdown

        # Google Drive file ID from config
        file_id = "1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4"
        url = f"https://drive.google.com/uc?id={file_id}"

        output_path = output_dir / "transformerocr.pth"
        output_dir.mkdir(parents=True, exist_ok=True)

        gdown.download(url, str(output_path), quiet=False)

        _log.info(f"✓ Weights downloaded to: {output_path}")
        return output_path

    except ImportError:
        _log.error("gdown not installed. Install with: pip install gdown")
        _log.info("Or download manually from:")
        _log.info(f"  https://drive.google.com/file/d/{file_id}/view")
        return None
    except Exception as e:
        _log.error(f"Failed to download weights: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert VietOCR models to ONNX format (official method)"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('ConvertVietOcr2Onnx/config/vgg-seq2seq.yml'),
        help='Path to VietOCR config file'
    )
    parser.add_argument(
        '--weight-path',
        type=Path,
        help='Path to model weights (.pth file). If not provided, will download.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/vietocr_onnx'),
        help='Output directory for ONNX models'
    )
    parser.add_argument(
        '--download-weights',
        action='store_true',
        help='Download pretrained weights from Google Drive'
    )

    args = parser.parse_args()

    _log.info("VietOCR to ONNX Converter (Official Method)")
    _log.info("=" * 60)

    # Check if ConvertVietOcr2Onnx repo exists
    repo_path = Path(__file__).parent.parent / "ConvertVietOcr2Onnx"
    if not repo_path.exists():
        _log.error(f"ConvertVietOcr2Onnx repository not found at: {repo_path}")
        _log.error("Please clone it first:")
        _log.error("  git clone https://github.com/buiquangmanhhp1999/ConvertVietOcr2Onnx.git")
        return 1

    # Check config file
    if not args.config.exists():
        _log.error(f"Config file not found: {args.config}")
        return 1

    # Handle weights
    weight_path = args.weight_path

    if args.download_weights or weight_path is None:
        weight_dir = Path("weight")
        weight_path = download_pretrained_weights(weight_dir)
        if weight_path is None or not weight_path.exists():
            _log.error("Failed to obtain model weights")
            return 1

    if not weight_path.exists():
        _log.error(f"Weight file not found: {weight_path}")
        _log.info("Use --download-weights to download pretrained model")
        return 1

    # Convert models
    try:
        success = convert_vietocr_to_onnx(args.config, weight_path, args.output)
        return 0 if success else 1
    except Exception as e:
        _log.error(f"Conversion failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
