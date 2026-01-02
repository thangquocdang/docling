"""VietOCR ONNX model integration for Docling."""

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import onnxruntime
from docling_core.types.doc import BoundingBox
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel

_log = logging.getLogger(__name__)


class VietOcrOnnxOptions(OcrOptions):
    """Configuration options for VietOCR ONNX model."""

    kind: str = "vietocr_onnx"

    # Model configuration
    cnn_model_path: str = "weight/cnn.onnx"
    encoder_model_path: str = "weight/encoder.onnx"
    decoder_model_path: str = "weight/decoder.onnx"
    vocab_path: str = "config/vocab.txt"

    # Inference parameters
    image_height: int = 32
    image_min_width: int = 32
    image_max_width: int = 512
    max_seq_length: int = 128
    sos_token: int = 1
    eos_token: int = 2

    # ONNX Runtime configuration
    num_threads: int = 2
    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1

    # Supported languages
    lang: List[str] = ["vie"]


class VocabularyOnnx:
    """Vocabulary handler for VietOCR ONNX model."""

    def __init__(self, vocab_path: str):
        """Initialize vocabulary from file."""
        self.char_to_idx = {}
        self.idx_to_char = {}

        if Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                chars = f.read().strip().split('\n')

            for idx, char in enumerate(chars):
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        else:
            _log.warning(f"Vocabulary file not found: {vocab_path}")

    def decode(self, token_indices: List[int]) -> str:
        """Convert token indices to text string."""
        chars = []
        for idx in token_indices:
            if idx == 2:  # EOS token
                break
            if idx == 1:  # SOS token
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)


class VietOcrOnnxModel(BaseOcrModel):
    """VietOCR ONNX model for Vietnamese text recognition in Docling."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: VietOcrOnnxOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initialize VietOCR ONNX model.

        Args:
            enabled: Whether OCR is enabled
            artifacts_path: Path to model artifacts directory
            options: VietOCR ONNX configuration options
            accelerator_options: Accelerator configuration
        """
        super().__init__(enabled, artifacts_path, options, accelerator_options)

        self.options: VietOcrOnnxOptions = options
        self.vocab = None
        self.cnn_session = None
        self.encoder_session = None
        self.decoder_session = None

        if enabled:
            self._initialize_model()

    @classmethod
    def get_options_type(cls):
        """Return the options class for this model."""
        return VietOcrOnnxOptions

    def _initialize_model(self):
        """Load ONNX models and vocabulary."""
        try:
            # Configure ONNX Runtime session options
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = self.options.intra_op_num_threads
            sess_options.inter_op_num_threads = self.options.inter_op_num_threads
            sess_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Determine providers based on accelerator
            providers = ['CPUExecutionProvider']
            if self.accelerator_options.device.value == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')

            # Load ONNX models
            _log.info(f"Loading VietOCR ONNX models from {self.artifacts_path}")

            cnn_path = self._get_model_path(self.options.cnn_model_path)
            encoder_path = self._get_model_path(self.options.encoder_model_path)
            decoder_path = self._get_model_path(self.options.decoder_model_path)

            self.cnn_session = onnxruntime.InferenceSession(
                str(cnn_path), sess_options=sess_options, providers=providers
            )
            self.encoder_session = onnxruntime.InferenceSession(
                str(encoder_path), sess_options=sess_options, providers=providers
            )
            self.decoder_session = onnxruntime.InferenceSession(
                str(decoder_path), sess_options=sess_options, providers=providers
            )

            # Load vocabulary
            vocab_path = self._get_model_path(self.options.vocab_path)
            self.vocab = VocabularyOnnx(str(vocab_path))

            _log.info("VietOCR ONNX models loaded successfully")

        except Exception as e:
            _log.error(f"Failed to initialize VietOCR ONNX model: {e}")
            raise

    def _get_model_path(self, relative_path: str) -> Path:
        """Get absolute path to model file."""
        if self.artifacts_path:
            return self.artifacts_path / relative_path
        return Path(relative_path)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for VietOCR ONNX model.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image as numpy array [B, C, H, W] with C=3 (RGB)

        Note:
            VietOCR ONNX models require RGB input (3 channels), not grayscale.
        """
        # Resize image
        w, h = image.size
        new_h = self.options.image_height
        new_w = int(w * new_h / h)

        # Clamp width to valid range
        new_w = max(self.options.image_min_width,
                   min(new_w, self.options.image_max_width))

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to RGB (VietOCR ONNX requires 3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array and normalize [H, W, C]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Transpose to [C, H, W]
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension [B, C, H, W]
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _translate_onnx(self, img: np.ndarray) -> List[int]:
        """Run ONNX inference to translate image to text tokens.

        Args:
            img: Preprocessed image array

        Returns:
            List of token indices
        """
        # CNN feature extraction
        cnn_input = {self.cnn_session.get_inputs()[0].name: img}
        src = self.cnn_session.run(None, cnn_input)[0]

        # Encoder processing
        encoder_input = {self.encoder_session.get_inputs()[0].name: src}
        encoder_outputs, hidden = self.encoder_session.run(None, encoder_input)

        # Initialize with start token
        batch_size = img.shape[0]
        translated_sentence = [[self.options.sos_token] * batch_size]
        max_length = 0

        # Autoregressive decoding
        while max_length <= self.options.max_seq_length:
            tgt_inp = np.array(translated_sentence[-1], dtype=np.int64)

            decoder_input = {
                self.decoder_session.get_inputs()[0].name: tgt_inp,
                self.decoder_session.get_inputs()[1].name: hidden,
                self.decoder_session.get_inputs()[2].name: encoder_outputs
            }

            output, hidden, _ = self.decoder_session.run(None, decoder_input)

            # Select highest probability token
            next_tokens = np.argmax(output, axis=-1).tolist()
            translated_sentence.append(next_tokens)

            # Check if all sequences have generated EOS token
            if all(self.options.eos_token in translated_sentence[i]
                   for i in range(len(translated_sentence))):
                break

            max_length += 1

        # Transpose to get tokens per batch item
        return np.array(translated_sentence).T[0].tolist()

    def _recognize_text(self, image: Image.Image) -> str:
        """Recognize text from image using VietOCR ONNX.

        Args:
            image: PIL Image containing text

        Returns:
            Recognized text string
        """
        try:
            # Preprocess
            img_array = self._preprocess_image(image)

            # Run inference
            token_indices = self._translate_onnx(img_array)

            # Decode to text
            text = self.vocab.decode(token_indices)

            return text.strip()

        except Exception as e:
            _log.error(f"VietOCR ONNX recognition failed: {e}")
            return ""

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Process pages with VietOCR ONNX OCR.

        Args:
            conv_res: Conversion result containing document data
            page_batch: Batch of pages to process

        Returns:
            Processed pages with OCR results
        """
        if not self.enabled:
            return page_batch

        _log.info("Running VietOCR ONNX OCR on page batch")

        for page in page_batch:
            # Get OCR cells that need text recognition
            ocr_rects = self.get_ocr_rects(page=page)

            if not ocr_rects:
                continue

            _log.debug(f"Processing {len(ocr_rects)} OCR regions on page {page.page_no}")

            # Process each OCR region
            for cell in ocr_rects:
                try:
                    # Extract image region
                    bbox = cell.bbox
                    page_image = page.image

                    # Crop region with some padding
                    x0 = max(0, int(bbox.l) - 2)
                    y0 = max(0, int(bbox.t) - 2)
                    x1 = min(page_image.width, int(bbox.r) + 2)
                    y1 = min(page_image.height, int(bbox.b) + 2)

                    cropped = page_image.crop((x0, y0, x1, y1))

                    # Recognize text
                    text = self._recognize_text(cropped)

                    if text:
                        cell.text = text
                        _log.debug(f"Recognized: '{text}'")

                except Exception as e:
                    _log.warning(f"Failed to process OCR cell: {e}")
                    continue

        return page_batch
