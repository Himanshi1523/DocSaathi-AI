"""
ocr_reader.py - OCR support for scanned images in DocSaathi AI
Uses pytesseract + Pillow to extract text from image files.
"""

from pathlib import Path
from typing import Optional
import io

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from src.utils import get_logger, clean_text

logger = get_logger("ocr_reader")


def _check_dependencies() -> None:
    if not PILLOW_AVAILABLE:
        raise RuntimeError("Pillow not installed. Run: pip install Pillow")
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed. Run: pip install pytesseract")


def _preprocess_image(image: "Image.Image") -> "Image.Image":
    """
    Enhance image for better OCR accuracy.
    Steps: grayscale → sharpen → contrast boost → denoise
    """
    # Convert to grayscale
    image = image.convert("L")
    # Sharpen edges
    image = image.filter(ImageFilter.SHARPEN)
    # Boost contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    # Scale up small images (OCR works better on larger images)
    w, h = image.size
    if w < 1000 or h < 1000:
        scale = max(1000 / w, 1000 / h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def extract_text_from_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Run OCR on image bytes and return extracted text.

    Args:
        image_bytes: Raw bytes of the image file.
        lang: Tesseract language code. Use 'hin' for Hindi, 'eng+hin' for both.

    Returns:
        Extracted text string.
    """
    _check_dependencies()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = _preprocess_image(image)
        # Tesseract config: OEM 3 (LSTM), PSM 6 (uniform block of text)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        logger.info("OCR extracted %d characters", len(text))
        return clean_text(text) if text.strip() else ""
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine not found. "
            "Install it from: https://github.com/tesseract-ocr/tesseract"
        )
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc


def extract_text_from_image_file(file_path: str, lang: str = "eng") -> str:
    """Run OCR on a file stored on disk."""
    _check_dependencies()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    with open(path, "rb") as fh:
        return extract_text_from_image_bytes(fh.read(), lang=lang)


def get_ocr_confidence(image_bytes: bytes) -> float:
    """
    Return mean confidence score (0-100) for OCR result.
    Useful for flagging low-quality scans.
    """
    _check_dependencies()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = _preprocess_image(image)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
        return sum(confidences) / len(confidences) if confidences else 0.0
    except Exception:
        return 0.0
