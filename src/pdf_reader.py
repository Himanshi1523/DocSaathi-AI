"""
pdf_reader.py - PDF text extraction for DocSaathi AI
Uses pdfplumber for reliable text + table extraction.
"""

from pathlib import Path
from typing import Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from src.utils import get_logger, clean_text

logger = get_logger("pdf_reader")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns concatenated text from every page.
    Raises FileNotFoundError if the file doesn't exist.
    Raises RuntimeError if pdfplumber is not installed.
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    all_text: list[str] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            logger.info("Opened PDF: %s  (%d pages)", path.name, total_pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        all_text.append(f"[Page {page_num}]\n{page_text}")
                    else:
                        logger.debug("Page %d has no extractable text", page_num)
                except Exception as exc:
                    logger.warning("Could not extract page %d: %s", page_num, exc)

    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", file_path, exc)
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    combined = "\n\n".join(all_text)
    logger.info("Extracted %d characters from PDF", len(combined))
    return clean_text(combined) if combined else ""


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract text from raw PDF bytes (e.g. from Streamlit UploadedFile).
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("pdfplumber is not installed.")

    import io
    all_text: list[str] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            logger.info("Opened PDF from bytes (%d pages)", len(pdf.pages))
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        all_text.append(f"[Page {page_num}]\n{page_text}")
                except Exception as exc:
                    logger.warning("Page %d error: %s", page_num, exc)
    except Exception as exc:
        raise RuntimeError(f"PDF bytes extraction failed: {exc}") from exc

    combined = "\n\n".join(all_text)
    return clean_text(combined) if combined else ""


def get_pdf_metadata(file_bytes: bytes) -> dict:
    """Return basic PDF metadata (page count, author, etc.)."""
    if not PDF_AVAILABLE:
        return {}
    import io
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            meta = pdf.metadata or {}
            meta["page_count"] = len(pdf.pages)
            return meta
    except Exception as exc:
        logger.warning("Could not read PDF metadata: %s", exc)
        return {}
