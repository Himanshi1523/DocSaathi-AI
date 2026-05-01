"""
utils.py - Utility functions for DocSaathi AI
Common helpers used across the project.
"""

import os
import re
import logging
from datetime import datetime

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DocSaathi")


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(f"DocSaathi.{name}")


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excess whitespace and control characters from text."""
    if not text:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[truncated]"


def split_into_sentences(text: str) -> list[str]:
    """Naively split text into sentences on '. ', '! ', '? '."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def word_count(text: str) -> int:
    """Return word count of text."""
    return len(text.split())


# ── File helpers ───────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot."""
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".")


def is_pdf(filename: str) -> bool:
    return get_file_extension(filename) == "pdf"


def is_image(filename: str) -> bool:
    return get_file_extension(filename) in {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}


def is_text(filename: str) -> bool:
    return get_file_extension(filename) in {"txt", "text"}


def save_upload(file_bytes: bytes, filename: str, upload_dir: str = "uploads") -> str:
    """Save uploaded file bytes to disk; return the full path."""
    ensure_dir(upload_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w.\-]", "_", filename)
    dest = os.path.join(upload_dir, f"{timestamp}_{safe_name}")
    with open(dest, "wb") as fh:
        fh.write(file_bytes)
    logger.info("Saved upload: %s", dest)
    return dest


# ── Formatting helpers ─────────────────────────────────────────────────────────

def bullet_list(items: list[str]) -> str:
    """Convert a list to a markdown bullet list string."""
    return "\n".join(f"• {item}" for item in items)


def risk_emoji(level: str) -> str:
    """Return emoji for risk level."""
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(level, "⚪")


def format_section(title: str, content: str) -> str:
    """Format a titled section for display."""
    return f"### {title}\n{content}\n"
