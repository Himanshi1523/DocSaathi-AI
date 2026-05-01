"""
preprocess.py - Text preprocessing pipeline for DocSaathi AI
Handles cleaning, tokenisation, stopword removal and lemmatisation.
"""

import re
import string

import nltk

# Download required NLTK resources on first run
for resource in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from src.utils import get_logger, clean_text

logger = get_logger("preprocess")

# Build English stopword set (keep legal/form-relevant words)
_STOP = set(stopwords.words("english")) - {
    "not", "no", "nor", "against", "before", "after",
    "between", "above", "below", "from", "until", "while",
}
_LEMMATIZER = WordNetLemmatizer()


def remove_special_characters(text: str) -> str:
    """Strip punctuation and non-alphanumeric chars (preserve spaces)."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def preprocess_for_ml(text: str) -> str:
    """
    Full preprocessing pipeline for ML model input.
    Returns a single cleaned string ready for TF-IDF vectorisation.
    """
    if not text:
        return ""
    text = clean_text(text)
    text = to_lowercase(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and len(t) > 1]
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    result = " ".join(tokens)
    logger.debug("Preprocessed text length: %d chars", len(result))
    return result


def preprocess_for_display(text: str) -> str:
    """
    Light preprocessing that retains readability for display purposes.
    Cleans whitespace / encoding artefacts only.
    """
    if not text:
        return ""
    # Fix common OCR/encoding artefacts
    text = text.replace("\x0c", "\n")          # form-feed
    text = re.sub(r"[ \t]{2,}", " ", text)     # multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # triple newlines
    text = clean_text(text)
    return text


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    try:
        return sent_tokenize(text)
    except Exception:
        return text.split(". ")


def extract_key_phrases(text: str, top_n: int = 10) -> list[str]:
    """
    Simple keyword extraction based on frequency after stopword removal.
    Returns top_n most frequent non-stopword tokens.
    """
    tokens = tokenize(to_lowercase(text))
    tokens = remove_stopwords(tokens)
    tokens = [t for t in tokens if len(t) > 3 and t.isalpha()]
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_tokens[:top_n]]
