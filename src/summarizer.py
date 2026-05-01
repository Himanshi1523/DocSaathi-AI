"""
summarizer.py - Text summarisation for DocSaathi AI

Uses extractive summarisation (TF-IDF sentence scoring) as the default
so there are zero heavy dependencies. If the `transformers` library is
available and the user opts in, an abstractive model can be swapped in.
"""

import re
import math
from collections import Counter
from typing import List

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

from src.preprocess import remove_stopwords, to_lowercase, tokenize
from src.utils import get_logger, clean_text

logger = get_logger("summarizer")


# ── Extractive summariser (TF-IDF sentence scoring) ──────────────────────────

def _compute_word_frequencies(text: str) -> dict[str, float]:
    """Compute normalised word frequencies (stopwords excluded)."""
    tokens = tokenize(to_lowercase(text))
    tokens = remove_stopwords(tokens)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    freq = Counter(tokens)
    max_freq = max(freq.values()) if freq else 1
    return {word: count / max_freq for word, count in freq.items()}


def _score_sentences(sentences: list[str], word_freq: dict[str, float]) -> dict[int, float]:
    """Score each sentence by sum of normalised word frequencies."""
    scores: dict[int, float] = {}
    for idx, sentence in enumerate(sentences):
        tokens = tokenize(to_lowercase(sentence))
        score = sum(word_freq.get(t, 0) for t in tokens)
        # Penalise very short or very long sentences
        word_count = len(tokens)
        if word_count < 5:
            score *= 0.5
        elif word_count > 50:
            score *= 0.8
        scores[idx] = score
    return scores


def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """
    Generate an extractive summary by selecting top-scored sentences.

    Args:
        text: Input document text.
        num_sentences: Number of sentences to include in the summary.

    Returns:
        Summary as a string.
    """
    if not text or not text.strip():
        return "No text available for summarisation."

    sentences = sent_tokenize(clean_text(text))

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    word_freq = _compute_word_frequencies(text)
    scores = _score_sentences(sentences, word_freq)

    # Pick top-N sentence indices in document order
    top_indices = sorted(
        sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    )
    summary = " ".join(sentences[i] for i in top_indices)
    logger.info("Summary generated: %d → %d sentences", len(sentences), num_sentences)
    return summary


def generate_bullet_summary(text: str, num_points: int = 5) -> list[str]:
    """
    Return a list of bullet-point strings summarising the document.
    """
    sentences = sent_tokenize(clean_text(text))
    if not sentences:
        return ["No content found."]

    word_freq = _compute_word_frequencies(text)
    scores = _score_sentences(sentences, word_freq)
    top_indices = sorted(
        sorted(scores, key=scores.get, reverse=True)[:num_points]
    )
    return [sentences[i].strip() for i in top_indices]


def get_document_stats(text: str) -> dict:
    """Return basic document statistics."""
    sentences = sent_tokenize(text) if text else []
    words = text.split() if text else []
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "char_count": len(text),
        "avg_sentence_length": round(len(words) / len(sentences), 1) if sentences else 0,
    }
