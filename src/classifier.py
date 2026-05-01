"""
classifier.py - Document type classifier for DocSaathi AI

Trains a TF-IDF + Logistic Regression model to classify text as:
  • legal
  • government_form

The model is saved to models/classifier.joblib and auto-loaded on subsequent runs.
"""

import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocess import preprocess_for_ml
from src.utils import get_logger, ensure_dir

logger = get_logger("classifier")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "classifier.joblib"
DATA_PATH = Path("data") / "training_data.csv"

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL_NAMES = {
    "legal": "Legal Document",
    "government_form": "Government Form",
}


def _build_pipeline(use_lr: bool = True) -> Pipeline:
    """Create sklearn Pipeline with TF-IDF + classifier."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )
    clf = LogisticRegression(max_iter=1000, C=5.0) if use_lr else MultinomialNB()
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_model(data_path: str = str(DATA_PATH)) -> Pipeline:
    """
    Train the document classifier from the CSV dataset.
    Saves the trained pipeline to MODEL_PATH.
    Returns the fitted pipeline.
    """
    logger.info("Loading training data from %s", data_path)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"])
    df["text_clean"] = df["text"].apply(preprocess_for_ml)

    X, y = df["text_clean"].tolist(), df["label"].tolist()
    logger.info("Training on %d samples with labels: %s", len(X), set(y))

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline(use_lr=True)
    pipeline.fit(X_train, y_train)

    # Log evaluation
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("Classification Report:\n%s", report)

    # Persist model
    ensure_dir(str(MODEL_DIR))
    joblib.dump(pipeline, str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)

    return pipeline


def load_or_train() -> Pipeline:
    """Load saved model if it exists, otherwise train a new one."""
    if MODEL_PATH.exists():
        logger.info("Loading existing model from %s", MODEL_PATH)
        try:
            return joblib.load(str(MODEL_PATH))
        except Exception as exc:
            logger.warning("Failed to load model (%s); retraining.", exc)

    logger.info("No saved model found – training now …")
    return train_model()


def classify_text(text: str, pipeline: Pipeline = None) -> Tuple[str, float]:
    """
    Classify a document text.

    Args:
        text: Raw document text.
        pipeline: Pre-loaded sklearn Pipeline. Auto-loads if None.

    Returns:
        Tuple of (label, confidence) e.g. ('legal', 0.92)
    """
    if not text or not text.strip():
        return "unknown", 0.0

    if pipeline is None:
        pipeline = load_or_train()

    cleaned = preprocess_for_ml(text)
    label = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    confidence = float(max(proba))

    logger.info("Classified as '%s' with confidence %.2f", label, confidence)
    return label, confidence


def get_label_display(label: str) -> str:
    """Return human-friendly label name."""
    return LABEL_NAMES.get(label, label.replace("_", " ").title())
