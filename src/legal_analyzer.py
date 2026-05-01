"""
legal_analyzer.py - Legal document analysis for DocSaathi AI

Detects legal document sub-types and produces structured analysis.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.summarizer import extractive_summary, generate_bullet_summary
from src.risk_detector import detect_risks, RiskReport
from src.utils import get_logger

logger = get_logger("legal_analyzer")


# ── Document sub-type detection ────────────────────────────────────────────────

LEGAL_SUBTYPES = {
    "Rent Agreement": [
        r"rent\s*agreement", r"lease\s*deed", r"tenancy", r"landlord", r"tenant",
        r"monthly\s*rent", r"security\s*deposit", r"premises"
    ],
    "Loan Agreement": [
        r"loan\s*agreement", r"borrower", r"lender", r"principal\s*amount",
        r"repayment", r"emi", r"interest\s*rate", r"disbursement"
    ],
    "Employment Contract": [
        r"employment\s*(contract|agreement)", r"employee", r"employer",
        r"salary", r"designation", r"probation", r"notice\s*period"
    ],
    "Legal Notice": [
        r"legal\s*notice", r"notice\s*is\s*hereby", r"demand\s*notice",
        r"advocate|attorney", r"cause\s*of\s*action"
    ],
    "Service Contract": [
        r"service\s*(agreement|contract)", r"service\s*provider", r"client",
        r"scope\s*of\s*work", r"deliverables", r"milestones"
    ],
    "Non-Disclosure Agreement": [
        r"non[\s-]?disclosure", r"nda", r"confidential(ity)?",
        r"proprietary\s*information", r"trade\s*secret"
    ],
    "Sale Deed": [
        r"sale\s*deed", r"vendor", r"purchaser", r"conveyance",
        r"immovable\s*property", r"registration"
    ],
}


def detect_legal_subtype(text: str) -> str:
    """Identify the most likely legal document sub-type."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for subtype, patterns in LEGAL_SUBTYPES.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[subtype] = score
    if not scores:
        return "General Legal Document"
    return max(scores, key=scores.get)


# ── Key clause extraction ──────────────────────────────────────────────────────

def extract_dates(text: str) -> list[str]:
    """Extract dates and time references from legal text."""
    patterns = [
        r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{4}\b",
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{4}\b",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))
    return list(dict.fromkeys(found))[:10]


def extract_monetary_values(text: str) -> list[str]:
    """Extract monetary amounts from legal text."""
    pattern = r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{1,2})?\s*(?:rupees?|lakh[s]?|crore[s]?)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(dict.fromkeys(matches))[:10]


def extract_parties(text: str) -> list[str]:
    """Try to extract party names from the document opening."""
    # Look for "between X and Y" pattern
    match = re.search(
        r"between\s+([A-Z][A-Za-z\s.]+?)\s+(?:\(hereinafter|and)\s+([A-Z][A-Za-z\s.]+?)[\s,]",
        text
    )
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    return []


# ── Main analysis function ─────────────────────────────────────────────────────

@dataclass
class LegalAnalysis:
    doc_type: str = "Legal Document"
    subtype: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    parties: List[str] = field(default_factory=list)
    important_dates: List[str] = field(default_factory=list)
    monetary_values: List[str] = field(default_factory=list)
    risk_report: Optional[RiskReport] = None


def analyze_legal_document(text: str) -> LegalAnalysis:
    """
    Full analysis pipeline for a legal document.

    Returns a LegalAnalysis dataclass with all findings.
    """
    logger.info("Starting legal document analysis (%d chars)", len(text))

    analysis = LegalAnalysis()
    analysis.doc_type = "Legal Document"
    analysis.subtype = detect_legal_subtype(text)

    # Summarise
    analysis.summary = extractive_summary(text, num_sentences=4)
    analysis.key_points = generate_bullet_summary(text, num_points=6)

    # Extract metadata
    analysis.parties = extract_parties(text)
    analysis.important_dates = extract_dates(text)
    analysis.monetary_values = extract_monetary_values(text)

    # Risk analysis
    analysis.risk_report = detect_risks(text)

    logger.info("Legal analysis complete: subtype=%s, risk=%s",
                analysis.subtype, analysis.risk_report.risk_level)
    return analysis
