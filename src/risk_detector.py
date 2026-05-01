"""
risk_detector.py - Risk analysis for legal documents in DocSaathi AI

Detects risky clauses, hidden charges, and computes an overall risk score.
Uses keyword/regex pattern matching — no external API required.
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.utils import get_logger

logger = get_logger("risk_detector")


# ── Risk pattern definitions ───────────────────────────────────────────────────

# Each entry: (pattern_regex, human_readable_warning, risk_weight 1-3)
RISK_PATTERNS = [
    # Financial risks
    (r"late\s*(payment)?\s*fee[s]?\s*(of|:)?\s*[\₹$]?\s*\d+", "Late payment fee detected", 2),
    (r"penalty\s*(of|:)?\s*[\₹$]?\s*\d+", "Penalty clause found", 2),
    (r"auto[\s-]?debit|automatic\s*deduction|standing\s*instruction", "Auto-debit / automatic deduction enabled", 2),
    (r"hidden\s*charge|processing\s*fee|service\s*charge", "Hidden charges or processing fees found", 3),
    (r"interest\s*rate.*?(\d{1,2})\s*%", "Interest rate clause detected", 1),
    (r"compounded?\s*(monthly|daily|annually)", "Compound interest clause found", 2),
    (r"foreclosure\s*(charge|penalty|fee)", "Foreclosure/prepayment charge found", 2),

    # Termination and eviction
    (r"terminate.*?without\s*(notice|reason)", "Termination without notice clause", 3),
    (r"evict(ion)?|vacate.*?(\d+)\s*days?", "Eviction / vacate notice clause", 2),
    (r"terminate.*?(\d+)\s*days?\s*notice", "Termination notice period clause", 1),
    (r"blacklist|credit\s*(bureau|score|report)", "Credit blacklisting / bureau reporting clause", 3),

    # Liability and indemnity
    (r"indemnif(y|ication)|hold\s*harmless", "Indemnification / hold harmless clause", 2),
    (r"unlimited\s*liability", "Unlimited liability clause – very risky", 3),
    (r"waive[rs]?\s*(all\s*)?(rights?|claims?)", "Rights waiver clause", 3),
    (r"non[\s-]?refundable", "Non-refundable payment clause", 2),

    # Arbitration and jurisdiction
    (r"arbitration\s*clause|binding\s*arbitration", "Mandatory arbitration clause (limits court access)", 2),
    (r"exclusive\s*jurisdiction", "Exclusive jurisdiction clause", 1),
    (r"governing\s*law.*?foreign|international\s*law", "Foreign governing law clause", 2),

    # Data and privacy
    (r"share.*?personal\s*data|sell.*?data|third[\s-]?party.*?data", "Personal data sharing with third parties", 3),
    (r"surveillance|monitor(ing)?\s*(calls|activity|usage)", "Surveillance / activity monitoring clause", 2),

    # Auto-renewal
    (r"auto[\s-]?renew(al|s)?|automatically\s*renew", "Auto-renewal clause detected", 2),
    (r"lock[\s-]?in\s*period|minimum\s*contract\s*period", "Lock-in period clause", 1),

    # Guarantor
    (r"personal\s*guarantee|guarantor\s*shall\s*be\s*liable", "Personal guarantee / guarantor liability clause", 3),
    (r"post[\s-]?dated\s*cheque[s]?|PDC", "Post-dated cheques required as security", 1),
]


@dataclass
class RiskReport:
    warnings: List[str] = field(default_factory=list)
    risky_clauses: List[str] = field(default_factory=list)
    hidden_charges: List[str] = field(default_factory=list)
    renewal_alerts: List[str] = field(default_factory=list)
    risk_score: int = 0          # 0-100
    risk_level: str = "Low"      # Low / Medium / High


def _extract_matching_sentences(text: str, pattern: str) -> list[str]:
    """Return sentences from text that match a given regex pattern."""
    results = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    regex = re.compile(pattern, re.IGNORECASE)
    for sentence in sentences:
        if regex.search(sentence):
            results.append(sentence.strip()[:200])  # cap at 200 chars
    return results[:3]  # return at most 3 matching sentences


def detect_risks(text: str) -> RiskReport:
    """
    Scan document text for risk patterns.

    Returns a RiskReport with categorised findings and an overall risk score.
    """
    report = RiskReport()
    total_weight = 0

    for pattern, warning, weight in RISK_PATTERNS:
        matches = _extract_matching_sentences(text, pattern)
        if matches:
            report.warnings.append(warning)
            report.risky_clauses.extend(matches)

            # Categorise by type
            if re.search(r"fee|charge|debit|penalty|interest|foreclosure", warning, re.I):
                report.hidden_charges.append(warning)
            if re.search(r"renew|lock[\s-]?in", warning, re.I):
                report.renewal_alerts.append(warning)

            total_weight += weight

    # Compute risk score (0-100)
    max_possible = sum(w for _, _, w in RISK_PATTERNS)
    report.risk_score = min(100, int((total_weight / max_possible) * 100))

    # Determine risk level
    if report.risk_score < 20:
        report.risk_level = "Low"
    elif report.risk_score < 50:
        report.risk_level = "Medium"
    else:
        report.risk_level = "High"

    # Deduplicate
    report.warnings = list(dict.fromkeys(report.warnings))
    report.risky_clauses = list(dict.fromkeys(report.risky_clauses))
    report.hidden_charges = list(dict.fromkeys(report.hidden_charges))
    report.renewal_alerts = list(dict.fromkeys(report.renewal_alerts))

    logger.info(
        "Risk scan complete: %d warnings, score=%d, level=%s",
        len(report.warnings), report.risk_score, report.risk_level,
    )
    return report
