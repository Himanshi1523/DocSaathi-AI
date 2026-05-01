"""
translator.py - Hinglish simplification for DocSaathi AI

Translates English legal/government jargon into simple Hinglish (Hindi + English mix)
that is easy to understand for common Indian users.

Note: This module uses a rule-based glossary approach so it works offline
without any API keys. For production, you could swap in the Google Translate
API or IndicTrans model.
"""

import re
from src.utils import get_logger

logger = get_logger("translator")


# ── Hinglish glossary ──────────────────────────────────────────────────────────
# Format: English term → Hinglish / simplified explanation

HINGLISH_GLOSSARY: dict[str, str] = {
    # Legal terms
    "lessor": "makan malik (landlord)",
    "lessee": "kiraye daar (tenant)",
    "landlord": "makan malik",
    "tenant": "kiraye daar",
    "hereinafter": "aage se",
    "herein": "is agreement mein",
    "hereunder": "is ke antargat",
    "whereas": "jabki",
    "notwithstanding": "iske bawajood",
    "indemnify": "nuksan ki bhari karna",
    "indemnification": "haarz-e-nuqsaan",
    "arbitration": "salah-e-aazaadi se faisle (out of court settlement)",
    "jurisdiction": "court ka adhikar kshetra",
    "force majeure": "asmani aafat (natural disaster / war)",
    "termination": "band karna / khatam karna",
    "breach": "contract todna",
    "default": "payment na karna (default)",
    "collateral": "zamanat / girvi rakhna",
    "guarantor": "zamanat deene waala",
    "lien": "sampatti par adhikar",
    "encumbrance": "sampatti par bojh",
    "sub-let": "aage kiraye par dena",
    "premises": "jagah / property",
    "security deposit": "zamanat rakam",
    "emi": "maasik kist (EMI)",
    "principal amount": "mool rakam",
    "prepayment": "waqt se pehle payment",
    "foreclosure": "loan pura khatam karna (foreclosure)",
    "auto-debit": "bank se automatic paise katna",
    "standing instruction": "bank ko paise katne ki permanent permission",
    "non-refundable": "wapas nahi milega",
    "compounded": "chaqrvrudhi byaj (compound interest)",
    "nominee": "jo paise lega agar aap na hon",
    "affidavit": "shapath patra (sworn statement)",
    "notary": "sarkari manyata prapt sahayak",
    "stamp duty": "sarkari tax on documents",
    "registration": "sarkari record mein daakhil karna",
    "eviction": "ghar khaali karwana",
    "vacate": "khaali karna",

    # Government form terms
    "domicile": "niwas praman patra",
    "bonafide": "asli chhatra praman patra",
    "gazette": "sarkari akhbaar",
    "gazetted officer": "sarkari adhikari",
    "attestation": "sarkari tasdeeq",
    "self-attested": "khud tasdeeq ki gayi copy",
    "enrollment": "naama dakhil / registration",
    "biometric": "ungli ke nishaan aur aankh ki scan",
    "beneficiary": "jise laabh milega",
    "disbursement": "paise dena",
    "stipend": "chhatravrutti / scholarship ki rakam",
    "acknowledgement": "receipt / confirmation",
    "allotment": "assignment / dena",

    # Risk terms
    "late payment fee": "der se dene par ₹ jada dene honge",
    "penalty": "juraana",
    "hidden charges": "chhupi fees",
    "processing fee": "process karne ki fees",
    "service charge": "seva shulk",
    "blacklist": "kredit history kharab ho sakti hai",
    "waiver": "haq chhodna",
    "liability": "zimmedari",
    "unlimited liability": "poori zimmedari aap par",
}


def apply_glossary(text: str) -> str:
    """Replace English jargon with Hinglish equivalents from glossary."""
    for english, hinglish in HINGLISH_GLOSSARY.items():
        pattern = re.compile(re.escape(english), re.IGNORECASE)
        text = pattern.sub(f"{english} [{hinglish}]", text)
    return text


def simplify_sentences(text: str) -> str:
    """
    Apply basic sentence simplification rules for plain language.
    Replaces complex constructs with simpler equivalents.
    """
    replacements = [
        # Passive to active-friendly
        (r"\bshall be liable\b", "zimmedaar hoga"),
        (r"\bshall not\b", "nahi karega"),
        (r"\bshall\b", "karega"),
        (r"\bmust be submitted\b", "jama karna hoga"),
        (r"\bhereby\b", "is dwara"),
        (r"\bpursuant to\b", "ke anusar"),
        (r"\bin accordance with\b", "ke mutabiq"),
        (r"\bwith respect to\b", "ke baare mein"),
        (r"\bin the event of\b", "agar ho"),
        (r"\bprior written consent\b", "pehle likhit anumati"),
        (r"\bpay to the order of\b", "ko payment"),
        (r"\bsubject to\b", "ke adhin"),
        (r"\bupon receipt\b", "milne ke baad"),
        (r"\bforth herein\b", "yahan bataya gaya"),
        (r"\bsupersede\b", "baad wala document lagega"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def translate_to_hinglish(text: str) -> str:
    """
    Convert document text to simplified Hinglish.

    Applies:
      1. Glossary substitutions (legal jargon → Hinglish)
      2. Sentence-level simplifications

    Returns the transformed text.
    """
    if not text or not text.strip():
        return text

    logger.info("Applying Hinglish translation (%d chars)", len(text))
    text = apply_glossary(text)
    text = simplify_sentences(text)
    return text


def translate_list(items: list[str]) -> list[str]:
    """Apply Hinglish translation to a list of strings."""
    return [translate_to_hinglish(item) for item in items]
