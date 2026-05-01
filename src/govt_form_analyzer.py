"""
govt_form_analyzer.py - Government form analysis for DocSaathi AI

Detects government form types and produces structured guidance.
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.summarizer import extractive_summary, generate_bullet_summary
from src.utils import get_logger

logger = get_logger("govt_form_analyzer")


# ── Form sub-type definitions ──────────────────────────────────────────────────

FORM_DEFINITIONS = {
    "Scholarship Form": {
        "patterns": [
            r"scholarship", r"stipend", r"fellowship", r"nsp|national\s*scholarship",
            r"merit.cum.means", r"post[\s-]?matric", r"pre[\s-]?matric"
        ],
        "purpose": (
            "This form is used to apply for a financial scholarship to support your "
            "education. It is submitted to the relevant government portal or institution."
        ),
        "required_docs": [
            "Aadhaar Card (mandatory for identity verification)",
            "Income Certificate issued by a gazetted officer",
            "Previous year marksheet / passing certificate",
            "Bonafide / enrollment certificate from school or college",
            "Bank account passbook (linked to Aadhaar)",
            "Caste certificate (if applying under SC/ST/OBC category)",
            "Passport-size photograph",
        ],
        "deadlines": [
            "Application window typically opens in August–October each year",
            "Last date for institute verification is usually November",
            "Check NSP portal (scholarships.gov.in) for exact dates",
        ],
        "filling_guide": [
            "Register on the NSP portal (scholarships.gov.in) with your Aadhaar number",
            "Select the correct scholarship scheme for your category and level of study",
            "Fill personal details exactly as they appear on Aadhaar",
            "Enter bank account details – must be in your own name and Aadhaar-linked",
            "Upload all required documents in prescribed format (PDF / JPG, max 200 KB)",
            "Submit and note the application reference number",
            "Get the application verified by your institute within the deadline",
        ],
    },
    "PAN Card Form": {
        "patterns": [
            r"pan\s*card", r"form\s*49a", r"permanent\s*account\s*number",
            r"income\s*tax.*pan", r"nsdl|utiitsl"
        ],
        "purpose": (
            "PAN (Permanent Account Number) is a unique 10-digit alphanumeric identifier "
            "issued by the Income Tax Department. Required for financial transactions, "
            "filing taxes, and opening bank accounts."
        ),
        "required_docs": [
            "Identity proof (Aadhaar / Passport / Voter ID / Driving Licence)",
            "Address proof (Aadhaar / Utility bill / Bank statement)",
            "Date of birth proof (Birth certificate / Matriculation certificate)",
            "2 recent passport-size photographs",
            "Application fee (currently ₹107 for Indian address, ₹1017 for foreign)",
        ],
        "deadlines": [
            "No specific deadline – PAN can be applied for at any time",
            "PAN is usually allotted within 15 working days of submission",
        ],
        "filling_guide": [
            "Apply online at www.onlineservices.nsdl.com or www.utiitsl.com",
            "Select Form 49A for Indian citizens",
            "Enter personal details exactly matching your identity documents",
            "Upload scanned copies of proof documents",
            "Pay the application fee online",
            "Note the 15-digit acknowledgement number for tracking",
        ],
    },
    "Pension Form": {
        "patterns": [
            r"pension\s*form", r"retirement\s*benefit", r"superannuation",
            r"epfo|employee\s*provident", r"ppf|public\s*provident",
            r"old\s*age\s*pension", r"widow\s*pension", r"family\s*pension"
        ],
        "purpose": (
            "This form is related to pension benefits for retired government employees, "
            "senior citizens, or beneficiaries under social security schemes."
        ),
        "required_docs": [
            "Service / Retirement certificate (for government employees)",
            "Aadhaar Card",
            "Bank passbook / cancelled cheque (for pension credit)",
            "Age proof (Birth certificate / Aadhaar)",
            "Photograph (passport size)",
            "Nomination form for family pension",
            "Disability certificate (if applicable)",
        ],
        "deadlines": [
            "Pension application must be submitted 6 months before retirement",
            "Old-age pension applications can be submitted any time after age 60",
            "Check state government portal for specific deadlines",
        ],
        "filling_guide": [
            "Obtain the correct pension form from your department / district office",
            "Fill personal and service details in block letters",
            "Attach all required documents attested by a gazetted officer",
            "Submit to your Head of Department (for service pension) or district welfare office",
            "Keep acknowledgement receipt for follow-up",
        ],
    },
    "Aadhaar Form": {
        "patterns": [
            r"aadhaar|aadhar|uid(ai)?", r"biometric\s*enrollment",
            r"address\s*update.*aadhaar", r"name\s*change.*aadhaar"
        ],
        "purpose": (
            "Aadhaar is a 12-digit unique identity number issued by UIDAI. "
            "This form is for new enrollment, or for updating details like "
            "name, address, date of birth, or mobile number."
        ),
        "required_docs": [
            "Proof of Identity (Passport / PAN / Voter ID / Driving Licence / NREGA job card)",
            "Proof of Address (Utility bill / Bank statement / Ration card – not older than 3 months)",
            "Date of Birth proof (for corrections only)",
            "Existing Aadhaar number (for update requests)",
        ],
        "deadlines": [
            "No specific deadline for new enrollment",
            "Address update requests are processed within 90 days",
        ],
        "filling_guide": [
            "Visit the nearest Aadhaar Seva Kendra or book an appointment at appointments.uidai.gov.in",
            "Fill the Aadhaar enrollment / update form at the centre",
            "Provide biometric data (fingerprints + iris scan) for new enrollment",
            "For online updates (address / mobile): visit ssup.uidai.gov.in",
            "Keep the acknowledgement slip with 14-digit EID for tracking",
        ],
    },
    "Job Application Form": {
        "patterns": [
            r"job\s*application", r"employment\s*application", r"recruitment",
            r"upsc|ssc|ibps|rrb|nra\s*cet", r"vacancy|post\s*applied"
        ],
        "purpose": (
            "This is a government job application form for a recruitment drive. "
            "It must be filled accurately to avoid disqualification."
        ),
        "required_docs": [
            "Educational certificates (10th, 12th, Graduation / PG)",
            "Aadhaar Card / Identity proof",
            "Date of Birth certificate",
            "Caste certificate (if applying under reserved category)",
            "Experience letter (if required for the post)",
            "Passport-size photographs as specified",
            "Application fee payment receipt",
        ],
        "deadlines": [
            "Closing date is mentioned in the official notification – do not miss it",
            "Fee payment deadline may be 1-2 days before form submission closes",
            "Hall ticket / admit card issued 2-3 weeks before exam",
        ],
        "filling_guide": [
            "Read the official notification carefully for eligibility criteria",
            "Register on the official recruitment portal",
            "Fill personal, educational, and category details accurately",
            "Upload photograph and signature in the exact size and format specified",
            "Pay the application fee through online banking / debit card / UPI",
            "Submit and download the confirmation page / printout",
        ],
    },
}

# Default for unrecognised government forms
_DEFAULT_FORM = {
    "purpose": "This appears to be a government form. Please read all instructions carefully.",
    "required_docs": ["Identity proof (Aadhaar / PAN / Passport)", "Address proof", "Relevant certificates"],
    "deadlines": ["Check the form header or official notification for deadlines"],
    "filling_guide": [
        "Read instructions on the form carefully",
        "Fill in block letters in English or Hindi as instructed",
        "Attach self-attested copies of all required documents",
        "Submit before the deadline",
    ],
}


def detect_form_subtype(text: str) -> str:
    """Identify the most likely government form type."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for form_type, defn in FORM_DEFINITIONS.items():
        score = sum(1 for p in defn["patterns"] if re.search(p, text_lower))
        if score > 0:
            scores[form_type] = score
    if not scores:
        return "Government Form"
    return max(scores, key=scores.get)


def extract_deadlines_from_text(text: str) -> list[str]:
    """Extract deadline-related sentences from form text."""
    deadline_pattern = re.compile(
        r"(?:last\s*date|deadline|due\s*date|submit.*by|before|not\s*later\s*than)[^.!?\n]{0,100}",
        re.IGNORECASE
    )
    return [m.group().strip() for m in deadline_pattern.finditer(text)][:5]


def extract_missing_field_warnings(text: str) -> list[str]:
    """Flag potentially incomplete fields in the text."""
    warnings = []
    blank_patterns = [
        (r"\b_+\b", "Blank fields detected – may indicate missing information"),
        (r"\[ *\]", "Unchecked checkboxes found"),
        (r"N/?A\b", "Fields marked N/A – verify if this is correct"),
        (r"TBD|to be (decided|filled|confirmed)", "Placeholder text (TBD) found"),
    ]
    for pat, warning in blank_patterns:
        if re.search(pat, text, re.IGNORECASE):
            warnings.append(warning)
    return warnings


@dataclass
class GovtFormAnalysis:
    doc_type: str = "Government Form"
    subtype: str = ""
    purpose: str = ""
    summary: str = ""
    required_docs: List[str] = field(default_factory=list)
    deadlines: List[str] = field(default_factory=list)
    filling_guide: List[str] = field(default_factory=list)
    missing_field_warnings: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)


def analyze_govt_form(text: str) -> GovtFormAnalysis:
    """
    Full analysis pipeline for a government form.
    Returns a GovtFormAnalysis dataclass.
    """
    logger.info("Starting government form analysis (%d chars)", len(text))

    analysis = GovtFormAnalysis()
    analysis.doc_type = "Government Form"
    analysis.subtype = detect_form_subtype(text)

    defn = FORM_DEFINITIONS.get(analysis.subtype, _DEFAULT_FORM)
    analysis.purpose = defn["purpose"]
    analysis.required_docs = defn["required_docs"]

    # Combine known + extracted deadlines
    known_deadlines = defn.get("deadlines", [])
    extracted_deadlines = extract_deadlines_from_text(text)
    analysis.deadlines = list(dict.fromkeys(extracted_deadlines + known_deadlines))

    analysis.filling_guide = defn.get("filling_guide", _DEFAULT_FORM["filling_guide"])
    analysis.missing_field_warnings = extract_missing_field_warnings(text)
    analysis.summary = extractive_summary(text, num_sentences=3)
    analysis.key_points = generate_bullet_summary(text, num_points=4)

    logger.info("Govt form analysis complete: subtype=%s", analysis.subtype)
    return analysis
