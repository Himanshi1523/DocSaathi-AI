"""
app.py - DocSaathi AI  |  Main Streamlit Application
=====================================================
Intelligent document simplification for legal documents and government forms.
Run with:  streamlit run app.py
"""

import os
import sys
import time
import traceback

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="DocSaathi AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.pdf_reader import extract_text_from_pdf_bytes
from src.ocr_reader import extract_text_from_image_bytes
from src.classifier import load_or_train, classify_text, get_label_display
from src.legal_analyzer import analyze_legal_document
from src.govt_form_analyzer import analyze_govt_form
from src.translator import translate_to_hinglish, translate_list
from src.utils import get_file_extension, is_pdf, is_image, risk_emoji

# ═══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;700;800&family=Nunito:wght@400;500;600;700&display=swap');

  /* ── Root tokens ── */
  :root {
    --primary:    #1a3c6e;
    --accent:     #f5a623;
    --accent2:    #e84c3d;
    --success:    #27ae60;
    --warn:       #f39c12;
    --danger:     #c0392b;
    --bg:         #f4f6fb;
    --card-bg:    #ffffff;
    --border:     #dce3f0;
    --text:       #1c2a3a;
    --muted:      #6b7a96;
    --radius:     14px;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: var(--text);
    background: var(--bg);
  }

  /* ── Header ── */
  .doc-header {
    background: linear-gradient(135deg, #1a3c6e 0%, #0d2347 100%);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 8px 32px rgba(26,60,110,0.18);
  }
  .doc-header .logo { font-size: 3rem; }
  .doc-header h1 {
    font-family: 'Baloo 2', cursive;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .doc-header p { margin: 0; opacity: 0.82; font-size: 1rem; }

  /* ── Cards ── */
  .result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
  }
  .result-card h3 {
    font-family: 'Baloo 2', cursive;
    color: var(--primary);
    margin-top: 0;
    font-size: 1.15rem;
  }

  /* ── Risk meter ── */
  .risk-meter {
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .risk-low    { background: #e8f8ef; color: #1a6e3c; border-left: 5px solid #27ae60; }
  .risk-medium { background: #fef9e7; color: #7d6008; border-left: 5px solid #f39c12; }
  .risk-high   { background: #fdecea; color: #78281f; border-left: 5px solid #c0392b; }

  /* ── Badge ── */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .badge-legal { background: #dce8fb; color: #1a3c6e; }
  .badge-govt  { background: #fde8b0; color: #7d4e00; }

  /* ── Warning list ── */
  .warn-item {
    padding: 0.5rem 0.75rem;
    border-left: 3px solid var(--warn);
    background: #fffbf0;
    border-radius: 6px;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
  }
  .danger-item {
    padding: 0.5rem 0.75rem;
    border-left: 3px solid var(--accent2);
    background: #fff5f5;
    border-radius: 6px;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
  }

  /* ── Step guide ── */
  .step-item {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: #f0f4ff;
    margin-bottom: 0.4rem;
    border-left: 4px solid var(--primary);
    font-size: 0.9rem;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    background: #f8faff !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d2347;
    color: white;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown p { color: #cdd9f0 !important; }

  /* ── Divider ── */
  hr { border-color: var(--border); opacity: 0.5; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached model loader
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️ Loading AI model…")
def get_model():
    """Load (or train) the classifier once per session."""
    return load_or_train()


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚖️ DocSaathi AI")
    st.markdown("---")
    hinglish_mode = st.toggle("🇮🇳 Hinglish Mode", value=False)
    st.markdown("---")
    st.markdown("### 📄 Supported Formats")
    st.markdown("""
- 📕 PDF files
- 🖼️ Scanned images (PNG / JPG)
- 📝 Text files (.txt)
    """)
    st.markdown("---")
    st.markdown("### 📋 Analyzes")
    st.markdown("""
**Legal Documents**
- Rent / Lease agreements
- Loan agreements
- Legal notices
- Employment contracts
- NDAs, Sale deeds

**Government Forms**
- Scholarship forms
- PAN card forms
- Pension forms
- Aadhaar forms
- Job application forms
    """)
    st.markdown("---")
    st.caption("DocSaathi AI v1.0  •  Built with ❤️ in India")


# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="doc-header">
  <div class="logo">⚖️</div>
  <div>
    <h1>DocSaathi AI</h1>
    <p>Your intelligent document companion — simplifying legal & government documents in seconds</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Text extraction helper
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(uploaded_file) -> str:
    """
    Extract text from an uploaded file.
    Supports PDF, images (via OCR), and plain text.
    """
    ext = get_file_extension(uploaded_file.name)
    file_bytes = uploaded_file.read()

    if ext == "pdf":
        return extract_text_from_pdf_bytes(file_bytes)
    elif ext in {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}:
        return extract_text_from_image_bytes(file_bytes)
    elif ext in {"txt", "text"}:
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Upload Section
# ═══════════════════════════════════════════════════════════════════════════════

col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.subheader("📤 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "txt"],
        help="Supported: PDF, scanned images (PNG/JPG/TIFF), plain text files",
    )

    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully  "
                   f"({uploaded_file.size / 1024:.1f} KB)")

with col_info:
    st.subheader("💡 How It Works")
    st.markdown("""
    1. **Upload** your document (PDF, image, or text)
    2. **AI analyses** the content automatically
    3. **Get results** — summary, risks, guidance
    4. Toggle **Hinglish mode** for Hindi-English output
    """)

st.markdown("---")

# ── Analyse button ─────────────────────────────────────────────────────────────

if uploaded_file:
    analyze_btn = st.button(
        "🔍 Analyse Document",
        type="primary",
        use_container_width=True,
    )
else:
    st.info("📎 Please upload a document to get started.")
    analyze_btn = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if analyze_btn and uploaded_file:

    with st.spinner("🧠 Reading and analysing your document…"):
        try:
            # 1. Extract text
            progress = st.progress(0, text="Extracting text…")
            raw_text = extract_text(uploaded_file)
            progress.progress(25, text="Text extracted…")

            if not raw_text.strip():
                st.error("❌ Could not extract any text from the document. "
                         "If it's a scanned image, ensure Tesseract OCR is installed.")
                st.stop()

            # 2. Classify document
            progress.progress(40, text="Classifying document type…")
            pipeline = get_model()
            label, confidence = classify_text(raw_text, pipeline)
            st.write("Detected Label:", label)
            progress.progress(60, text="Running analysis…")

            # 3. Analyse
            if label == "legal":
                analysis = analyze_legal_document(raw_text)
            else:
                analysis = analyze_govt_form(raw_text)

            progress.progress(85, text="Applying Hinglish mode…" if hinglish_mode else "Finalising…")

            # 4. Apply Hinglish translation if enabled
            if hinglish_mode:
                              
                analysis.summary = translate_to_hinglish(analysis.summary)
                analysis.key_points = translate_list(analysis.key_points)

                if label == "legal" and hasattr(analysis, "risk_report"):
                    analysis.risk_report.warnings = translate_list(analysis.risk_report.warnings)
                    analysis.risk_report.hidden_charges = translate_list(analysis.risk_report.hidden_charges)
                    analysis.risk_report.renewal_alerts = translate_list(analysis.risk_report.renewal_alerts)
                elif label == "government_form":
                  
                    analysis.required_docs = translate_list(analysis.required_docs)
                    analysis.deadlines = translate_list(analysis.deadlines)

            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())
            st.stop()

    # ═══════════════════════════════════════════════════════════════════════════
    #  Results
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("## 📊 Analysis Results")

    # ── Document type banner ──────────────────────────────────────────────────
    badge_class = "badge-legal" if label == "legal" else "badge-govt"
    badge_text  = "Legal Document" if label == "legal" else "Government Form"
    confidence_pct = f"{confidence * 100:.0f}%"

    st.markdown(f"""
    <div class="result-card" style="border-left: 5px solid {'#1a3c6e' if label == 'legal' else '#f5a623'};">
      <h3 style="margin:0">
        {'⚖️' if label == 'legal' else '📋'}
        &nbsp; <span class="badge {badge_class}">{badge_text}</span>
        &nbsp; {analysis.subtype}
      </h3>
      <p style="margin:0.5rem 0 0; color:#6b7a96; font-size:0.9rem;">
        AI confidence: {confidence_pct}
        {'&nbsp;|&nbsp; 🇮🇳 Hinglish mode ON' if hinglish_mode else ''}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Main content columns ──────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ── LEFT: Summary + key points ────────────────────────────────────────────
    with left_col:

        # Summary card
        st.markdown(f"""
        <div class="result-card">
          <h3>📝 Document Summary</h3>
          <p style="line-height:1.7">{analysis.summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key points
        if analysis.key_points:
            points_html = "".join(
                f'<div class="step-item">• {pt}</div>' for pt in analysis.key_points
            )
            st.markdown(f"""
            <div class="result-card">
              <h3>🔑 Key Points</h3>
              {points_html}
            </div>
            """, unsafe_allow_html=True)

        # ── LEGAL-specific ────────────────────────────────────────────────────
        if label == "legal":
            risk = analysis.risk_report

            # Risk meter
            level = risk.risk_level
            meter_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[level]
            emoji = risk_emoji(level)
            st.markdown(f"""
            <div class="risk-meter {meter_class}">
              <span style="font-size:1.8rem">{emoji}</span>
              <div>
                <div>Risk Level: <strong>{level}</strong></div>
                <div style="font-size:0.82rem;font-weight:400;opacity:0.8">
                  Risk score: {risk.risk_score}/100
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Warnings
            if risk.warnings:
                warn_html = "".join(
                    f'<div class="{"danger-item" if "waiver" in w.lower() or "unlimited" in w.lower() or "blacklist" in w.lower() else "warn-item"}">'
                    f'⚠️ {w}</div>'
                    for w in risk.warnings
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>🚨 Risky Clauses / Warnings</h3>
                  {warn_html}
                </div>
                """, unsafe_allow_html=True)

            # Hidden charges
            if risk.hidden_charges:
                charges_html = "".join(
                    f'<div class="danger-item">💸 {c}</div>' for c in risk.hidden_charges
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>💰 Hidden Charges Detected</h3>
                  {charges_html}
                </div>
                """, unsafe_allow_html=True)

            # Renewal alerts
            if risk.renewal_alerts:
                renewal_html = "".join(
                    f'<div class="warn-item">🔄 {r}</div>' for r in risk.renewal_alerts
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>🔔 Renewal / Termination Alerts</h3>
                  {renewal_html}
                </div>
                """, unsafe_allow_html=True)

        # ── GOVT-specific ──────────────────────────────────────────────────────
        else:
            # Purpose
            st.markdown(f"""
            <div class="result-card">
              <h3>🎯 Purpose of This Form</h3>
              <p style="line-height:1.7">{analysis.purpose}</p>
            </div>
            """, unsafe_allow_html=True)

            # Step-by-step guide
            if analysis.filling_guide:
                guide_html = "".join(
                    f'<div class="step-item"><strong>Step {i}:</strong> {step}</div>'
                    for i, step in enumerate(analysis.filling_guide, 1)
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>📝 Step-by-Step Filling Guide</h3>
                  {guide_html}
                </div>
                """, unsafe_allow_html=True)

    # ── RIGHT: Metadata panel ─────────────────────────────────────────────────
    with right_col:

        if label == "legal":
            # Parties
            if analysis.parties:
                parties_html = "".join(
                    f'<div style="padding:0.4rem;background:#f0f4ff;border-radius:6px;margin-bottom:0.3rem">'
                    f'👤 {p}</div>' for p in analysis.parties
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>👥 Parties Involved</h3>
                  {parties_html}
                </div>
                """, unsafe_allow_html=True)

            # Monetary values
            if analysis.monetary_values:
                money_html = "".join(
                    f'<div style="padding:0.3rem 0.5rem;font-family:monospace;font-size:0.9rem">'
                    f'💵 {v}</div>' for v in analysis.monetary_values[:8]
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>💰 Monetary Values Found</h3>
                  {money_html}
                </div>
                """, unsafe_allow_html=True)

            # Important dates
            if analysis.important_dates:
                dates_html = "".join(
                    f'<div style="padding:0.3rem 0.5rem;font-size:0.9rem">📅 {d}</div>'
                    for d in analysis.important_dates[:8]
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>📅 Important Dates</h3>
                  {dates_html}
                </div>
                """, unsafe_allow_html=True)

        else:
            # Required documents
            if analysis.required_docs:
                docs_html = "".join(
                    f'<div style="padding:0.4rem 0.5rem;border-left:3px solid #f5a623;'
                    f'background:#fffbf0;border-radius:4px;margin-bottom:0.35rem;font-size:0.88rem">'
                    f'📌 {doc}</div>' for doc in analysis.required_docs
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>📎 Required Documents</h3>
                  {docs_html}
                </div>
                """, unsafe_allow_html=True)

            # Deadlines
            if analysis.deadlines:
                deadline_html = "".join(
                    f'<div style="padding:0.4rem 0.5rem;border-left:3px solid #e84c3d;'
                    f'background:#fff5f5;border-radius:4px;margin-bottom:0.35rem;font-size:0.88rem">'
                    f'⏰ {dl}</div>' for dl in analysis.deadlines
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>⏰ Important Deadlines</h3>
                  {deadline_html}
                </div>
                """, unsafe_allow_html=True)

            # Missing field warnings
            if analysis.missing_field_warnings:
                mfw_html = "".join(
                    f'<div class="warn-item">⚠️ {w}</div>'
                    for w in analysis.missing_field_warnings
                )
                st.markdown(f"""
                <div class="result-card">
                  <h3>⚠️ Missing Field Warnings</h3>
                  {mfw_html}
                </div>
                """, unsafe_allow_html=True)

    # ── Raw text expander ─────────────────────────────────────────────────────
    with st.expander("📄 View Extracted Raw Text"):
        st.text_area(
            "Raw extracted text",
            value=raw_text[:3000] + (" …[truncated]" if len(raw_text) > 3000 else ""),
            height=250,
            label_visibility="collapsed",
        )

    # ── Document stats ────────────────────────────────────────────────────────
    from src.summarizer import get_document_stats
    stats = get_document_stats(raw_text)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Words", f"{stats['word_count']:,}")
    c2.metric("🔤 Characters", f"{stats['char_count']:,}")
    c3.metric("📖 Sentences", f"{stats['sentence_count']:,}")
    c4.metric("📏 Avg Sentence", f"{stats['avg_sentence_length']} words")
