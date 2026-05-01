# ⚖️ DocSaathi AI

> **Intelligent document simplification for legal & government documents in India**

DocSaathi AI reads your PDF, scanned image, or text file and explains it in **plain English** and **Hinglish** — detecting risks, hidden charges, required documents, and more.

---

## 🎯 What It Does

### Legal Documents
| Feature | Description |
|---|---|
| 📋 Document type detection | Identifies Rent Agreement, Loan Agreement, NDA, Notice, etc. |
| 📝 Plain-language summary | Extractive summary in simple English |
| 🚨 Risky clause detection | Highlights dangerous clauses (waiver of rights, unlimited liability, etc.) |
| 💰 Hidden charges detection | Finds late fees, processing fees, auto-debit mandates |
| 🔄 Renewal / termination alerts | Flags auto-renewal and lock-in periods |
| 🟢🟡🔴 Risk score | Low / Medium / High overall risk rating |

### Government Forms
| Feature | Description |
|---|---|
| 📋 Form type detection | Scholarship, PAN, Pension, Aadhaar, Job application |
| 🎯 Purpose explanation | Plain-language explanation of what the form is for |
| 📎 Required documents list | Exactly what you need to bring |
| ⏰ Deadline extraction | Important dates pulled from the form + known deadlines |
| 📝 Step-by-step filling guide | How to fill and submit the form |
| ⚠️ Missing field warnings | Detects blank / TBD fields in uploaded forms |

### Common Features
- 📄 PDF text extraction (pdfplumber)
- 🖼️ OCR for scanned images (pytesseract + Pillow)
- 🤖 ML text classification (TF-IDF + Logistic Regression)
- 🇮🇳 Hinglish mode toggle
- 📊 Document statistics (word count, sentence count)
- 🎨 Clean, modern Streamlit UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML | scikit-learn (TF-IDF + Logistic Regression) |
| NLP | NLTK |
| PDF | pdfplumber |
| OCR | pytesseract + Pillow |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
DocSaathi-AI/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── training_data.csv     # ML training dataset (40 samples)
├── models/                   # Auto-created; stores trained classifier
├── uploads/                  # Auto-created; temporary upload storage
└── src/
    ├── __init__.py
    ├── utils.py              # Shared helpers, logging, formatting
    ├── preprocess.py         # NLP preprocessing pipeline
    ├── pdf_reader.py         # PDF text extraction
    ├── ocr_reader.py         # Image OCR
    ├── classifier.py         # ML document classifier
    ├── summarizer.py         # Extractive summarisation
    ├── risk_detector.py      # Risk pattern detection
    ├── legal_analyzer.py     # Legal document analysis
    ├── govt_form_analyzer.py # Government form analysis
    └── translator.py         # Hinglish translation
```

---

## 🚀 How to Run Locally

### 1. Prerequisites

- Python 3.10 or higher
- pip
- **Tesseract OCR** (for scanned image support)

**Install Tesseract:**

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Download installer from [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract/releases) |

### 2. Clone / download the project

```bash
git clone https://github.com/your-username/DocSaathi-AI.git
cd DocSaathi-AI
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Download NLTK data (automatic)

NLTK data is downloaded automatically on first run. If you're offline:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 6. Run the application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧠 ML Model

The classifier is a **TF-IDF + Logistic Regression** pipeline trained on `data/training_data.csv`.

- **Labels:** `legal`, `government_form`
- **Features:** 1–2-gram TF-IDF with 5,000 features
- **Training:** Automatic on first run; model saved to `models/classifier.joblib`

To retrain from scratch:

```python
from src.classifier import train_model
train_model()
```

---

## 📊 Example Outputs

### Loan Agreement upload
```
Document Type : Legal Document  →  Loan Agreement
Risk Level    : 🟡 Medium  (score 42/100)
Warnings      : Auto-debit / automatic deduction enabled
                Late payment fee detected
                Compound interest clause found
Hidden Charges: Late payment fee detected
                Processing fee found
Summary       : This agreement outlines the terms of a personal loan including
                the principal, interest rate, EMI schedule, and default penalties.
```

### Scholarship Form upload
```
Document Type : Government Form  →  Scholarship Form
Purpose       : This form is used to apply for a financial scholarship…
Required Docs : Aadhaar Card, Income Certificate, Marksheet, Bank Passbook…
Deadlines     : Application window opens August–October each year
Step 1        : Register on NSP portal with Aadhaar number
Step 2        : Select correct scholarship scheme…
```

---

## 📝 Adding More Training Data

Open `data/training_data.csv` and add rows:

```csv
text,label
"your document text here",legal
"your form text here",government_form
```

Then delete `models/classifier.joblib` and restart the app to retrain.

---

## ⚠️ Limitations

- OCR quality depends on Tesseract + image quality
- Hinglish mode is rule-based (glossary), not a neural translation
- Risk detection is pattern-based; always consult a lawyer for legal advice
- ML model trained on a small dataset; accuracy improves with more data

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Made with ❤️ for India — Samajhna aasaan, jeena aasaan.*
