"""
Enhanced Brain Tumor MRI Classifier — app.py
Streamlit front-end for handcrafted XGBoost model.
CPU only | No GPU | No OpenCV
"""

import io
import base64
import datetime
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import streamlit as st
import joblib

from skimage import exposure
from skimage.feature import hog, local_binary_pattern

# ── Optional PDF support ─────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Image as RLImage, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:          #080c10;
    --surface:     #0d1117;
    --surface2:    #161b22;
    --border:      #21262d;
    --border2:     #30363d;
    --text:        #e6edf3;
    --muted:       #8b949e;
    --dim:         #484f58;
    --accent:      #58a6ff;
    --green:       #3fb950;
    --red:         #f85149;
    --yellow:      #d29922;
    --purple:      #a371f7;
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
  }
  [data-testid="stHeader"] { background: var(--bg) !important; }
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stMain"] > div { padding-top: 1.5rem; }

  /* ── Hero ── */
  .hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }
  .hero-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1a3a5c 0%, #0d1f3c 100%);
    border: 1px solid #1f4070;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.1;
  }
  .hero-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 0.2rem;
  }
  .version-pill {
    display: inline-block;
    background: #1c2d42;
    border: 1px solid #1f4070;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.06em;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploaderDropzone"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.2rem !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  /* ── Cards ── */
  .card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
  }
  .card-sm {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
  }

  /* ── Section label ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--dim);
    margin-bottom: 0.5rem;
  }

  /* ── Diagnosis ── */
  .diagnosis-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0.3rem 0 0.5rem;
    line-height: 1.1;
  }
  .conf-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .rank-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.65rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    background: #1c2d42;
    color: var(--accent);
    border: 1px solid #1f4070;
    margin-left: 0.5rem;
    vertical-align: middle;
  }

  /* ── Confidence bars ── */
  .bar-row { margin: 0.55rem 0; }
  .bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
    font-size: 0.86rem;
  }
  .bar-track {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
  }

  /* ── Size estimation ── */
  .size-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #12201a;
    border: 1px solid #1e3a2a;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #3fb950;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
  }

  /* ── Tumor info cards ── */
  .tumor-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
  }
  .tumor-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
  }
  .tumor-card-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
  }
  .symptom-pill {
    display: inline-block;
    background: #1c2128;
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: var(--muted);
    margin: 2px;
  }

  /* ── Divider ── */
  .my-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
  }

  /* ── Placeholder ── */
  .placeholder-card {
    background: var(--surface2);
    border: 1px dashed var(--border2);
    border-radius: 14px;
    min-height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.7rem;
    color: var(--dim);
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #1a1500;
    border: 1px solid #3d2e00;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.76rem;
    color: #b08000;
    margin-top: 1.2rem;
    line-height: 1.6;
  }

  /* ── Sidebar labels ── */
  [data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
  }
  [data-testid="stSidebar"] .stTextInput input,
  [data-testid="stSidebar"] .stSelectbox select,
  [data-testid="stSidebar"] .stNumberInput input,
  [data-testid="stSidebar"] .stTextArea textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.86rem !important;
  }

  /* ── Heatmap note ── */
  .heatmap-note {
    font-size: 0.72rem;
    color: var(--dim);
    margin-top: 0.5rem;
    font-style: italic;
  }

  /* ── Metric box ── */
  .metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
  }
  .metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
  }
  .metric-lbl {
    font-size: 0.7rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
  }

  /* Streamlit button override */
  [data-testid="stDownloadButton"] button,
  .stButton button {
    background: #1c2d42 !important;
    border: 1px solid #1f4070 !important;
    color: var(--accent) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
  }
  [data-testid="stDownloadButton"] button:hover,
  .stButton button:hover {
    background: #243650 !important;
    border-color: var(--accent) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models") / "handcrafted_xgb_brain_tumor.pkl"
IMG_SIZE   = 224
CLASSES    = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

CLASS_META = {
    "glioma_tumor":     {"label": "Glioma Tumor",      "color": "#f87171", "hex_bg": "#2a1010"},
    "meningioma_tumor": {"label": "Meningioma Tumor",  "color": "#a78bfa", "hex_bg": "#1a1230"},
    "no_tumor":         {"label": "No Tumor Detected", "color": "#34d399", "hex_bg": "#0f2a20"},
    "pituitary_tumor":  {"label": "Pituitary Tumor",   "color": "#fbbf24", "hex_bg": "#2a2000"},
}
TEXT = {
    "English": {
        "patient_info": "Patient Information",
        "history_title": "Prior Medical History",
        "full_name": "Full Name",
        "age": "Age",
        "sex": "Sex",
        "email": "Email",
        "address": "Address",
        "history": "History",
        "classifier": "🔬 Classifier",
        "about": "📖 About Tumors",
        "upload_title": "MRI Scan Upload",
        "upload_mri": "Upload MRI",
        "drop_scan": "Drop an MRI scan here",
        "results_here": "Results will appear here",
        "upload_begin": "Upload a scan to begin analysis",
        "analysing": "Analysing scan…",
        "ai_diagnosis": "AI Diagnosis",
        "top_predictions": "Top Predictions",
        "full_breakdown": "▸ Full confidence breakdown (all 4 classes)",
        "highlight_title": "Highlighted Suspicious Bright Region",
        "original_mri": "Original MRI",
        "highlighted_region": "Highlighted brighter region",
        "bright_note": "Highlighted area = pixels significantly brighter than surrounding brain tissue.",
        "about_intro": "Educational reference for the four categories recognised by this classifier.",
        "symptoms": "Symptoms",
        "risk_factors": "Risk Factors",
        "treatment": "Treatment",
        "prognosis": "Prognosis",
        "disclaimer": "This tool is intended for educational and research purposes only. It must NOT be used as a substitute for professional medical advice, clinical diagnosis, or treatment decisions. Always consult a qualified physician or radiologist.",
    },
    "Hindi": {
        "patient_info": "रोगी की जानकारी",
        "history_title": "पूर्व चिकित्सा इतिहास",
        "full_name": "पूरा नाम",
        "age": "उम्र",
        "sex": "लिंग",
        "email": "ईमेल",
        "address": "पता",
        "history": "इतिहास",
        "classifier": "🔬 वर्गीकरण",
        "about": "📖 ट्यूमर के बारे में",
        "upload_title": "MRI स्कैन अपलोड",
        "upload_mri": "MRI अपलोड करें",
        "drop_scan": "यहाँ MRI स्कैन डालें",
        "results_here": "परिणाम यहाँ दिखेंगे",
        "upload_begin": "विश्लेषण शुरू करने के लिए स्कैन अपलोड करें",
        "analysing": "स्कैन का विश्लेषण हो रहा है…",
        "ai_diagnosis": "AI निदान",
        "top_predictions": "शीर्ष अनुमान",
        "full_breakdown": "▸ सभी 4 वर्गों का पूर्ण confidence breakdown",
        "highlight_title": "संदिग्ध चमकीला क्षेत्र",
        "original_mri": "मूल MRI",
        "highlighted_region": "चमकीला क्षेत्र हाइलाइट किया गया",
        "bright_note": "हाइलाइट किया गया भाग आसपास के brain tissue से ज्यादा चमकीले pixels दिखाता है.",
        "about_intro": "इस classifier द्वारा पहचानी जाने वाली चार categories की educational जानकारी.",
        "symptoms": "लक्षण",
        "risk_factors": "जोखिम कारक",
        "treatment": "उपचार",
        "prognosis": "पूर्वानुमान",
        "disclaimer": "यह tool केवल educational और research purposes के लिए है। इसे professional medical advice, diagnosis या treatment decision का substitute न मानें। हमेशा qualified physician या radiologist से सलाह लें।",
    },
    "Tamil": {
        "patient_info": "நோயாளர் தகவல்",
        "history_title": "முந்தைய மருத்துவ வரலாறு",
        "full_name": "முழு பெயர்",
        "age": "வயது",
        "sex": "பாலினம்",
        "email": "மின்னஞ்சல்",
        "address": "முகவரி",
        "history": "வரலாறு",
        "classifier": "🔬 வகைப்படுத்தி",
        "about": "📖 கட்டிகள் பற்றி",
        "upload_title": "MRI ஸ்கேன் பதிவேற்றம்",
        "upload_mri": "MRI பதிவேற்றவும்",
        "drop_scan": "MRI ஸ்கேன் இங்கே விடவும்",
        "results_here": "முடிவுகள் இங்கே தோன்றும்",
        "upload_begin": "ஆய்வு தொடங்க ஸ்கேன் பதிவேற்றவும்",
        "analysing": "ஸ்கேன் ஆய்வு செய்யப்படுகிறது…",
        "ai_diagnosis": "AI நோயறிதல்",
        "top_predictions": "முக்கிய கணிப்புகள்",
        "full_breakdown": "▸ அனைத்து 4 வகைகளின் confidence breakdown",
        "highlight_title": "சந்தேகமான பிரகாசமான பகுதி",
        "original_mri": "அசல் MRI",
        "highlighted_region": "பிரகாசமான பகுதி குறிக்கப்பட்டது",
        "bright_note": "குறிக்கப்பட்ட பகுதி சுற்றியுள்ள brain tissue-ஐ விட அதிகமாக பிரகாசிக்கும் pixels-ஐ காட்டுகிறது.",
        "about_intro": "இந்த classifier அடையாளம் காணும் நான்கு categories பற்றிய educational reference.",
        "symptoms": "அறிகுறிகள்",
        "risk_factors": "ஆபத்து காரணிகள்",
        "treatment": "சிகிச்சை",
        "prognosis": "முன்னறிவு",
        "disclaimer": "இந்த tool educational மற்றும் research purposes-க்காக மட்டுமே. Professional medical advice, diagnosis அல்லது treatment decision-க்கு பதிலாக பயன்படுத்தக்கூடாது. Qualified physician அல்லது radiologist-ஐ அணுகவும்.",
    },
}
TUMOR_DATA = {
    "glioma_tumor": {
        "description": (
            "Gliomas arise from the brain's glial cells — the supportive tissue that surrounds "
            "and protects neurons. They represent the most common type of primary brain tumor "
            "(≈33% of all brain tumors) and are classified by WHO grade (I–IV). Grade IV gliomas "
            "(Glioblastoma Multiforme) are among the most aggressive human cancers."
        ),
        "symptoms": [
            "Persistent or worsening headaches",
            "Nausea & vomiting (especially morning)",
            "New-onset seizures",
            "Cognitive / personality changes",
            "Progressive weakness (arm or leg)",
            "Speech or language difficulties",
            "Blurred or double vision",
            "Memory impairment",
        ],
        "risk_factors": "Age (peak 45–75), male sex, prior radiation exposure, rare genetic syndromes (NF1, Li-Fraumeni).",
        "treatment": "Surgery, radiation therapy, temozolomide chemotherapy; bevacizumab for recurrence.",
        "prognosis": "Highly variable by grade. GBM median survival ~15 months with standard of care.",
    },
    "meningioma_tumor": {
        "description": (
            "Meningiomas grow from the meninges — the three-layered membrane enveloping the brain "
            "and spinal cord. They account for ~37% of all brain tumors and are the most common "
            "benign intracranial tumor. Most are WHO grade I (benign), slow-growing, and discovered "
            "incidentally. Atypical (grade II) and anaplastic (grade III) forms do exist."
        ),
        "symptoms": [
            "Gradual headaches",
            "Vision changes or loss",
            "Hearing loss or tinnitus",
            "Anosmia (loss of smell)",
            "Seizures",
            "Memory or concentration problems",
            "Weakness in limbs",
            "Personality changes (frontal lobe tumors)",
        ],
        "risk_factors": "Female sex (2:1), age 40–70, prior ionising radiation to head, NF2 gene mutations.",
        "treatment": "Watchful waiting for small asymptomatic tumors; surgery; stereotactic radiosurgery (SRS).",
        "prognosis": "Grade I: excellent after complete resection. Grade II–III: higher recurrence, requires adjuvant RT.",
    },
    "no_tumor": {
        "description": (
            "No tumor tissue was detected in this MRI scan. The scan appears within normal limits "
            "for the imaged brain structures. A negative classification does not rule out all "
            "pathology — always correlate with clinical symptoms and consult a radiologist."
        ),
        "symptoms": ["N/A — no tumor signal detected."],
        "risk_factors": "N/A",
        "treatment": "N/A",
        "prognosis": "N/A",
    },
    "pituitary_tumor": {
        "description": (
            "Pituitary adenomas originate in the anterior pituitary gland at the base of the brain. "
            "They are almost always benign (WHO grade I) and represent ~15% of all primary brain "
            "tumors. Classified as microadenomas (<10 mm) or macroadenomas (≥10 mm) and as "
            "functional (hormone-secreting) or non-functional."
        ),
        "symptoms": [
            "Bitemporal visual field defects ('tunnel vision')",
            "Headaches (retro-orbital)",
            "Hormonal excess: acromegaly, Cushing's disease, hyperprolactinaemia",
            "Hormonal deficiency: hypogonadism, hypothyroidism, adrenal insufficiency",
            "Fatigue & weight gain",
            "Sexual dysfunction",
            "Mood disturbances",
            "Galactorrhoea (prolactinoma)",
        ],
        "risk_factors": "Mostly sporadic; rare genetic: MEN1, AIP, PRKAR1A mutations.",
        "treatment": "Dopamine agonists (prolactinoma); transsphenoidal surgery; somatostatin analogues; radiosurgery.",
        "prognosis": "Generally excellent. Most functional tumors are controllable; macroadenomas may require surgery.",
    },
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ── Feature pipeline (mirrors train.py exactly) ───────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = ImageOps.equalize(img)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = exposure.equalize_adapthist(arr, clip_limit=0.03)
    return arr


def extract_features(arr: np.ndarray) -> np.ndarray:
    hog_fine = hog(arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    hog_coarse = hog(arr, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                     block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    lbp1 = local_binary_pattern(arr, P=8,  R=1, method="uniform")
    lbp1_hist, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10), density=True)
    lbp3 = local_binary_pattern(arr, P=16, R=3, method="uniform")
    lbp3_hist, _ = np.histogram(lbp3.ravel(), bins=18, range=(0, 18), density=True)
    stats = np.array([
        arr.mean(), arr.std(), arr.min(), arr.max(), np.median(arr),
	np.percentile(arr, 10), np.percentile(arr, 25),
        np.percentile(arr, 75), np.percentile(arr, 90),
        float(np.sum(arr > 0.5)) / arr.size,
        float(np.sum(arr > 0.7)) / arr.size,
    ], dtype=np.float32)
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    h, w = arr.shape
    quads = [arr[:h//2, :w//2], arr[:h//2, w//2:], arr[h//2:, :w//2], arr[h//2:, w//2:]]
    quad_feats = np.array(
        [v for q in quads for v in (q.mean(), q.std(), float(np.percentile(q, 75)))],
        dtype=np.float32,
    )
    return np.concatenate(
        [hog_fine, hog_coarse, lbp1_hist, lbp3_hist, stats, hist, quad_feats]
    ).astype(np.float32)


def run_inference(model, img: Image.Image):
    arr   = preprocess_image(img)
    feats = extract_features(arr).reshape(1, -1)
    probs = model.predict_proba(feats)[0]
    pred  = CLASSES[int(np.argmax(probs))]
    return pred, probs, arr


# ── Occlusion Saliency Map ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_saliency(_model, arr_bytes: bytes, pred_idx: int, grid: int = 10):
    """
    Grid-based occlusion saliency map.
    Occlude each cell → measure confidence drop for predicted class.
    Cached per image+prediction to avoid recomputation on re-renders.
    """
    arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
    base_probs = _model.predict_proba(extract_features(arr).reshape(1, -1))[0]
    base_conf  = base_probs[pred_idx]
    fill_val   = float(arr.mean())
    cell_h     = IMG_SIZE // grid
    cell_w     = IMG_SIZE // grid
    saliency   = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            occ = arr.copy()
            occ[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = fill_val
            probs = _model.predict_proba(extract_features(occ).reshape(1, -1))[0]
            saliency[i, j] = base_conf - probs[pred_idx]
    saliency = np.maximum(saliency, 0)
    if saliency.max() > 1e-6:
        saliency /= saliency.max()
    return saliency


def render_heatmap(orig_pil: Image.Image, saliency: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlay saliency map on original MRI as a colour heatmap."""
    sal_up = np.array(
        Image.fromarray((saliency * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BICUBIC
        ), dtype=np.float32
    ) / 255.0
    sal_pil = Image.fromarray((sal_up * 255).astype(np.uint8))
    sal_pil = sal_pil.filter(ImageFilter.GaussianBlur(radius=8))
    sal_up  = np.array(sal_pil, dtype=np.float32) / 255.0

    cmap     = cm.get_cmap("jet")
    heat_rgb = cmap(sal_up)[..., :3]
    heat_pil = Image.fromarray((heat_rgb * 255).astype(np.uint8))

    base = orig_pil.convert("L").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    out  = Image.blend(base, heat_pil, alpha)
    return out


def estimate_tumor_size(saliency: np.ndarray, threshold: float = 0.5):
    """Estimate fraction of image occupied by high-saliency region."""
    sal_up = np.array(
        Image.fromarray((saliency * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BICUBIC
        ), dtype=np.float32
    ) / 255.0
    mask  = sal_up >= threshold
    pct   = float(mask.sum()) / mask.size * 100.0
    rows  = np.any(mask, axis=1)
    cols  = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_w = int(cmax - cmin)
        bbox_h = int(rmax - rmin)
    else:
        bbox_w = bbox_h = 0
    return pct, bbox_w, bbox_h
def detect_bright_regions(arr: np.ndarray, z_thresh: float = 1.45):
    brain_mask = arr > 0.05
    if brain_mask.sum() < 100:
        return False, 0.0, np.zeros_like(arr)

    brain_pixels = arr[brain_mask]
    mu = brain_pixels.mean()
    sigma = brain_pixels.std()

    bright_mask = (arr > (mu + z_thresh * sigma)) & brain_mask

    # Remove skull / outer border: keep only central brain-ish region
    h, w = arr.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2

    ellipse = (
        ((xx - cx) / (w * 0.34)) ** 2 +
        ((yy - cy) / (h * 0.38)) ** 2
    ) <= 1.0

    bright_mask = bright_mask & ellipse

    bright_pct = float(bright_mask.sum()) / brain_mask.sum() * 100.0
    flagged = bright_pct > 0.2

    return flagged, bright_pct, bright_mask.astype(np.float32)
# ── PDF Report ────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def generate_pdf_report(
    patient: dict,
    pred_class: str,
    probs: np.ndarray,
    orig_img: Image.Image,
    heatmap_img: Image.Image,
    tumor_pct: float,
    bbox_w: int,
    bbox_h: int,
) -> bytes:
    if not REPORTLAB_OK:
        return b""

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        topMargin=0.8*inch,  bottomMargin=0.8*inch,
    )
    styles = getSampleStyleSheet()
    W, _   = A4

    def S(name, **kw):
        return ParagraphStyle(name, **{
            "fontName": "Helvetica", "textColor": rl_colors.HexColor("#e6edf3"),
            "backColor": rl_colors.HexColor("#0d1117"), **kw
        })

    title_s = S("title", fontSize=18, fontName="Helvetica-Bold", spaceAfter=4, leading=22)
    sub_s   = S("sub",   fontSize=9,  textColor=rl_colors.HexColor("#8b949e"), spaceAfter=16)
    label_s = S("label", fontSize=7,  fontName="Helvetica-Bold",
                textColor=rl_colors.HexColor("#484f58"), spaceBefore=8, spaceAfter=2, leading=10)
    body_s  = S("body",  fontSize=9,  leading=14, spaceAfter=4)
    warn_s  = S("warn",  fontSize=7.5,
                textColor=rl_colors.HexColor("#b08000"),
                backColor=rl_colors.HexColor("#1a1500"),
                borderColor=rl_colors.HexColor("#3d2e00"),
                borderWidth=1, borderPadding=6, leading=12)

    COLOR_MAP = {
        "glioma_tumor":     "#f87171",
        "meningioma_tumor": "#a78bfa",
        "no_tumor":         "#34d399",
        "pituitary_tumor":  "#fbbf24",
    }

    elements = []

    # ── Header ──
    elements.append(Paragraph("NeuroScan AI — Diagnostic Report", title_s))
    elements.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
        f"Model: Handcrafted XGBoost (CPU) v1.0",
        sub_s,
    ))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=rl_colors.HexColor("#21262d"), spaceAfter=12))

    # ── Patient info table ──
    elements.append(Paragraph("PATIENT INFORMATION", label_s))
    fields = [
        ["Full Name",    patient.get("name", "—"),  "Date of Birth / Age", patient.get("age", "—")],
        ["Sex",          patient.get("sex", "—"),   "Email",               patient.get("email", "—")],
        ["Address",      patient.get("address", "—"), "Report Date",       datetime.datetime.now().strftime("%d %b %Y")],
    ]
    pt_data = []
    for row in fields:
        pt_data.append([
            Paragraph(row[0], ParagraphStyle("lh",  fontName="Helvetica-Bold", fontSize=7.5,
                textColor=rl_colors.HexColor("#8b949e"))),
            Paragraph(str(row[1]), ParagraphStyle("lv", fontName="Helvetica", fontSize=8.5,
                textColor=rl_colors.HexColor("#e6edf3"))),
            Paragraph(row[2], ParagraphStyle("lh2", fontName="Helvetica-Bold", fontSize=7.5,
                textColor=rl_colors.HexColor("#8b949e"))),
            Paragraph(str(row[3]), ParagraphStyle("lv2", fontName="Helvetica", fontSize=8.5,
                textColor=rl_colors.HexColor("#e6edf3"))),
        ])
    pt_table = Table(pt_data, colWidths=[1.1*inch, 2.6*inch, 1.5*inch, 2.1*inch])
    pt_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#161b22")),
        ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#21262d")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    elements.append(pt_table)
    elements.append(Spacer(1, 10))

    # ── Prior medical history ──
    if patient.get("history", "").strip():
        elements.append(Paragraph("PRIOR MEDICAL HISTORY", label_s))
        history_table = Table(
            [[Paragraph(patient["history"], body_s)]],
            colWidths=[7.2*inch]
        )
        history_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#161b22")),
            ("BOX",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#21262d")),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        elements.append(history_table)
        elements.append(Spacer(1, 10))

    elements.append(HRFlowable(width="100%", thickness=1,
                               color=rl_colors.HexColor("#21262d"), spaceAfter=10))

    # ── AI Diagnosis ──
    elements.append(Paragraph("AI CLASSIFICATION RESULT", label_s))
    pred_color = rl_colors.HexColor(COLOR_MAP.get(pred_class, "#e6edf3"))
    diag_data  = [[
        Paragraph("Primary Prediction", ParagraphStyle("ph", fontName="Helvetica-Bold",
            fontSize=8, textColor=rl_colors.HexColor("#8b949e"))),
        Paragraph(CLASS_META[pred_class]["label"], ParagraphStyle("pv",
            fontName="Helvetica-Bold", fontSize=14, textColor=pred_color)),
        Paragraph("Confidence", ParagraphStyle("ch", fontName="Helvetica-Bold",
            fontSize=8, textColor=rl_colors.HexColor("#8b949e"))),
        Paragraph(f"{probs[CLASSES.index(pred_class)]:.1%}", ParagraphStyle("cv",
            fontName="Helvetica-Bold", fontSize=14, textColor=pred_color)),
    ]]
    diag_table = Table(diag_data, colWidths=[1.4*inch, 3.0*inch, 1.0*inch, 1.8*inch])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#161b22")),
        ("BOX",           (0,0), (-1,-1), 1, pred_color),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 8))

    # ── Confidence breakdown ──
    elements.append(Paragraph("CONFIDENCE BREAKDOWN (ALL CLASSES)", label_s))
    sorted_preds = sorted(zip(CLASSES, probs), key=lambda x: -x[1])
    conf_rows = []
    for rank, (cls, prob) in enumerate(sorted_preds, 1):
        c = rl_colors.HexColor(COLOR_MAP.get(cls, "#e6edf3"))
        conf_rows.append([
            Paragraph(f"#{rank}", ParagraphStyle("rk", fontName="Helvetica-Bold",
                fontSize=9, textColor=rl_colors.HexColor("#484f58"))),
            Paragraph(CLASS_META[cls]["label"], ParagraphStyle("cn",
                fontName="Helvetica", fontSize=9.5, textColor=c)),
            Paragraph(f"{prob:.4f}", ParagraphStyle("cv2",
                fontName="Helvetica-Bold", fontSize=9.5, textColor=c)),
            Paragraph(f"{prob:.1%}", ParagraphStyle("cp",
                fontName="Helvetica-Bold", fontSize=9.5, textColor=c)),
        ])
    conf_table = Table(conf_rows, colWidths=[0.5*inch, 3.0*inch, 1.5*inch, 1.5*inch])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#0d1117")),
        ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#21262d")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    elements.append(conf_table)
    elements.append(Spacer(1, 10))

    # ── Tumor size ──
    if pred_class != "no_tumor":
        elements.append(Paragraph("AI-ESTIMATED REGION (NOT A CLINICAL MEASUREMENT)", label_s))
        size_data = [[
            Paragraph(f"{tumor_pct:.1f}%", ParagraphStyle("tv", fontName="Helvetica-Bold",
                fontSize=13, textColor=rl_colors.HexColor("#3fb950"))),
            Paragraph("of scan area flagged\nas high-attention region",
                ParagraphStyle("td", fontName="Helvetica", fontSize=8,
                textColor=rl_colors.HexColor("#8b949e"))),
            Paragraph(f"{bbox_w} x {bbox_h} px", ParagraphStyle("bv",
                fontName="Helvetica-Bold", fontSize=13,
                textColor=rl_colors.HexColor("#58a6ff"))),
            Paragraph("bounding box\n(pixel estimate)",
                ParagraphStyle("bd", fontName="Helvetica", fontSize=8,
                textColor=rl_colors.HexColor("#8b949e"))),
        ]]
        size_table = Table(size_data, colWidths=[1.2*inch, 2.2*inch, 1.5*inch, 2.0*inch])
        size_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), rl_colors.HexColor("#0f2a20")),
            ("BOX",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#1e3a2a")),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        elements.append(size_table)
        elements.append(Spacer(1, 10))

    elements.append(HRFlowable(width="100%", thickness=1,
                               color=rl_colors.HexColor("#21262d"), spaceAfter=10))

    # ── Images ──
    elements.append(Paragraph("SCAN IMAGES", label_s))
    img_size_pt = 2.8 * inch
    orig_rgb = orig_img.convert("RGB").resize((300, 300), Image.LANCZOS)
    heat_rgb = heatmap_img.resize((300, 300), Image.LANCZOS)
    orig_b   = pil_to_bytes(orig_rgb)
    heat_b   = pil_to_bytes(heat_rgb)
    img_data = [[
        RLImage(io.BytesIO(orig_b), width=img_size_pt, height=img_size_pt),
        RLImage(io.BytesIO(heat_b), width=img_size_pt, height=img_size_pt),
    ]]
    cap_data = [[
        Paragraph("Original MRI Scan", ParagraphStyle("ic", fontSize=8,
            textColor=rl_colors.HexColor("#8b949e"), alignment=TA_CENTER)),
        Paragraph("AI Attention Heatmap (Occlusion Saliency)", ParagraphStyle("ic2",
            fontSize=8, textColor=rl_colors.HexColor("#8b949e"), alignment=TA_CENTER)),
    ]]
    img_table = Table(img_data + cap_data, colWidths=[img_size_pt + 0.4*inch] * 2)
    img_table.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 10))

    # ── About the tumor ──
    tdata = TUMOR_DATA[pred_class]
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=rl_colors.HexColor("#21262d"), spaceAfter=10))
    elements.append(Paragraph(f"ABOUT: {CLASS_META[pred_class]['label'].upper()}", label_s))
    elements.append(Paragraph(tdata["description"], body_s))
    elements.append(Spacer(1, 6))

    if pred_class != "no_tumor":
        elements.append(Paragraph("Key Symptoms:", ParagraphStyle("sh",
            fontName="Helvetica-Bold", fontSize=8.5,
            textColor=rl_colors.HexColor("#8b949e"), spaceAfter=3)))
        for sym in tdata["symptoms"]:
            elements.append(Paragraph(f"  • {sym}", body_s))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"<b>Risk Factors:</b> {tdata['risk_factors']}", body_s))
        elements.append(Paragraph(f"<b>Treatment:</b> {tdata['treatment']}", body_s))
        elements.append(Paragraph(f"<b>Prognosis:</b> {tdata['prognosis']}", body_s))

    elements.append(Spacer(1, 14))

    # ── Disclaimer ──
    elements.append(Paragraph(
        "WARNING: This report is generated by an AI research tool for educational and "
        "non-clinical purposes ONLY. It must NOT be used as a substitute for professional "
        "medical advice, diagnosis, or treatment. Always seek the guidance of a qualified "
        "physician or radiologist with any questions regarding a medical condition.",
        warn_s,
    ))

    doc.build(elements)
    return buf.getvalue()


# ── Sidebar — Patient Info ────────────────────────────────────────────────────
with st.sidebar:
    language = st.selectbox("Language / भाषा / மொழி", ["English", "Hindi", "Tamil"])
    T = TEXT[language]
    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
        'color:#484f58;letter-spacing:0.12em;text-transform:uppercase;'
        'font-weight:700;margin-bottom:0.5rem">Patient Information</p>',
        unsafe_allow_html=True,
    )
    p_name  = st.text_input("Full Name", placeholder="e.g. John Doe")
    col1, col2 = st.columns(2)
    with col1:
        p_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    with col2:
        p_sex = st.selectbox("Sex", ["—", "Male", "Female", "Other"])
    p_email   = st.text_input("Email",   placeholder="patient@email.com")
    p_address = st.text_area("Address", placeholder="Street, City, State", height=68)
    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
        'color:#484f58;letter-spacing:0.12em;text-transform:uppercase;'
        'font-weight:700;margin-bottom:0.5rem">Prior Medical History</p>',
        unsafe_allow_html=True,
    )
    p_history = st.text_area(
        "History",
        placeholder="List any prior diagnoses, surgeries, medications, allergies, "
                    "family history of neurological conditions, etc.",
        height=130,
        label_visibility="collapsed",
    )
    st.markdown("---")
    if not REPORTLAB_OK:
        st.warning("Install `reportlab` for PDF export:\n```\npip install reportlab\n```")

patient = {
    "name": p_name, "age": f"{p_age}" if p_age else "—",
    "sex": p_sex, "email": p_email,
    "address": p_address, "history": p_history,
}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🧠</div>
  <div>
    <div class="hero-title">NeuroScan AI</div>
    <div class="hero-sub">
      Brain Tumor MRI Classifier &nbsp;
      <span class="version-pill">XGBoost · CPU · v1.0</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_classifier, tab_about = st.tabs(["🔬 Classifier", "📖 About Tumors"])

model = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_classifier:
    if model is None:
        st.error(
            "⚠️ Model not found at `models/handcrafted_xgb_brain_tumor.pkl`. "
            "Run `python train.py` to generate it first."
        )
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: Upload + image ──────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">MRI Scan Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload MRI",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_column_width=True, caption="Uploaded MRI scan")
        else:
            st.markdown("""
            <div class="placeholder-card" style="min-height:260px">
              <div style="font-size:2.8rem;opacity:0.4">🩻</div>
              <div style="font-size:0.88rem">Drop an MRI scan here</div>
              <div style="font-size:0.72rem">JPG · JPEG · PNG accepted</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right: Results ────────────────────────────────────────────────────────
    with col_right:
        if not uploaded:
            st.markdown("""
            <div class="placeholder-card">
              <div style="font-size:2.8rem;opacity:0.4">📊</div>
              <div style="font-size:0.88rem">Results will appear here</div>
              <div style="font-size:0.72rem">Upload a scan to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analysing scan…"):
                pred_class, probs, arr = run_inference(model, pil_img)

                pred_idx  = CLASSES.index(pred_class)
                meta      = CLASS_META[pred_class]
                pred_prob = probs[pred_idx]
                color     = meta["color"]
                bg_color  = meta["hex_bg"]
                sorted_preds = sorted(zip(CLASSES, probs), key=lambda x: -x[1])
                top2 = sorted_preds[:2]
                bright_flagged, bright_pct, bright_mask = detect_bright_regions(arr)

                if pred_class == "no_tumor" and bright_flagged:
                    st.markdown(f"""
                        <div style="background:#1a0a0a;border:1px solid #7f1d1d;border-left:4px solid #ef4444;
                        border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.8rem">
                        <div style="color:#ef4444;font-family:'Space Mono',monospace;font-size:0.72rem;
                        font-weight:700;letter-spacing:0.1em;margin-bottom:0.3rem">
                        ⚠ MODEL UNCERTAINTY WARNING
                        </div>
                        <div style="color:#fca5a5;font-size:0.84rem;line-height:1.5">
                        The model predicted <strong>No Tumor</strong> ({pred_prob:.1%} confidence),
                        but the scan contains <strong>{bright_pct:.1f}% anomalously bright pixels</strong>
                        — which may indicate a lesion the model missed.
                        <br><br>
                        <strong>This scan should be reviewed by a qualified radiologist.</strong>
                        Handcrafted feature models (HOG+LBP+XGBoost) have known limitations on
                        heterogeneous or contrast-enhanced MRIs.
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                elif pred_prob < 0.60:
                    st.markdown(f"""
                        <div style="background:#1a1500;border:1px solid #713f12;border-left:4px solid #f59e0b;
                        border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.8rem">
                        <div style="color:#f59e0b;font-family:'Space Mono',monospace;font-size:0.72rem;
                        font-weight:700;letter-spacing:0.1em;margin-bottom:0.3rem">
                        ⚠ LOW CONFIDENCE PREDICTION
                        </div>
                        <div style="color:#fde68a;font-size:0.84rem;line-height:1.5">
                        Model confidence is only <strong>{pred_prob:.1%}</strong>.
                        Treat this result with caution and seek radiologist review.
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

                    sorted_preds = sorted(zip(CLASSES, probs), key=lambda x: -x[1])
                    top2 = sorted_preds[:2]

            # ── Diagnosis card ──
            st.markdown(f"""
            <div class="card">
              <div class="section-label">AI Diagnosis</div>
              <div class="diagnosis-name" style="color:{color}">
                {meta['label']}
                <span class="rank-badge">#1</span>
              </div>
              <span class="conf-badge" style="background:{bg_color};color:{color};border:1px solid {color}40">
                {pred_prob:.1%} confidence
              </span>
              <div style="color:#8b949e;font-size:0.84rem;margin-top:0.8rem;line-height:1.55">
                {TUMOR_DATA[pred_class]['description'][:220]}…
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Top-2 predictions ──
            st.markdown('<div class="section-label" style="margin-top:0.2rem">Top Predictions</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card" style="padding:1.2rem">', unsafe_allow_html=True)
            for rank, (cls, prob) in enumerate(top2, 1):
                m         = CLASS_META[cls]
                is_top    = rank == 1
                label_col = "#e6edf3" if is_top else "#8b949e"
                font_w    = "600"     if is_top else "400"
                bar_col   = m["color"] if is_top else m["color"] + "66"
                st.markdown(f"""
                <div class="bar-row">
                  <div class="bar-header">
                    <span style="color:{label_col};font-weight:{font_w}">
                      #{rank} &nbsp; {m['label']}
                    </span>
                    <span style="color:{label_col};font-family:'Space Mono',monospace;font-size:0.82rem">
                      {prob:.1%}
                    </span>
                  </div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{prob*100:.1f}%;background:{bar_col}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Full confidence breakdown ──
            with st.expander("▸ Full confidence breakdown (all 4 classes)"):
                for cls, prob in sorted_preds:
                    m = CLASS_META[cls]
                    is_pred = cls == pred_class
                    st.markdown(f"""
                    <div class="bar-row">
                      <div class="bar-header">
                        <span style="color:{'#e6edf3' if is_pred else '#8b949e'};
                                     font-weight:{'600' if is_pred else '400'}">
                          {m['label']}
                        </span>
                        <span style="color:{'#e6edf3' if is_pred else '#8b949e'};
                                     font-family:'Space Mono',monospace;font-size:0.82rem">
                          {prob:.4f} &nbsp; ({prob:.1%})
                        </span>
                      </div>
                      <div class="bar-track">
                        <div class="bar-fill"
                             style="width:{prob*100:.1f}%;
                                    background:{m['color'] if is_pred else m['color']+'55'}">
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Bright Region Highlight ───────────────────────────────────────────────────
if uploaded and pred_class != "no_tumor":
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Highlighted Suspicious Bright Region</div>',
                unsafe_allow_html=True)

    bright_flagged, bright_pct, bright_mask = detect_bright_regions(arr)
    base = pil_img.convert("L").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    mask_img = Image.fromarray((bright_mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

    # Make highlighted region thicker, no OpenCV needed
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=5))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
    overlay = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 255, 170))

    highlighted = Image.composite(overlay, base, mask_img)
    highlighted = Image.blend(base, highlighted, 0.55)

    st.image(highlighted, use_column_width=True, caption="Detected bright anomaly region")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.image(base, use_column_width=True, caption="Original MRI")

    with col_b:
        st.image(
            highlighted,
            use_column_width=True,
            caption="Highlighted brighter region"
        )
        st.markdown(
            f"""
            <p class="heatmap-note">
            Highlighted area = pixels significantly brighter than surrounding brain tissue.
            Bright region estimate: <strong>{bright_pct:.2f}%</strong> of brain area.
            This is not a clinical tumor boundary.
            </p>
            """,
            unsafe_allow_html=True,
        )
    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>DISCLAIMER</strong> — This tool is intended for educational and research purposes
      only. It must <strong>NOT</strong> be used as a substitute for professional medical advice,
      clinical diagnosis, or treatment decisions. Always consult a qualified physician or
      radiologist. Confidence scores are model outputs and do not represent clinical certainty.
      Tumor size estimates are AI approximations, not radiological measurements.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT TUMORS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        '<p style="color:#8b949e;font-size:0.9rem;margin-bottom:1.5rem">'
        'Educational reference for the four categories recognised by this classifier.</p>',
        unsafe_allow_html=True,
    )

    ACCENT_COLORS = {
        "glioma_tumor":     ("#f87171", "#2a1010"),
        "meningioma_tumor": ("#a78bfa", "#1a1230"),
        "no_tumor":         ("#34d399", "#0f2a20"),
        "pituitary_tumor":  ("#fbbf24", "#2a2000"),
    }

    for cls in CLASSES:
        color, bg = ACCENT_COLORS[cls]
        td        = TUMOR_DATA[cls]
        meta      = CLASS_META[cls]

        st.markdown(f"""
        <div class="tumor-card" style="border-left:4px solid {color}">
          <div class="tumor-card-name" style="color:{color}">{meta['label']}</div>
          <p style="font-size:0.86rem;color:#8b949e;line-height:1.65;margin-bottom:0.8rem">
            {td['description']}
          </p>
        """, unsafe_allow_html=True)

        if cls != "no_tumor":
            sym_pills = "".join(
                f'<span class="symptom-pill">⚡ {s}</span>' for s in td["symptoms"]
            )
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
              <div class="section-label" style="margin-bottom:0.4rem">Symptoms</div>
              <div>{sym_pills}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.5rem">
              <div class="card-sm">
                <div class="section-label">Risk Factors</div>
                <div style="font-size:0.82rem;color:#8b949e;line-height:1.5">{td['risk_factors']}</div>
              </div>
              <div class="card-sm">
                <div class="section-label">Treatment</div>
                <div style="font-size:0.82rem;color:#8b949e;line-height:1.5">{td['treatment']}</div>
              </div>
            </div>
            <div class="card-sm" style="margin-top:0.6rem">
              <div class="section-label">Prognosis</div>
              <div style="font-size:0.82rem;color:#8b949e;line-height:1.5">{td['prognosis']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ The information above is a general educational summary and does not constitute medical
      advice. Symptoms, treatments, and prognosis vary significantly by individual case.
      Always consult a qualified medical professional.
    </div>
    """, unsafe_allow_html=True)
