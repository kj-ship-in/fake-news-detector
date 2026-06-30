import streamlit as st
import joblib
from pathlib import Path

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

# -------------------------------
# LOAD MODEL (SAFE)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model_resources():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer
        except Exception:
            return None, None
    return None, None

model, vectorizer = load_model_resources()

# -------------------------------
# STYLES
# -------------------------------
st.markdown(
    """
    <style>
    .stApp, .block-container, .main {
        background: radial-gradient(circle at top, #1a0f2e, #0b0715) !important;
        color: #f8fafc !important;
    }

    h1, h2, h3, h4, h5 {
        color: #c084fc !important;
    }

    .stTextArea textarea {
        background-color: #1b102f !important;
        color: #f8fafc !important;
        border: 1px solid #2d1250 !important;
        border-radius: 12px !important;
    }

    .stButton button {
        background: linear-gradient(90deg, #7c3aed, #c084fc) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .result-box {
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("📌 Project Info")
    st.write("Fake News Detection AI using Machine Learning")

    st.markdown("### How to use")
    st.markdown("""
    1. Paste a news article  
    2. Click Analyze News  
    3. View prediction result  
    """)

    st.markdown("---")

    if model is None or vectorizer is None:
        st.warning("⚠ Backend model not available yet.")
        st.write("Waiting for Omar’s backend (model.pkl & vectorizer.pkl).")
    else:
        st.success("✅ Model loaded successfully!")

    st.markdown("---")
    st.caption("Built with Streamlit • Scikit-learn • Joblib")

# -------------------------------
# MAIN UI
# -------------------------------
st.title("📰 Fake News Detector AI")

st.markdown("""
### AI-powered fake news classification system  
Check whether a news article is real or fake using machine learning.
""")

st.markdown("---")

# Session state
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

sample_text = (
    "Breaking: Local officials announce a new community program to improve city parks and schools. "
    "The initiative will provide grants and local support for environmental improvements."
)

with st.expander("📄 Sample article"):
    st.write(sample_text)
    if st.button("Use sample article"):
        st.session_state.news_text = sample_text
        st.rerun()

# Input box
news = st.text_area(
    "✍️ Paste news article below",
    value=st.session_state.news_text,
    height=260,
    placeholder="Paste your news article here..."
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze News", use_container_width=True)

with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state.news_text = ""
    st.rerun()

# -------------------------------
# PREDICTION LOGIC (SAFE)
# -------------------------------
if analyze:
    st.session_state.news_text = news

    if not news.strip():
        st.warning("Please enter a news article before analyzing.")

    elif model is None or vectorizer is None:
        st.error("Backend model not ready yet. Waiting for Omar’s training output.")

    else:
        with st.spinner("Analyzing article..."):
            features = vectorizer.transform([news])
            prediction = model.predict(features)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    confidence = max(model.predict_proba(features)[0]) * 100
                except:
                    confidence = None

        st.markdown("## 🤖 Prediction Result")

        if prediction == 1:
            st.success("🟢 REAL NEWS")
            st.markdown("This article appears credible based on AI analysis.")
        else:
            st.error("🔴 FAKE NEWS")
            st.markdown("This article shows patterns of misinformation.")

        if confidence is not None:
            st.info(f"Confidence: {confidence:.1f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#c084fc;'>AI Fake News Detector Project</div>",
    unsafe_allow_html=True
)