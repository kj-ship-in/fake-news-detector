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

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

# -------------------------------
# STYLES
# -------------------------------
st.markdown(
    """
    <style>
    .stApp, .block-container, .main { background: radial-gradient(circle at top, #1a0f2e, #0b0715) !important; color: #f8fafc !important; }
    h1, h2, h3, h4, h5 { color: #c084fc !important; }
    .stTextArea textarea, textarea, .streamlit-expanderContent { background-color: #1b102f !important; color: #f8fafc !important; border: 1px solid #2d1250 !important; border-radius: 12px !important; }
    .stButton button { background: linear-gradient(90deg, #7c3aed, #c084fc) !important; color: white !important; border-radius: 12px !important; padding: 0.75rem 1rem !important; font-weight: 700 !important; }
    .stButton button:hover { transform: translateY(-1px) scale(1.02) !important; }
    [data-testid="stSidebar"] { background-color: #120a22 !important; color: #e9d5ff !important; }
    .block-container, .stMarkdown, .stWrite, .stText { background: transparent !important; color: #f8fafc !important; }
    .stExpander, .streamlit-expanderHeader, .streamlit-expanderContent, details { background: rgba(11,7,21,0.35) !important; color: #f8fafc !important; border-radius: 10px !important; }
    .result-box { padding: 20px; border-radius: 14px; text-align: center; margin-top: 12px; border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02); }
    .real { background-color: rgba(34,197,94,0.08) !important; }
    .fake { background-color: rgba(239,68,68,0.08) !important; }
    .stMetric, .stMetricValue, .stMetricDelta, .stMetricLabel { color: #f8fafc !important; background: transparent !important; }
    footer, .reportview-footer { background: transparent !important; color: #c084fc !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("📌 Project Info")
    st.write(
        "A polished fake news detector with a friendly UX and fast NLP predictions."
    )
    st.markdown("**How to use**")
    st.markdown(
        """
        1. Paste a news article or headline.
        2. Click **Analyze News**.
        3. Review the prediction and confidence score.
        """
    )
    st.markdown("---")

    if model is None or vectorizer is None:
        st.warning("Model files are missing or could not be loaded.")
        st.write("Ensure `models/model.pkl` and `models/vectorizer.pkl` exist.")
    else:
        st.success("Model loaded and ready to classify!")

    st.markdown("---")
    st.markdown(
        """
        **Best results:**
        - Paste at least one full paragraph.
        - Avoid headline-only text.
        - Use English articles for this version.
        """
    )
    st.caption("Built with Streamlit • Scikit-learn • Joblib")

# -------------------------------
# MAIN PAGE
# -------------------------------
st.title("📰 Fake News Detector AI")
st.markdown("### Polished interface for smarter news classification")

st.markdown(
    "Paste a news article below and get a fast AI-powered prediction for whether the content is likely real or fake."
)

st.markdown("---")

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

sample_text = (
    "Breaking: Local officials announce a new community program to improve city parks and schools. "
    "The initiative will provide grants and local support for environmental improvements."
)

with st.expander("Sample article"):
    st.write(sample_text)
    if st.button("Use sample article"):
        st.session_state.news_text = sample_text
        st.experimental_rerun()

news = st.text_area(
    "Enter article text for analysis",
    value=st.session_state.news_text,
    height=260,
    placeholder="Paste your news article here...",
)

col1, col2 = st.columns([2, 1])
with col1:
    analyze = st.button("🔍 Analyze News", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear text", use_container_width=True)

if clear:
    st.session_state.news_text = ""
    st.experimental_rerun()

if analyze:
    st.session_state.news_text = news
    if not news.strip():
        st.warning("Please enter a news article before analyzing.")
    elif model is None or vectorizer is None:
        st.error("The model is not ready. Check the `models` folder and reload the app.")
    else:
        with st.spinner("Analyzing the article..."):
            features = vectorizer.transform([news])
            prediction = model.predict(features)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    confidence = max(model.predict_proba(features)[0]) * 100
                except Exception:
                    confidence = None

        st.markdown("## 🤖 Prediction Result")

        if prediction == 1:
            st.markdown(
                """
                <div class="result-box real">
                    <h2>🟢 REAL NEWS</h2>
                    <p>The article appears credible based on AI analysis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-box fake">
                    <h2>🔴 FAKE NEWS</h2>
                    <p>This article shows patterns of misinformation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if confidence is not None:
            st.metric("Confidence", f"{confidence:.1f}%")

        with st.expander("Why this prediction matters"):
            st.write(
                "This tool is a helpful indicator, not a final fact checker. "
                "Verify suspicious articles with trusted sources and fact-checking websites."
            )

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#c084fc; font-size:14px'>
    🎓 AI Group Project | Fake News Detector | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
