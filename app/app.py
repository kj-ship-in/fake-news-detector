import streamlit as st
import joblib
from pathlib import Path

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------
# PATHS
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

# ----------------------------------
# DARK PURPLE THEME
# ----------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: radial-gradient(circle at top,#1a0f2e,#0b0715);
    color:white;
}

/* Titles */
h1{
    color:#c084fc;
    text-align:center;
    font-weight:800;
}

h2,h3,h4{
    color:#e0b3ff;
}

/* Text Area */
textarea{
    background:#1b102f !important;
    color:white !important;
    border:1px solid #c084fc !important;
    border-radius:10px !important;
}

/* Buttons */
.stButton button{
    background:linear-gradient(90deg,#7c3aed,#c084fc);
    color:white;
    border:none;
    border-radius:12px;
    padding:0.6rem 1rem;
    font-weight:bold;
}

.stButton button:hover{
    background:linear-gradient(90deg,#6d28d9,#a855f7);
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:#120a22;
}

/* Result Cards */
.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
}

.real{
    background:#14532d;
    border:2px solid #22c55e;
}

.fake{
    background:#7f1d1d;
    border:2px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception:
        return None, None

model, vectorizer = load_assets()

# ----------------------------------
# SIDEBAR
# ----------------------------------
with st.sidebar:

    st.title("📌 Project Information")

    st.markdown("---")

    st.success("Model: Logistic Regression + TF-IDF")

    st.info("""
### Technologies

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Joblib
- Pandas
""")

    st.markdown("### How to Use")

    st.markdown("""
1. Paste a news article.

2. Click **Analyze News**.

3. View prediction results.
""")

    st.markdown("---")

    if model is None:
        st.error("❌ Model not loaded")
    else:
        st.success("✅ Model Loaded Successfully")

# ----------------------------------
# MAIN TITLE
# ----------------------------------
st.title("📰 Fake News Detector AI")

st.markdown("""
AI-powered fake news classification using Machine Learning and NLP.

Paste a news article below and click **Analyze News**.
""")

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

sample_article = (
    "Breaking: Local officials announce a new community program "
    "to improve parks and schools. The initiative will provide "
    "grants for environmental improvements."
)

# ----------------------------------
# SAMPLE ARTICLE
# ----------------------------------
with st.expander("📄 Sample Article"):

    st.write(sample_article)

    if st.button("Use Sample Article"):
        st.session_state.news_text = sample_article
        st.rerun()

# ----------------------------------
# INPUT
# ----------------------------------
news = st.text_area(
    "News Article",
    value=st.session_state.news_text,
    height=260,
    placeholder="Paste your news article here..."
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze News", use_container_width=True)

with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state.news_text = ""
    st.rerun()

# ----------------------------------
# ANALYSIS
# ----------------------------------
if analyze:

    st.session_state.news_text = news

    if news.strip() == "":
        st.warning("Please paste a news article.")

    elif model is None or vectorizer is None:
        st.error("Model files not found.")

    else:

        with st.spinner("Analyzing article..."):

            transformed = vectorizer.transform([news])

            prediction = model.predict(transformed)[0]

            confidence = None
            fake_prob = None
            real_prob = None

            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(transformed)[0]

                fake_prob = probs[0] * 100
                real_prob = probs[1] * 100

                confidence = max(fake_prob, real_prob)

        st.markdown("## 🤖 Prediction")

        if prediction == 1:

            st.markdown("""
<div class="result-box real">
<h2>🟢 REAL NEWS</h2>
<p>The article appears to be credible.</p>
</div>
""", unsafe_allow_html=True)

        else:

            st.markdown("""
<div class="result-box fake">
<h2>🔴 FAKE NEWS</h2>
<p>The article appears to contain misinformation.</p>
</div>
""", unsafe_allow_html=True)

        # -----------------------------
        # Confidence
        # -----------------------------
        if confidence is not None:

            st.markdown("### 📊 Confidence Score")

            st.progress(confidence / 100)

            st.write(f"**Confidence:** {confidence:.2f}%")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Fake Probability", f"{fake_prob:.2f}%")

            with c2:
                st.metric("Real Probability", f"{real_prob:.2f}%")

        # -----------------------------
        # Statistics
        # -----------------------------
        st.markdown("### 📈 Article Statistics")

        words = len(news.split())
        chars = len(news)
        reading = max(1, words // 200)

        a, b, c = st.columns(3)

        a.metric("Words", words)
        b.metric("Characters", chars)
        c.metric("Reading Time", f"{reading} min")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#c084fc;">

🎓 AI Fake News Detector • Built with Streamlit

</div>
""", unsafe_allow_html=True)