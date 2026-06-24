import streamlit as st
import joblib
from pathlib import Path

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)

# -------------------------------
# DARK + PURPLE THEME (POLISHED)
# -------------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background: radial-gradient(circle at top, #1a0f2e, #0b0715);
    color: white;
}

/* Title */
h1 {
    color: #c084fc;
    text-align: center;
    font-weight: 800;
}

/* Subheaders */
h2, h3 {
    color: #e0b3ff;
}

/* Text area */
textarea {
    background-color: #1b102f !important;
    color: white !important;
    border: 1px solid #c084fc !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #7c3aed, #c084fc);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #6d28d9, #a855f7);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #120a22;
}

/* Result cards */
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.real {
    background-color: #14532d;
    border: 1px solid #22c55e;
}

.fake {
    background-color: #7f1d1d;
    border: 1px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("📰 Fake News Detector AI")
st.subheader("Advanced Machine Learning News Classifier")

st.markdown("""
This AI system detects whether a news article is **REAL** or **FAKE** using NLP and Machine Learning.

Paste any article below and let the AI analyze it.
""")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
**Fake News Detector AI**

AI-based classification system for detecting misinformation.

**Tech Stack:**
- Python
- Streamlit
- Scikit-learn
- NLP

**Team:**
- Model Development
- UI & Deployment
""")

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = Path("models/model.pkl")
vectorizer_path = Path("models/vectorizer.pkl")

model = None
vectorizer = None

if model_path.exists() and vectorizer_path.exists():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

# -------------------------------
# INPUT
# -------------------------------
st.markdown("## ✍️ Enter News Article")

news = st.text_area(
    "",
    height=250,
    placeholder="Paste a news article here..."
)

# -------------------------------
# BUTTONS
# -------------------------------
col1, col2 = st.columns(2)

analyze = col1.button("🔍 Analyze News")
clear = col2.button("🧹 Clear")

if clear:
    st.rerun()

# -------------------------------
# ANALYSIS
# -------------------------------
if analyze:

    if news.strip() == "":
        st.warning("Please enter a news article first.")

    elif model is None or vectorizer is None:
        st.info("Model not available yet. Waiting for your teammate.")

    else:
        with st.spinner("AI is analyzing the article..."):
            transformed = vectorizer.transform([news])
            prediction = model.predict(transformed)[0]

        st.markdown("## 🤖 Prediction Result")

        if prediction == 1:
            st.markdown("""
            <div class="result-box real">
                <h2>🟢 REAL NEWS</h2>
                <p>The article appears credible based on AI analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box fake">
                <h2>🔴 FAKE NEWS</h2>
                <p>This article shows patterns of misinformation.</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#c084fc; font-size:14px'>
🎓 AI Group Project | Fake News Detector | Built with Streamlit
</div>
""", unsafe_allow_html=True)