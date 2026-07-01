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
    "",
    height=250,
    placeholder="Paste a news article here..."
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
        st.info("Model not available yet. Waiting for your teammate.")

    else:
        with st.spinner("AI is analyzing the article..."):
            transformed = vectorizer.transform([news])
            prediction = model.predict(transformed)[0]

            confidence = None
            fake_prob = None
            real_prob = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(transformed)[0]

                fake_prob = probabilities[0] * 100
                real_prob = probabilities[1] * 100

                confidence = max(fake_prob, real_prob)

        st.markdown("## 🤖 Prediction")

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

        # -----------------------
        # Confidence Score
        # -----------------------

        if confidence is not None:

            st.markdown("### 📊 Confidence Score")

            st.progress(confidence / 100)

            st.write(f"**Confidence:** {confidence:.2f}%")

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Fake Probability", f"{fake_prob:.2f}%")

            with col_b:
                st.metric("Real Probability", f"{real_prob:.2f}%")

        # -----------------------
        # Article Statistics
        # -----------------------

        st.markdown("### 📈 Article Statistics")

        word_count = len(news.split())
        char_count = len(news)
        reading_time = max(1, word_count // 200)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Words", word_count)

        with col2:
            st.metric("Characters", char_count)

        with col3:
            st.metric("Reading Time", f"{reading_time} min")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#c084fc; font-size:14px'>
🎓 AI Group Project | Fake News Detector | Built with Streamlit
</div>
""", unsafe_allow_html=True)