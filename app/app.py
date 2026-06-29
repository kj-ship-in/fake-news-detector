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
# DARK PURPLE THEME
# -------------------------------
st.markdown("""
<style>

/* Background */
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

/* Headers */
h2, h3 {
    color: #e0b3ff;
}

/* Text Area */
textarea {
    background-color: #1b102f !important;
    color: white !important;
    border: 1px solid #c084fc !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg,#7c3aed,#c084fc);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(90deg,#6d28d9,#a855f7);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color:#120a22;
}

/* Result Cards */
.result-box{
    padding:20px;
    border-radius:12px;
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

# -------------------------------
# TITLE
# -------------------------------
st.title("📰 Fake News Detector AI")
st.subheader("Advanced Machine Learning News Classifier")

st.markdown("""
This AI system detects whether a news article is **REAL** or **FAKE**
using Natural Language Processing and Machine Learning.

Paste a news article below and click **Analyze News**.
""")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 Project Information")
st.sidebar.markdown("---")

st.sidebar.success("Current Model: Logistic Regression + TF-IDF")

st.sidebar.write("Dataset: Kaggle Fake and Real News Dataset")

st.sidebar.info("""
### Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Joblib
- Pandas
""")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"


@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


model, vectorizer = load_assets()

# Show model status AFTER loading
if model is not None:
    st.success("✅ AI Model Loaded Successfully")
else:
    st.error("❌ Model could not be loaded")

# -------------------------------
# INPUT
# -------------------------------
st.markdown("## ✍️ Enter News Article")

news = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste a news article here...",
    label_visibility="collapsed"
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
        st.error("Model files are missing.")

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
<div style="text-align:center;color:#c084fc;font-size:14px;">

🎓 AI Group Project • Fake News Detector • Built with Streamlit

</div>
""", unsafe_allow_html=True)