import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Title and Description
# -----------------------------
st.title("📰 Fake News Detector")

st.markdown("""
Welcome!

Paste a news article into the box below and click **Predict**.
Once the trained AI model is connected, the application will determine whether the news is likely **Fake** or **Real**.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.info(
    "This application is part of an AI group project. "
    "It uses machine learning to classify news articles."
)

# -----------------------------
# User Input
# -----------------------------
news_text = st.text_area(
    "Enter News Article",
    placeholder="Paste the news article here..."
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        st.info(
            "The trained AI model has not been connected yet.\n\n"
            "When available, this application will display whether "
            "the news is Fake or Real."
        )