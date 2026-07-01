# 📰 Fake News Detector AI (Machine Learning Project)

A full-stack machine learning web application that detects whether a news article is **real or fake** using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression. The project includes a Streamlit frontend and a trained backend model, built collaboratively using Git branching.

---

# 🚀 Project Overview

This project is a collaborative AI system designed to classify news articles as **REAL or FAKE** using machine learning techniques.

It demonstrates:
- Natural Language Processing (NLP)
- Machine Learning model training
- Web application development using Streamlit
- Git/GitHub collaboration workflow (branching + merging)

---

# 👨‍💻 Team Collaboration

This project was developed by two contributors using separate branches:

## Frontend Developer: Kiera
Responsible for:
- Streamlit UI design
- User interaction interface
- Input handling and result display
- Styling and user experience

## Backend Developer: Omar
Responsible for:
- Dataset processing and cleaning
- Model training using Scikit-learn
- TF-IDF vectorization
- Logistic Regression model creation
- Generating model files (`model.pkl`, `vectorizer.pkl`)

---

# ⚙️ How the System Works

1. User enters a news article or headline in the UI
2. Text is preprocessed (cleaning, normalization)
3. TF-IDF vectorizer converts text into numerical features
4. Logistic Regression model predicts classification
5. Output is displayed as:
   - 🟢 REAL NEWS
   - 🔴 FAKE NEWS
6. Confidence score is shown (if available)

---

# 🧠 Machine Learning Pipeline

## 1. Dataset
- Fake news dataset: `Fake.csv`
- Real news dataset: `True.csv`

## 2. Data Preprocessing
- Lowercasing text
- Removing URLs
- Removing special characters
- Cleaning whitespace and noise

## 3. Feature Extraction
- TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency)
- Removes stopwords
- Limits noisy high-frequency terms

## 4. Model Training
- Algorithm: Logistic Regression
- Train/Test Split: 80/20
- Optimization: max_iter=1000

## 5. Output
- Binary classification:
  - 0 → Fake News
  - 1 → Real News

---

# 📁 Project Structure
fake-news-detector/
│
├── app/
│ └── app.py # Streamlit frontend application
│
├── data/
│ ├── Fake.csv # Fake news dataset
│ └── True.csv # Real news dataset
│
├── docs/
│ └── project_notes.txt # Documentation / notes
│
├── frontend/
│ └── (frontend-related files if any)
│
├── models/
│ ├── model.pkl # Trained ML model
│ └── vectorizer.pkl # TF-IDF vectorizer
│
├── train_model.py # Model training script
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

# ▶️ How to Run the Project

## 1. Clone Repository
```bash
git clone https://github.com/kj-ship-in/fake-news-detector.git
cd fake-news-detector

## 2. Install Dependencies
pip install -r requirements.txt

If needed manually:
pip install streamlit scikit-learn pandas numpy joblib

3. Run the Application

📦 Dependencies
streamlit
scikit-learn
pandas
numpy
joblib


🎯 Features
Clean and modern UI (Streamlit)
Real-time news prediction
Fake vs Real classification
Confidence score display
Styled result indicators
Expandable sample article section
Clear/reset input functionality
Team-based modular architecture


🧠 Model Details
Algorithm: Logistic Regression
Feature Extraction: TF-IDF Vectorizer
Input: News article text
Output: Binary classification (0 = Fake, 1 = Real)
Library: Scikit-learn

⚠️ Disclaimer

This project is built for educational purposes only.
The predictions are based on statistical patterns and should not be considered as factual verification.

📌 Future Improvements
Upgrade model to BERT or LSTM (Deep Learning)
Add FastAPI backend for API-based architecture
Deploy application (Render / HuggingFace / Streamlit Cloud)
Improve dataset size and accuracy
Add user authentication system
Add real-time news scraping feature


🔀 Git Workflow Summary

This project used professional Git collaboration:

Feature branching (frontend-kiera, backend)
Independent development
Merge conflict resolution
Final integration into main branch


🏁 Final Status

✔ Frontend completed (Streamlit UI)
✔ Backend completed (ML model trained)
✔ Dataset preprocessing completed
✔ Git collaboration successfully done
✔ Merge conflicts resolved
✔ Fully working integrated AI system


🎉 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline including:

Data preprocessing
NLP feature extraction
Model training
Web application deployment
Team collaboration using Git
Full system integration

It serves as a strong beginner-to-intermediate level AI project suitable for academic submission and portfolio presentation.




---

# 🚀 NEXT STEP (IMPORTANT)

After pasting this:

```bash
git add README.md
git commit -m "Final professional README"
git push origin main