import pandas as pd
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

print(df[["text", "label"]].head())


# Features and labels
X = df["text"]
y = df["label"]


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

import joblib

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n Model saved successfully!")