
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "news_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check required columns
required_columns = {"text", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain {required_columns}. "
        f"Found: {df.columns.tolist()}"
    )

# Keep useful columns and clean missing values
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text"])

# Convert labels to integers
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("Dataset shape:", df.shape)
print("Label counts:")
print(df["label"].value_counts())

if df["label"].nunique() < 2:
    raise ValueError("The dataset must contain at least two classes.")

X = df["text"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Text vectorization + classification in one pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)

print("\nTest accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)
