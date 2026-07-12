import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Load Dataset
# =========================

true_news = pd.read_csv("Dataset/True.csv/True.csv")
fake_news = pd.read_csv("Dataset/Fake.csv/Fake.csv")

# =========================
# Add Labels
# =========================

# 1 = Real News
# 0 = Fake News

true_news["label"] = 1
fake_news["label"] = 0


# =========================
# Combine Dataset
# =========================

data = pd.concat(
    [true_news, fake_news],
    ignore_index=True
)


# Keep only text and label

data = data[["text", "label"]]


# Remove empty rows

data.dropna(inplace=True)


# =========================
# Features and Labels
# =========================

X = data["text"]
y = data["label"]


# =========================
# TF-IDF Vectorization
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X)


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Train Model
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# =========================
# Check Accuracy
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("==============================")
print("Model Accuracy:", accuracy)
print("==============================")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================
# Save Model
# =========================

joblib.dump(
    model,
    "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


print("\n✅ Model trained and saved successfully!")