
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ============================================================
# 1. LOAD OLD DATASET
# ============================================================

old_data = pd.read_csv(
    "dataset/spam.csv/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Old dataset loaded!")
print("Old messages:", len(old_data))


# ============================================================
# 2. LOAD NEW DATASET
# ============================================================

new_data = pd.read_csv(
    "dataset/new_spam.csv/combined_dataset.csv"
)

print("New dataset loaded!")
print("New messages:", len(new_data))


# ============================================================
# 3. MAKE SURE NEW DATASET HAS CORRECT COLUMNS
# ============================================================

new_data = new_data[["label", "message"]]


# ============================================================
# 4. COMBINE OLD + NEW DATASETS
# ============================================================

data = pd.concat(
    [old_data, new_data],
    ignore_index=True
)

print("\n===================================")
print("DATASETS COMBINED")
print("===================================")

print("Total messages:", len(data))


# ============================================================
# 5. REMOVE DUPLICATE MESSAGES
# ============================================================

before = len(data)

data = data.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

after = len(data)

print("Duplicates removed:", before - after)
print("Final messages:", after)


# ============================================================
# 6. CHECK LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(data["label"].value_counts())


# ============================================================
# 7. CLEAN LABELS
# ============================================================

data["label"] = data["label"].astype(str).str.lower().str.strip()

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# Remove rows with unknown labels
data = data.dropna(subset=["label"])

data["label"] = data["label"].astype(int)


# ============================================================
# 8. SEPARATE MESSAGES AND LABELS
# ============================================================

X = data["message"].astype(str)
y = data["label"]


# ============================================================
# 9. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 10. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = MultinomialNB()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


# ============================================================
# 16. SUCCESS MESSAGE
# ============================================================

print("\n===================================")
print("FILES SAVED SUCCESSFULLY")
print("===================================")

print(
    "Model: model/spam_model.pkl"
)

print(
    "Vectorizer: model/tfidf_vectorizer.pkl"
)

print(
    "\nYour model is now trained using BOTH datasets!"
)

