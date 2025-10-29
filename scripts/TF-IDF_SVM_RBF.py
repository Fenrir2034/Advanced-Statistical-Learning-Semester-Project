# scripts/svm_text_classifier.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- Load dataset ---
data_path = Path("data/sms_spam.csv")
df = pd.read_csv(data_path)
X, y = df["text"].astype(str), df["label"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Pipeline ---
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95, min_df=2
    )),
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
])

# --- Hyperparameter tuning ---
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": [1e-3, 1e-2, 1e-1]
}
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

# --- Evaluate ---
y_pred = grid.predict(X_test)
print("\n=== SVM (RBF) report ===")
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))
