#!/usr/bin/env python3
"""
Generate confusion matrices for LDA and QDA using the SAME text-preprocessing
stack (TF-IDF -> TruncatedSVD -> StandardScaler) as the other scripts.
Outputs go to: <repo-root>/outputs/figures/
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]     # repo root
DATA_PATH = ROOT / "data" / "sms_spam.csv"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
# shared preprocessing: TF-IDF -> SVD -> scaler
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)
svd = TruncatedSVD(n_components=200, random_state=42)
scaler = StandardScaler()

# fit on train, apply on train+test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

X_train_num = scaler.fit_transform(X_train_svd)
X_test_num = scaler.transform(X_test_svd)

# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

lda.fit(X_train_num, y_train)
qda.fit(X_train_num, y_train)

# ---------------------------------------------------------------------
# confusion matrices
# ---------------------------------------------------------------------
models = [
    ("lda", lda),
    ("qda", qda),
]

for name, model in models:
    y_pred = model.predict(X_test_num)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["ham", "spam"],
    )
    disp.plot(cmap="Purples", values_format="d", colorbar=False)
    plt.title(f"{name.upper()} confusion matrix (test set)")
    plt.tight_layout()
    out_path = FIG_DIR / f"{name}_cm.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

print("Saved LDA and QDA confusion matrices to:", FIG_DIR)
