# scripts/lda_qda_text.py
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline as SkPipeline  # for the projection plots

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- Load dataset from the existing CSV ---
data_path = Path(__file__).resolve().parents[1] / "data" / "sms_spam.csv"
if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. "
        "Run your dataset download script first (e.g. scripts/download_dataset.py)."
    )

df = pd.read_csv(data_path)
X_text = df["text"].astype(str)
# encode label: spam -> 1, ham -> 0
y = (df["label"].astype(str).str.lower().str.strip() == "spam").astype(int).to_numpy()

# --- 1) Train/test split ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)
print("Class balance (train/test):", Counter(y_train), Counter(y_test))

# --- 2) TF-IDF -> SVD -> Standardize -----------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2),
)
svd = TruncatedSVD(n_components=200, random_state=42)
scaler = StandardScaler()

# --- 3) LDA pipeline with SMOTE ----------------------------------------------
lda_clf = Pipeline(steps=[
    ("tfidf", tfidf),
    ("svd", svd),
    ("scaler", scaler),
    ("smote", SMOTE(random_state=42)),
    ("lda", LinearDiscriminantAnalysis())
])

lda_clf.fit(X_train, y_train)
y_pred_lda = lda_clf.predict(X_test)
print("\n=== LDA report ===")
print(classification_report(y_test, y_pred_lda, digits=3))
print(confusion_matrix(y_test, y_pred_lda))

# --- 4) QDA pipeline with SMOTE ----------------------------------------------
qda_clf = Pipeline(steps=[
    ("tfidf", tfidf),
    ("svd", svd),
    ("scaler", scaler),
    ("smote", SMOTE(random_state=42)),
    ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1))
])

qda_clf.fit(X_train, y_train)
y_pred_qda = qda_clf.predict(X_test)
print("\n=== QDA report ===")
print(classification_report(y_test, y_pred_qda, digits=3))
print(confusion_matrix(y_test, y_pred_qda))

# --- 5) Visualize the 1D LDA projection as histograms ------------------------
outdir = Path("outputs/figures"); outdir.mkdir(parents=True, exist_ok=True)

num_transform = SkPipeline(steps=[("tfidf", tfidf), ("svd", svd), ("scaler", scaler)])
X_train_num = num_transform.fit_transform(X_train, y_train)  # fit on train only
X_test_num  = num_transform.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train_num, y_train)
z_train = lda.transform(X_train_num).ravel()
z_test  = lda.transform(X_test_num).ravel()

plt.figure(figsize=(7,4))
plt.hist(z_train[y_train==0], bins=30, alpha=0.6, label="ham (train)", density=True)
plt.hist(z_train[y_train==1], bins=30, alpha=0.6, label="spam (train)", density=True)
plt.title("LDA 1D projection (train)")
plt.xlabel("LDA component 1"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
plt.savefig(outdir / "lda_train_hist.png", dpi=150)

plt.figure(figsize=(7,4))
plt.hist(z_test[y_test==0], bins=30, alpha=0.6, label="ham (test)", density=True)
plt.hist(z_test[y_test==1], bins=30, alpha=0.6, label="spam (test)", density=True)
plt.title("LDA 1D projection (test)")
plt.xlabel("LDA component 1"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
plt.savefig(outdir / "lda_test_hist.png", dpi=150)

print("Saved:")
print(" - outputs/figures/lda_train_hist.png")
print(" - outputs/figures/lda_test_hist.png")
