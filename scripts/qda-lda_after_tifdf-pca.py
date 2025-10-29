# scripts/lda_qda_text.py
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- 0) Load your data --------------------------------------------------------
# Expect lists/arrays: texts (list[str]), labels (list[int] or np.array with {0,1})
# Replace this with your actual loader.
def load_sms_dataset(path="data/sms.tsv"):
    # example reader: class \t message per line
    labels, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lbl, msg = line.split("\t", 1)
            labels.append(1 if lbl.strip().lower()=="spam" else 0)
            texts.append(msg.strip())
    return texts, np.array(labels)

texts, y = load_sms_dataset("data/sms.tsv")

# --- 1) Train/test split ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, stratify=y, random_state=42
)
print("Class balance (train/test):", Counter(y_train), Counter(y_test))

# --- 2) Shared text -> numeric transform -------------------------------------
#   TF-IDF -> TruncatedSVD (LSA) -> Standardize
# Notes:
# - TruncatedSVD works on sparse; output is dense (good for SMOTE + QDA).
# - Keep n_components modest (e.g., 100–300) to avoid overfitting/memory blowups.

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",        # set None if you prefer raw
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2)           # try (1,1) first if small data
)
svd = TruncatedSVD(n_components=200, random_state=42)
scaler = StandardScaler()        # OK after SVD (dense)

# --- 3) LDA pipeline with SMOTE ----------------------------------------------
lda_clf = Pipeline(steps=[
    ("tfidf", tfidf),
    ("svd", svd),
    ("scaler", scaler),
    ("smote", SMOTE(random_state=42, n_jobs=1)),
    ("lda", LinearDiscriminantAnalysis())
])

lda_clf.fit(X_train, y_train)
y_pred_lda = lda_clf.predict(X_test)
print("\n=== LDA report ===")
print(classification_report(y_test, y_pred_lda, digits=3))
print(confusion_matrix(y_test, y_pred_lda))

# --- 4) QDA pipeline with SMOTE ----------------------------------------------
# QDA estimates a covariance per class; it can overfit. You may need a reg_param.
qda_clf = Pipeline(steps=[
    ("tfidf", tfidf),
    ("svd", svd),
    ("scaler", scaler),
    ("smote", SMOTE(random_state=42, n_jobs=1)),
    ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1))  # tune reg_param ∈ [0,1]
])

qda_clf.fit(X_train, y_train)
y_pred_qda = qda_clf.predict(X_test)
print("\n=== QDA report ===")
print(classification_report(y_test, y_pred_qda, digits=3))
print(confusion_matrix(y_test, y_pred_qda))

# --- 5) Visualize the 1D LDA projection as histograms ------------------------
# We re-use the fitted transform steps to get dense features, then project with LDA.
# (For 2 classes, n_components=1.)
outdir = Path("outputs/figures"); outdir.mkdir(parents=True, exist_ok=True)

# Get train features up to scaler (post-SVD, dense)
from sklearn.pipeline import Pipeline as SkPipeline
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
plt.xlabel("LDA component 1")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "lda_train_hist.png", dpi=150)

plt.figure(figsize=(7,4))
plt.hist(z_test[y_test==0], bins=30, alpha=0.6, label="ham (test)", density=True)
plt.hist(z_test[y_test==1], bins=30, alpha=0.6, label="spam (test)", density=True)
plt.title("LDA 1D projection (test)")
plt.xlabel("LDA component 1")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "lda_test_hist.png", dpi=150)

print("Saved:")
print(" - outputs/figures/lda_train_hist.png")
print(" - outputs/figures/lda_test_hist.png")
