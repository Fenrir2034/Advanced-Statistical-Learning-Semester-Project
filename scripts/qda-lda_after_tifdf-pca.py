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
# Always save under the project root, not the scripts directory
project_root = Path(__file__).resolve().parents[1]
outdir = project_root / "outputs" / "figures"
outdir.mkdir(parents=True, exist_ok=True)


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

# --- 6) QDA decision boundaries in 2D (via SVD to 2 comps) -------------------
# Fit transforms on TRAIN only to avoid leakage
svd2 = TruncatedSVD(n_components=2, random_state=42)
num2 = SkPipeline(steps=[("tfidf", tfidf), ("svd2", svd2)])

X2_train = num2.fit_transform(X_train, y_train)
X2_test  = num2.transform(X_test)

qda2 = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda2.fit(X2_train, y_train)

def _plot_qda_boundary(X2, y, clf, path, title):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = X2[:,0].min() - 1.0, X2[:,0].max() + 1.0
    y_min, y_max = X2[:,1].min() - 1.0, X2[:,1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6.5,5))
    plt.contourf(xx, yy, Z, alpha=0.20)  # default cmap; no colors specified
    plt.scatter(X2[y==0,0], X2[y==0,1], s=10, label="ham")
    plt.scatter(X2[y==1,0], X2[y==1,1], s=10, label="spam")
    plt.title(title)
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

print("==> Generating QDA boundary plots...")
_plot_qda_boundary(X2_train, y_train, qda2, outdir / "qda_train_boundary.png",
                   "QDA decision regions (train, SVD-2D)")
_plot_qda_boundary(X2_test, y_test, qda2, outdir / "qda_test_boundary.png",
                   "QDA decision regions (test, SVD-2D)")

print("Saved:")
print(" - outputs/figures/qda_train_boundary.png")
print(" - outputs/figures/qda_test_boundary.png")

# --- 7) ROC & PR curves for LDA and QDA (on test split) ----------------------
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
import matplotlib.pyplot as plt

# LDA probabilities
proba_lda = lda_clf.predict_proba(X_test)[:, 1]
auc_lda = roc_auc_score(y_test, proba_lda)

RocCurveDisplay.from_predictions(y_test, proba_lda)
plt.title(f"LDA ROC (AUC = {auc_lda:.3f})")
plt.tight_layout()
plt.savefig(outdir / "lda_roc.png", dpi=150)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, proba_lda)
plt.title("LDA Precision–Recall")
plt.tight_layout()
plt.savefig(outdir / "lda_pr.png", dpi=150)
plt.close()

# QDA probabilities
proba_qda = qda_clf.predict_proba(X_test)[:, 1]
auc_qda = roc_auc_score(y_test, proba_qda)

RocCurveDisplay.from_predictions(y_test, proba_qda)
plt.title(f"QDA ROC (AUC = {auc_qda:.3f})")
plt.tight_layout()
plt.savefig(outdir / "qda_roc.png", dpi=150)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, proba_qda)
plt.title("QDA Precision–Recall")
plt.tight_layout()
plt.savefig(outdir / "qda_pr.png", dpi=150)
plt.close()

print("Saved:")
print(" - outputs/figures/lda_roc.png")
print(" - outputs/figures/lda_pr.png")
print(" - outputs/figures/qda_roc.png")
print(" - outputs/figures/qda_pr.png")

#!/usr/bin/env python3
# scripts/TF-IDF_SVM_RBF.py

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sms_spam.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = (df["label"].str.lower().str.strip() == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train/Test sizes: {len(X_train)}, {len(X_test)}")
print(f"Class balance (train): {y_train.value_counts().to_dict()}")

# --- Pipeline -----------------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2)
)
scaler = StandardScaler(with_mean=False)  # sparse safety
smote = SMOTE(random_state=42)
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

pipe = Pipeline(steps=[
    ("tfidf", tfidf),
    ("smote", smote),
    ("svm", svm)
])

param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": [1e-3, 1e-2, 1e-1]
}

grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")

# --- Evaluation ---------------------------------------------------------------
y_pred = grid.predict(X_test)
proba = grid.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, proba)

print("\n=== SVM (RBF) Report ===")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc:.4f}")

# --- Save figures -------------------------------------------------------------
RocCurveDisplay.from_predictions(y_test, proba)
plt.title(f"SVM (RBF) ROC Curve (AUC = {roc_auc:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "svm_rbf_roc.png", dpi=150)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, proba)
plt.title("SVM (RBF) Precision–Recall")
plt.tight_layout()
plt.savefig(FIG_DIR / "svm_rbf_pr.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("SVM (RBF) Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["ham","spam"]); ax.set_yticklabels(["ham","spam"])
for (i,j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "svm_rbf_cm.png", dpi=150)
plt.close()

print("\nSaved SVM figures:")
print(f" - {FIG_DIR / 'svm_rbf_roc.png'}")
print(f" - {FIG_DIR / 'svm_rbf_pr.png'}")
print(f" - {FIG_DIR / 'svm_rbf_cm.png'}")

# --- Save model & params ------------------------------------------------------
import joblib, json
joblib.dump(grid.best_estimator_, MODEL_DIR / "svm_rbf.joblib")
with open(MODEL_DIR / "svm_rbf_params.json", "w") as f:
    json.dump(grid.best_params_, f, indent=2)

print(f"Saved model to {MODEL_DIR / 'svm_rbf.joblib'}")
print("==> DONE.")
