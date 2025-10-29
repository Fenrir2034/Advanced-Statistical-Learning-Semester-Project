#!/usr/bin/env python3
"""
Visualize QDA 'projection' (log-odds score) before and after SMOTE on SMS spam data.
Outputs:
  outputs/figures/qda_original.png
  outputs/figures/qda_smote.png
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import matplotlib as plt
plt.use("Agg")
# ---------------------------
# Settings / paths
# ---------------------------
DATA_CSV = Path("data/sms_spam.csv")
OUT_DIR = Path("outputs/figures"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
SVD_DIM = 50          # reduce TF-IDF to something QDA can handle
QDA_REG = 0.05        # slight shrinkage for numerical stability
JITTER = 0.01         # vertical jitter so points don't overlap

# ---------------------------
# Helpers
# ---------------------------
def load_sms(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {"label","text"}.issubset(df.columns), "CSV must have columns: label,text"
    y = df["label"]
    if y.dtype == object:
        y = y.map({"ham":0, "spam":1}).astype(int)
    return df["text"], y.values

def build_text_svd():
    """TF-IDF (1–2 grams) -> TruncatedSVD -> Standardize."""
    tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                            ngram_range=(1,2), min_df=2, max_df=0.95)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    std = StandardScaler(with_mean=False)  # SVD is dense but still ok to use with_mean=False
    return make_pipeline(tfidf, svd, std)

def qda_scores(X_lowdim, y):
    """Fit QDA and return signed log-odds scores for class 1 (spam)."""
    qda = QuadraticDiscriminantAnalysis(reg_param=QDA_REG)
    qda.fit(X_lowdim, y)
    # Use predict_proba → log-odds as a 1D score
    p1 = qda.predict_proba(X_lowdim)[:,1]
    eps = 1e-9
    scores = np.log((p1 + eps) / (1 - p1 + eps))
    return scores, qda

def scatter_1d(scores, y, title, outfile):
    x = scores
    # place ham and spam around y=0 with tiny vertical jitter
    y_pos = np.zeros_like(x, dtype=float)
    y_pos[y==0] = 0.00 + (np.random.RandomState(SEED).rand(np.sum(y==0))-0.5)*JITTER
    y_pos[y==1] = 0.02 + (np.random.RandomState(SEED+1).rand(np.sum(y==1))-0.5)*JITTER

    plt.figure(figsize=(6,4))
    plt.scatter(x[y==0], y_pos[y==0], s=10, alpha=0.6, label="ham", c="blue")
    plt.scatter(x[y==1], y_pos[y==1], s=10, alpha=0.6, label="spam", c="red")
    plt.title(title)
    plt.xlabel("QDA log-odds score (spam vs ham)")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    X_text, y = load_sms(DATA_CSV)

    # 1) Original data
    text_svd = build_text_svd()
    X_low = text_svd.fit_transform(X_text)
    scores, _ = qda_scores(X_low, y)
    scatter_1d(scores, y,
               "QDA: Quadratic Discriminant 'projection' (original data)",
               OUT_DIR / "qda_original.png")

    # 2) SMOTEd data (balance classes in low-dim space)
    smote = SMOTE(random_state=SEED)
    X_low_sm, y_sm = smote.fit_resample(X_low, y)
    scores_sm, _ = qda_scores(X_low_sm, y_sm)
    scatter_1d(scores_sm, y_sm,
               "QDA Projection of SMS Data (SMOTEd data)",
               OUT_DIR / "qda_smote.png")

    print("✅ Saved:")
    print(" -", OUT_DIR / "qda_original.png")
    print(" -", OUT_DIR / "qda_smote.png")

if __name__ == "__main__":
    main()
