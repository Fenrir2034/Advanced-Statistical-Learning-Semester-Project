#!/usr/bin/env python3
"""
Generate confusion matrices for LDA and QDA using the SAME text-preprocessing
stack (TF-IDF -> TruncatedSVD -> StandardScaler) as the other scripts.
Outputs go to: <repo-root>/outputs/figures/
"""

import sys
from pathlib import Path    # repo root

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
DATA_PATH = ROOT / "data" / "sms_spam.csv"  # data path
FIG_DIR = ROOT / "outputs" / "figures" # figures output dir
FIG_DIR.mkdir(parents=True, exist_ok=True) # ensure output dir exists

# ---------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH) # load SMS spam dataset
df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int) # map labels to 0 and 1

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
# shared preprocessing: TF-IDF -> SVD -> scaler
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer( # TF-IDF vectorizer: we need this because LDA/QDA need numeric input and we have text
    lowercase=True, # make lowercase
    strip_accents="unicode", # strip accents
    ngram_range=(1, 2), # unigrams + bigrams: that means that we consider single words and pairs of consecutive words
    min_df=2, # ignore very rare words (appear in less than 2 documents)
    max_df=0.95, # ignore very common words (appear in more than 95% of documents)
)
svd = TruncatedSVD(n_components=200, random_state=42) # SVD to reduce dimensionality: that means we reduce the number of features to 200
scaler = StandardScaler() # standard scaler to normalize features: normalization is when we make sure that all features have similar scale

# fit on train, apply on train+test
X_train_tfidf = vectorizer.fit_transform(X_train) # fit and transform train data
X_test_tfidf = vectorizer.transform(X_test) # transform test data

X_train_svd = svd.fit_transform(X_train_tfidf) # fit and transform train data
X_test_svd = svd.transform(X_test_tfidf) # transform test data

X_train_num = scaler.fit_transform(X_train_svd) # fit and transform train data
X_test_num = scaler.transform(X_test_svd) # transform test data
# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------
lda = LinearDiscriminantAnalysis()  # LDA model
qda = QuadraticDiscriminantAnalysis(reg_param=0.1) # QDA model with regularization

lda.fit(X_train_num, y_train) # fit LDA
qda.fit(X_train_num, y_train)   # fit QDA
 
# ---------------------------------------------------------------------
# confusion matrices
# ---------------------------------------------------------------------
models = [ # list of models to evaluate
    ("lda", lda), 
    ("qda", qda),
]

for name, model in models: # for each model
    y_pred = model.predict(X_test_num) # get predictions
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]) # confusion matrix
    disp = ConfusionMatrixDisplay( # display the confusion matrix
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
