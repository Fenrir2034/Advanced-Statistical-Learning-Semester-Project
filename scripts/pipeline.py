#!/usr/bin/env python3
"""
SMS Spam Classification — condensed project skeleton
----------------------------------------------------
Shows the essential workflow with small code snippets.
Each block represents what the real scripts in your repo do.
"""

# === 0. Imports ===============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
import joblib

# 1) env ----------------------------------------------------------------
# (see setup_and_run.sh for full details)

# 1. download_dataset.py
import zipfile, io, urllib.request, pandas as pd, pathlib

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip" # UCI SMS Spam Collection
pathlib.Path("data").mkdir(parents=True, exist_ok=True) # ensure data/ exists
zip_bytes = urllib.request.urlopen(url).read() # download zip
zf = zipfile.ZipFile(io.BytesIO(zip_bytes)) # read zip from bytes
with zf.open("SMSSpamCollection") as f: # extract file
    df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="utf-8") # load into DataFrame
df.to_csv("data/sms_spam.csv", index=False) # save as CSV
print("Saved data/sms_spam.csv") # confirm

# ==============================================================================

# === 2. Data Loading and Splitting ===========================================
# (setup_and_run.sh ensures this CSV exists)
#df = pd.read_csv("data/sms_spam.csv")
df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1}) # encode labels

# stratified 80/20 split (keeps spam/ham ratio)
X_train, X_test, y_train, y_test = train_test_split( # train-test split
    df['text'], df['label'], # features and labels
    test_size=0.2, stratify=df['label'], random_state=42 # stratify and reproducibility
)

# ==============================================================================

# === 3. Preprocessing Pipeline ===============================================
# TF–IDF: turn text → numerical features (1–2 grams)
# otherwise they may overwrite each other's fitted state. Use small helpers to create fresh ones.

def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, # filter rare/common
                           lowercase=True, stop_words="english") # text normalization

def make_svd():
    return TruncatedSVD(n_components=200, random_state=42) # we set random_state for reproducibility, a random state is when a random process is made deterministic by initializing it with a fixed value

def make_scaler():
    return StandardScaler() # default settings, mean=0, variance=1, we set these as such because many models assume standardized data

# SMOTE = Synthetic Minority Oversampling Technique
smote = SMOTE(random_state=42) # we set random_state for reproducibility, a random state is when a random process is made deterministic by initializing it with a fixed value

# Example: preprocessing + model joined into one pipeline
# NOTE (FIX): Keep LDA pipeline self-contained with its own TF-IDF/SVD/Scaler
lda_pipe = Pipeline([ # create a pipeline object
    ("tfidf", make_tfidf()), # convert text to TF-IDF features
    ("svd", make_svd()), # reduce dimensionality with SVD
    ("scaler", make_scaler()), # standardize features
    ("smote", smote), # apply SMOTE to balance classes
    ("lda", LinearDiscriminantAnalysis()) # final model: LDA
])

# ==============================================================================

# === 4. Train a few representative models ====================================
print("Training LDA, QDA, and SVM (RBF)...") # notify start of training

lda_pipe.fit(X_train, y_train) # fit the pipeline on training data

# NOTE (FIX): Define QDA as its OWN pipeline instead of mutating LDA with set_params.
qda_pipe = Pipeline([
    ("tfidf", make_tfidf()),
    ("svd", make_svd()),
    ("scaler", make_scaler()),
    ("smote", smote),
    ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1)) # QDA with regularization
])
qda_pipe.fit(X_train, y_train) # fit QDA pipeline

# SVM (RBF kernel) with grid search for C and gamma
# NOTE (OK): For RBF SVM we skip SVD/Scaler because SVC handles sparse TF-IDF well and we tune (C, gamma)
svm_pipe = Pipeline([   # create a pipeline object
    ("tfidf", make_tfidf()), # convert text to TF-IDF features 
    #("svd", make_svd()), # reduce dimensionality with SVD (optional for RBF)
    #("scaler", make_scaler()), # standardize features (optional for RBF on TF-IDF)
    ("smote", smote), # apply SMOTE to balance classes
    ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")) # final model: SVM with RBF kernel
])

param_grid = {"svm__C": [0.1, 1, 10], "svm__gamma": [1e-3, 1e-2, 1e-1]} # hyperparameter grid
grid = GridSearchCV(svm_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1) # grid search with 5-fold CV, we use 5-fold because it balances bias and variance well for model evaluation
grid.fit(X_train, y_train) # fit grid search
best_svm = grid.best_estimator_ # best model from grid search

# Calibrated Linear SVM (adds probability calibration)
lin_svm = Pipeline([
    ("tfidf", make_tfidf()), # vectorize text
    ("cal", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42))) # base linear SVM with calibration
])
lin_svm.fit(X_train, y_train) # fit on raw text; pipeline handles TF-IDF

# Random Forest ensemble baseline
rf = Pipeline([
    ("tfidf", make_tfidf()), # vectorize text
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)) # 200 trees
])
rf.fit(X_train, y_train) # fit on raw text; pipeline handles TF-IDF

# Save models (like train.py)
Path("outputs/models").mkdir(parents=True, exist_ok=True) # ensure output dir exists
for name, model in {"lda": lda_pipe, "qda": qda_pipe, "svm_rbf": best_svm,  # save each model
                    "lin_svm_cal": lin_svm, "rf": rf}.items(): # model name and object
    joblib.dump(model, f"outputs/models/{name}.joblib") # save model
    print(f"Saved outputs/models/{name}.joblib") # confirm saving

# ==============================================================================

# === 5. Evaluation on Test Split =============================================
def evaluate(model, X_test, y_test, name):  # here we define a function to evaluate models        
    """Compute core metrics and save a confusion matrix.""" 
    y_pred = model.predict(X_test)  # get predictions (raw text OK: pipelines vectorize internally)
    # Some models have predict_proba; some need decision_function
    if hasattr(model, "predict_proba"):  # check if model has predict_proba method
        y_score = model.predict_proba(X_test)[:, 1]  # get probabilities for positive class
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)  # get decision scores
    else:
        # Fallback if no scores are available (rare here)
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)  # accuracy
    f1 = f1_score(y_test, y_pred)  # F1 score
    # Guard: if y_score is just labels (0/1), skip AUC
    auc = roc_auc_score(y_test, y_score) if np.unique(y_score).size > 2 else np.nan  # AUC-ROC
    cm = confusion_matrix(y_test, y_pred)  # confusion matrix

    # --- FIXED PRINT BLOCK ---
    if np.isnan(auc):
        auc_str = "nan"
    else:
        auc_str = f"{auc:.3f}"

    print(f"{name}: acc={acc:.3f}, f1={f1:.3f}, auc={auc_str}")  # print metrics
    print(cm)   # print confusion matrix
    return acc, f1, auc, cm  # return metrics

# ==============================================================================

# === 6. Bootstrap Example (uncertainty estimation) ===========================
# Simplified version of bootstrap_eval.py
def bootstrap_metric(model, X, y, n_boot=100): # we define a function for bootstrap evaluation
    """Resample the test set and estimate variability of F1.""" 
    rng = np.random.default_rng(42) # reproducible random number generator
    f1s = [] # list to store F1 scores
    for _ in range(n_boot): # for each bootstrap iteration
        idx = rng.choice(len(y), size=len(y), replace=True) # resample indices with replacement
        # NOTE: X and y are pandas Series; iloc keeps alignment
        f1s.append(f1_score(y.iloc[idx], model.predict(X.iloc[idx]))) # compute F1 on resampled data
    return np.mean(f1s), np.percentile(f1s, [2.5, 97.5]) # return mean and 95% CI

mean_f1, ci = bootstrap_metric(best_svm, X_test, y_test, n_boot=200) # bootstrap on best SVM
print(f"SVM bootstrap F1 ≈ {mean_f1:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f})") # print bootstrap results

# ==============================================================================

# === 7. Reproducibility =============================================
# - All random processes use fixed random_state=42.
# - Pipelines ensure identical preprocessing during training and inference.
# - setup_and_run.sh would:
#     (1) create the env,
#     (2) download dataset if missing,
#     (3) run this script end-to-end,
#     (4) collect results in outputs/.

print("\nPipeline complete — models trained, evaluated, and saved.")
