#!/usr/bin/env python3
"""
Bootstrap evaluation of SMS spam classifiers.
This script estimates model uncertainty via bootstrap resampling and refitting.
All paths and parameters are defined in the YAML config file.

Usage:
  python bootstrap_eval.py --config config/default.yaml
"""

import argparse # for command-line argument parsing
import sys # for system-specific parameters and functions
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
from pathlib import Path # for filesystem path manipulations
import yaml # for YAML file parsing
import random   # for random number generation
import os # for operating system interfaces
from sklearn.model_selection import train_test_split # for splitting data into train and test sets
from sklearn.feature_extraction.text import TfidfVectorizer # for text feature extraction
from sklearn.pipeline import Pipeline # for creating machine learning pipelines
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score # for evaluation metrics
)
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.svm import LinearSVC # for linear support vector machine model
from sklearn.calibration import CalibratedClassifierCV # for probability calibration
from sklearn.ensemble import RandomForestClassifier # for random forest model
from urllib.request import urlretrieve # for downloading files from URLs
# --- ensure repo root is on sys.path ---
from pathlib import Path    # to manage filesystem paths
import sys               # to manipulate Python runtime environment
ROOT = Path(__file__).resolve().parents[1] # assuming this script is in scripts/ subdir
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------
def load_config(): # load YAML config file from command-line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.") # add config argument
    args = ap.parse_args() # parse arguments
    with open(args.config, "r") as f: # open config file
        return yaml.safe_load(f) # load YAML content


CFG = load_config() # load configuration

PROJECT = CFG.get("project", {}) # get project-specific config section
DATA_CSV = Path(PROJECT.get("data_csv", "data/sms_spam.csv")) # path to data CSV
OUT_DIR = Path(PROJECT.get("out_dir", "outputs"))   # output directory
SEED = int(PROJECT.get("seed", 42)) # random seed for reproducibility
BOOT_ITERS_TEST = int(PROJECT.get("bootstrap_test_iters", 1000)) # bootstrap iterations for test set
BOOT_ITERS_REFIT = int(PROJECT.get("bootstrap_refit_iters", 200)) # bootstrap iterations for refitting
TEST_SIZE = float(PROJECT.get("test_size", 0.2))    # test set size fraction

(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True) # create output directories


np.random.seed(SEED) #  set numpy random seed for reproducibility
random.seed(SEED) # set python random seed for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def load_sms(path: Path = None) -> pd.DataFrame: # load SMS Spam dataset
    """
    Load SMS Spam dataset, either from config-specified CSV or download from UCI repository.
    """
    if path is None or not path.exists():
        print(f"Dataset not found at {path}, downloading UCI SMS Spam Collection...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = Path("data/smsspamcollection.zip")
        Path("data").mkdir(exist_ok=True, parents=True) # ensure data directory exists
        if not zip_path.exists():       # download zip if not already present
            urlretrieve(url, zip_path.as_posix()) # download the dataset
        import zipfile # import zipfile module
        with zipfile.ZipFile(zip_path, "r") as zf: # open the zip file
            with zf.open("SMSSpamCollection") as f: # read the specific file
                df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="utf-8") # load into DataFrame
    else:
        df = pd.read_csv(path) # load from specified CSV
        assert {"label", "text"} <= set(df.columns), "CSV must have columns: label,text" # check required columns

    # Map label to binary
    if df["label"].dtype == object: # if labels are strings
        df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int) # map to 0/1
    else: # if labels are already numeric
        df["label"] = df["label"].astype(int) # ensure integer type

    return df # return the loaded DataFrame


# ---------------------------------------------------------------------
# MODEL FACTORIES
# ---------------------------------------------------------------------
def make_pipelines(): # create machine learning pipelines
    """ Create model pipelines for SMS spam classification. """
    vectorizer = TfidfVectorizer( # TF-IDF vectorizer: this is done in order to make the data suitable for ML models
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),    # unigrams and bigrams
        min_df=2, max_df=0.95 # ignore very rare and very common terms
    )

    logit = Pipeline([ # logistic regression
        ("tfidf", vectorizer), # text vectorization
        ("clf", LogisticRegression(max_iter=5000, solver="liblinear", C=1.0)) # logistic regression classifier
    ])
    linsvm = Pipeline([ # linear SVM
        ("tfidf", vectorizer), # text vectorization
        ("clf", LinearSVC(C=1.0)) # linear SVM classifier
    ])
    linsvm_cal = Pipeline([ # calibrated linear SVM
        ("tfidf", vectorizer), # text vectorization
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=5, method="sigmoid")) # calibrated SVM
    ])
    rf = Pipeline([ # random forest
        ("tfidf", vectorizer), # text vectorization
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)) # random forest
    ])
    return {
        "logistic": logit,  # logistic regression
        "linear_svm": linsvm, # linear SVM
        "linear_svm_cal": linsvm_cal,   # calibrated linear SVM
        "random_forest": rf # random forest
    }


# ---------------------------------------------------------------------
# METRICS & HELPERS
# ---------------------------------------------------------------------
def _get_scores(y_true, y_pred, y_score=None): # compute evaluation metrics
    acc = accuracy_score(y_true, y_pred) # accuracy
    f1 = f1_score(y_true, y_pred, pos_label=1) # F1 score for positive class
    roc = ap = np.nan # initialize ROC AUC and Average Precision
    if y_score is not None: # if scores are provided
        try: 
            roc = roc_auc_score(y_true, y_score)    # ROC AUC
        except ValueError:
            pass
        try:
            ap = average_precision_score(y_true, y_score) # Average Precision
        except ValueError:
            pass
    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": ap} # return metrics as dictionary


def _predict_scores(model: Pipeline, X): # get predictions and scores from model
    if hasattr(model, "decision_function"): # if model has decision_function
        s = model.decision_function(X) # get decision scores
        y_score = 1 / (1 + np.exp(-s)) # convert to probabilities via sigmoid
    elif hasattr(model, "predict_proba"): # if model has predict_proba
        y_score = model.predict_proba(X)[:, 1] # get positive class probabilities
    else:
        y_score = None
    y_pred = model.predict(X) # get class predictions
    return y_pred, y_score


# ---------------------------------------------------------------------
# BOOTSTRAP LOGIC
# ---------------------------------------------------------------------
def bootstrap_testset_metrics(model, X_test, y_test, B=1000, random_state=42): # bootstrap on test set
    rng = np.random.RandomState(random_state) # random number generator
    metrics = [] # list to store metrics
    n = len(y_test) # number of test samples
    for _ in range(B): # for each bootstrap iteration
        idx = rng.randint(0, n, size=n) # sample indices with replacement
        y_true_b = y_test[idx] # true labels for bootstrap sample
        X_b = X_test.iloc[idx] # test features for bootstrap sample
        y_pred_b, y_score_b = _predict_scores(model, X_b) # get predictions and scores
        metrics.append(_get_scores(y_true_b, y_pred_b, y_score_b)) # compute and store metrics
    return pd.DataFrame(metrics) # return metrics as DataFrame


def bootstrap_refit(model_factory, X, y, test_size=0.2, B=200, random_state=42): # bootstrap with refitting
    rng = np.random.RandomState(random_state) # random number generator
    n = len(y) # number of samples
    metrics = []    # list to store metrics
    for _ in range(B): # for each bootstrap iteration
        train_idx = rng.randint(0, n, size=n) # sample training indices with replacement
        mask = np.ones(n, dtype=bool) # boolean mask for test indices
        mask[train_idx] = False # mark training indices as False
        test_idx = np.where(mask)[0] # get test indices
        if len(test_idx) < max(30, int(0.05 * n)): # if test set too small
            from sklearn.model_selection import train_test_split # import train_test_split
            tr_idx, te_idx = train_test_split(  # split data to ensure sufficient test size
                np.arange(n), test_size=test_size, stratify=y, #    stratified split: maintain class proportions: stratified is when we 
                                                                #ensure that the class proportions
                random_state=rng.randint(0, 10**9) # random state for reproducibility
            )
            train_idx, test_idx = tr_idx, te_idx # assign train and test indices
        X_tr, y_tr = X.iloc[train_idx], y[train_idx] # training data
        X_te, y_te = X.iloc[test_idx], y[test_idx] # test data
        model = model_factory() # create new model instance
        model.fit(X_tr, y_tr) # fit model on training data 
        y_pred, y_score = _predict_scores(model, X_te) # get predictions and scores on test data
        metrics.append(_get_scores(y_te, y_pred, y_score)) # compute and store metrics
    return pd.DataFrame(metrics) # return metrics as DataFrame


def ci_from_samples(samples, alpha=0.05): # compute confidence interval from bootstrap samples
    lo = np.percentile(samples, 100 * (alpha / 2)) # lower bound
    hi = np.percentile(samples, 100 * (1 - alpha / 2)) # upper bound
    return float(np.mean(samples)), float(lo), float(hi) # return mean and confidence interval


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(): # main function to run bootstrap evaluation
    print("Loading data...") # load dataset
    df = load_sms(DATA_CSV) # load SMS spam dataset
    X = df["text"] # features: text messages
    y = df["label"].values # labels: spam/ham

    X_tr, X_te, y_tr, y_te = train_test_split( # split into train and test sets
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED # stratified split for class balance
    )

    models = make_pipelines() # create model pipelines
    rows = [] # list to store summary rows

    for name, pipe in models.items(): # for each model
        print(f"\n=== {name.upper()} ===") # print model name
        pipe.fit(X_tr, y_tr) # fit model on training data
        y_pred, y_score = _predict_scores(pipe, X_te) # get predictions and scores on test data
        base = _get_scores(y_te, y_pred, y_score) # compute base metrics
        print("Holdout metrics:", base) # print holdout metrics

        # --- Bootstrap ---
        df_bs_test = bootstrap_testset_metrics(pipe, X_te, y_te, B=BOOT_ITERS_TEST, random_state=SEED) # bootstrap on test set
        df_bs_test.to_csv(OUT_DIR / f"{name}_bootstrap_testset.csv", index=False) # save test set bootstrap results

        df_bs_refit = bootstrap_refit(  # bootstrap with refitting
            lambda: make_pipelines()[name], X, y, # model factory
            test_size=TEST_SIZE, B=BOOT_ITERS_REFIT, random_state=SEED # parameters
        )
        df_bs_refit.to_csv(OUT_DIR / f"{name}_bootstrap_refit.csv", index=False) # save refit bootstrap results

        for label, df_bs in [("testset", df_bs_test), ("refit", df_bs_refit)]: # for each bootstrap type
            for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]: # for each metric
                vals = df_bs[metric].dropna().values # get metric values
                if len(vals) == 0: # if no values, skip
                    continue
                mean, lo, hi = ci_from_samples(vals) # compute confidence interval
                rows.append({ 
                    "model": name, "bootstrap": label, "metric": metric,    
                    "mean": mean, "ci95_lo": lo, "ci95_hi": hi
                })

                plt.figure()
                plt.hist(vals, bins=40)
                plt.title(f"{name} — {metric} ({label})\nmean={mean:.3f}, 95% CI [{lo:.3f},{hi:.3f}]")
                plt.xlabel(metric)
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(OUT_DIR / "figures" / f"{name}_{metric}_{label}_hist.png", dpi=160)
                plt.close()

    pd.DataFrame(rows).to_csv(OUT_DIR / "bootstrap_summary.csv", index=False)
    print(f"\nSaved bootstrap summary -> {OUT_DIR / 'bootstrap_summary.csv'}")


if __name__ == "__main__":
    main()
