#!/usr/bin/env python3
"""
Train and evaluate SMS spam classifiers with config-driven paths.

Usage:
  python train.py --config config/default.yaml
"""
# in this file we set up a training script that uses a YAML config file to manage paths and parameters
# we need it to be able to load data, train models, save outputs, and evaluate performance
#the moddels we will use are from sklearn and imblearn: logistic regression, random forest, svm, naive bayes, etc.

import argparse
import json
from pathlib import Path

import joblib
import yaml
import pandas as pd
import numpy as np
import random
import os
# Project utils
# --- ensure repo root is on sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

from src.preprocessing import stratified_split, load_or_download
from src.models import make_pipelines
from src.metrics import get_scores, plot_roc_pr


# -----------------------------
# Config
# -----------------------------
def load_config() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()

# Required keys with sensible defaults
PROJECT = CFG.get("project", {})
CSV_PATH = Path(PROJECT.get("data_csv", "data/sms_spam.csv"))
OUT_MODELS = Path(PROJECT.get("out_models", "outputs/models"))
OUT_FIGS = Path(PROJECT.get("out_figs", "outputs/figures"))
OUT_REPORTS = Path(PROJECT.get("out_reports", "outputs/reports"))
TEST_SIZE = float(PROJECT.get("test_size", 0.2))
SEED = int(PROJECT.get("seed", 42))


np.random.seed(SEED) # for numpy
random.seed(SEED) # for random
os.environ["PYTHONHASHSEED"] = str(SEED)
# Ensure output dirs exist
for p in [OUT_MODELS, OUT_FIGS, OUT_REPORTS]:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading (config-first)
# -----------------------------
def load_dataframe() -> pd.DataFrame: # load dataframe from config CSV or fallback
    """
    Load dataframe from config CSV if it exists; else fall back to project helper.
    Tries to normalize column names to the expected ['text', 'label'].
    """
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        # Fallback to legacy helper; keep bc older code may still download/prepare the dataset.
        df = load_or_download()

    # Normalize common column name variants to text/label
    col_map = {} # mapping of columns to rename
    lower_cols = {c.lower(): c for c in df.columns} # map lowercase to original

    # Map a likely text column
    for cand in ["text", "message", "sms_text", "content", "body"]: # candidate names
        if cand in lower_cols: # if candidate found
            col_map[lower_cols[cand]] = "text" # map to "text"
            break # stop after first match

    # Map a likely label column
    for cand in ["label", "class", "category", "target"]: # candidate names
        if cand in lower_cols: # if candidate found
            col_map[lower_cols[cand]] = "label" # map to "label"
            break

    if col_map: # if we have any mappings
        df = df.rename(columns=col_map) # rename columns

    # Basic checks
    if "text" not in df.columns or "label" not in df.columns: # if required columns missing
        raise ValueError( 
            f"Expected columns 'text' and 'label' in the dataset. "
            f"Found columns: {list(df.columns)}. "
            f"If your CSV uses different names, either rename them or extend the mapping above."
        )

    # Ensure label is a binary factor-like string (e.g., 'spam'/'ham' or '1'/'0')
    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    return df


# -----------------------------
# Helpers
# -----------------------------
def _predict_scores(model, X): # get predictions and scores from model
    """
    Returns (y_pred, y_score) with y_score in [0,1] if possible.
    """
    if hasattr(model, "decision_function"): # if model has decision_function
        s = model.decision_function(X)  # this gives raw scores                        
        # Logistic squashing to [0,1] (probabilities)
        y_score = 1.0 / (1.0 + np.exp(-s))  # sigmoid function
    elif hasattr(model, "predict_proba"):  # if model has predict_proba
        y_score = model.predict_proba(X)[:, 1] # get positive class probabilities
    else:
        y_score = None
    y_pred = model.predict(X)
    return y_pred, y_score


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Data
    df = load_dataframe()
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=TEST_SIZE, random_state=SEED) #we split the data into training and testing sets using stratified sampling to maintain class distribution

    # 2) Models
    # If you want model options from YAML, you can pass CFG.get("models") to make_pipelines
    models = make_pipelines() # get model pipelines

    all_scores = {} # to store all model scores

    # 3) Train/eval loop
    for name, pipe in models.items(): # for each model pipeline
        print(f"\n=== Training {name} ===") 
        pipe.fit(X_tr, y_tr)

        # Save trained model
        model_path = OUT_MODELS / f"{name}.joblib" # path to save model
        joblib.dump(pipe, model_path)

        # Evaluate
        y_pred, y_score = _predict_scores(pipe, X_te) # get predictions and scores
        scores = get_scores(y_te, y_pred, y_score) # compute evaluation metrics
        all_scores[name] = scores # store scores
        print(f"Holdout scores for {name}: {scores}") 

        # Curves
        if y_score is not None: # if we have scores
            plot_roc_pr(y_te, y_score, OUT_FIGS / f"{name}")

    # 4) Persist a small report (JSON)
    report_path = OUT_REPORTS / "holdout_scores.json"
    with open(report_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved metrics JSON -> {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
