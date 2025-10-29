#!/usr/bin/env python3
"""
Train and evaluate SMS spam classifiers with config-driven paths.

Usage:
  python train.py --config config/default.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import yaml
import pandas as pd
import numpy as np

# Project utils
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

# Ensure output dirs exist
for p in [OUT_MODELS, OUT_FIGS, OUT_REPORTS]:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading (config-first)
# -----------------------------
def load_dataframe() -> pd.DataFrame:
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
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Map a likely text column
    for cand in ["text", "message", "sms_text", "content", "body"]:
        if cand in lower_cols:
            col_map[lower_cols[cand]] = "text"
            break

    # Map a likely label column
    for cand in ["label", "class", "category", "target"]:
        if cand in lower_cols:
            col_map[lower_cols[cand]] = "label"
            break

    if col_map:
        df = df.rename(columns=col_map)

    # Basic checks
    if "text" not in df.columns or "label" not in df.columns:
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
def _predict_scores(model, X):
    """
    Returns (y_pred, y_score) with y_score in [0,1] if possible.
    """
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # Logistic squashing to [0,1]
        y_score = 1.0 / (1.0 + np.exp(-s))
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
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
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=TEST_SIZE, random_state=SEED)

    # 2) Models
    # If you want model options from YAML, you can pass CFG.get("models") to make_pipelines
    models = make_pipelines()

    all_scores = {}

    # 3) Train/eval loop
    for name, pipe in models.items():
        print(f"\n=== Training {name} ===")
        pipe.fit(X_tr, y_tr)

        # Save trained model
        model_path = OUT_MODELS / f"{name}.joblib"
        joblib.dump(pipe, model_path)

        # Evaluate
        y_pred, y_score = _predict_scores(pipe, X_te)
        scores = get_scores(y_te, y_pred, y_score)
        all_scores[name] = scores
        print(f"Holdout scores for {name}: {scores}")

        # Curves
        if y_score is not None:
            plot_roc_pr(y_te, y_score, OUT_FIGS / f"{name}")

    # 4) Persist a small report (JSON)
    report_path = OUT_REPORTS / "holdout_scores.json"
    with open(report_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved metrics JSON -> {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
