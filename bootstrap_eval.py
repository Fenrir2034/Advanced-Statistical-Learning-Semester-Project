#!/usr/bin/env python3
"""
Bootstrap evaluation of SMS spam classifiers.
This script estimates model uncertainty via bootstrap resampling and refitting.
All paths and parameters are defined in the YAML config file.

Usage:
  python bootstrap_eval.py --config config/default.yaml
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlretrieve


# ---------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------
def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()

PROJECT = CFG.get("project", {})
DATA_CSV = Path(PROJECT.get("data_csv", "data/sms_spam.csv"))
OUT_DIR = Path(PROJECT.get("out_dir", "outputs"))
SEED = int(PROJECT.get("seed", 42))
BOOT_ITERS_TEST = int(PROJECT.get("bootstrap_test_iters", 1000))
BOOT_ITERS_REFIT = int(PROJECT.get("bootstrap_refit_iters", 200))
TEST_SIZE = float(PROJECT.get("test_size", 0.2))

(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def load_sms(path: Path = None) -> pd.DataFrame:
    """
    Load SMS Spam dataset, either from config-specified CSV or download from UCI repository.
    """
    if path is None or not path.exists():
        print(f"Dataset not found at {path}, downloading UCI SMS Spam Collection...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = Path("data/smsspamcollection.zip")
        Path("data").mkdir(exist_ok=True, parents=True)
        if not zip_path.exists():
            urlretrieve(url, zip_path.as_posix())
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    else:
        df = pd.read_csv(path)
        assert {"label", "text"} <= set(df.columns), "CSV must have columns: label,text"

    # Map label to binary
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    else:
        df["label"] = df["label"].astype(int)

    return df


# ---------------------------------------------------------------------
# MODEL FACTORIES
# ---------------------------------------------------------------------
def make_pipelines():
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_df=0.95
    )

    logit = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(max_iter=5000, solver="liblinear", C=1.0))
    ])
    linsvm = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LinearSVC(C=1.0))
    ])
    linsvm_cal = Pipeline([
        ("tfidf", vectorizer),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=5, method="sigmoid"))
    ])
    rf = Pipeline([
        ("tfidf", vectorizer),
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED))
    ])
    return {
        "logistic": logit,
        "linear_svm": linsvm,
        "linear_svm_cal": linsvm_cal,
        "random_forest": rf
    }


# ---------------------------------------------------------------------
# METRICS & HELPERS
# ---------------------------------------------------------------------
def _get_scores(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    roc = ap = np.nan
    if y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
        except ValueError:
            pass
        try:
            ap = average_precision_score(y_true, y_score)
        except ValueError:
            pass
    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": ap}


def _predict_scores(model: Pipeline, X):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        y_score = 1 / (1 + np.exp(-s))
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = None
    y_pred = model.predict(X)
    return y_pred, y_score


# ---------------------------------------------------------------------
# BOOTSTRAP LOGIC
# ---------------------------------------------------------------------
def bootstrap_testset_metrics(model, X_test, y_test, B=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    metrics = []
    n = len(y_test)
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        y_true_b = y_test[idx]
        X_b = X_test.iloc[idx]
        y_pred_b, y_score_b = _predict_scores(model, X_b)
        metrics.append(_get_scores(y_true_b, y_pred_b, y_score_b))
    return pd.DataFrame(metrics)


def bootstrap_refit(model_factory, X, y, test_size=0.2, B=200, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y)
    metrics = []
    for _ in range(B):
        train_idx = rng.randint(0, n, size=n)
        mask = np.ones(n, dtype=bool)
        mask[train_idx] = False
        test_idx = np.where(mask)[0]
        if len(test_idx) < max(30, int(0.05 * n)):
            from sklearn.model_selection import train_test_split
            tr_idx, te_idx = train_test_split(
                np.arange(n), test_size=test_size, stratify=y,
                random_state=rng.randint(0, 10**9)
            )
            train_idx, test_idx = tr_idx, te_idx
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred, y_score = _predict_scores(model, X_te)
        metrics.append(_get_scores(y_te, y_pred, y_score))
    return pd.DataFrame(metrics)


def ci_from_samples(samples, alpha=0.05):
    lo = np.percentile(samples, 100 * (alpha / 2))
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(np.mean(samples)), float(lo), float(hi)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_sms(DATA_CSV)
    X = df["text"]
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    models = make_pipelines()
    rows = []

    for name, pipe in models.items():
        print(f"\n=== {name.upper()} ===")
        pipe.fit(X_tr, y_tr)
        y_pred, y_score = _predict_scores(pipe, X_te)
        base = _get_scores(y_te, y_pred, y_score)
        print("Holdout metrics:", base)

        # --- Bootstrap ---
        df_bs_test = bootstrap_testset_metrics(pipe, X_te, y_te, B=BOOT_ITERS_TEST, random_state=SEED)
        df_bs_test.to_csv(OUT_DIR / f"{name}_bootstrap_testset.csv", index=False)

        df_bs_refit = bootstrap_refit(
            lambda: make_pipelines()[name], X, y,
            test_size=TEST_SIZE, B=BOOT_ITERS_REFIT, random_state=SEED
        )
        df_bs_refit.to_csv(OUT_DIR / f"{name}_bootstrap_refit.csv", index=False)

        for label, df_bs in [("testset", df_bs_test), ("refit", df_bs_refit)]:
            for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]:
                vals = df_bs[metric].dropna().values
                if len(vals) == 0:
                    continue
                mean, lo, hi = ci_from_samples(vals)
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
