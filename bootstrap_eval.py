#!/usr/bin/env python3
import argparse, yaml
import sys, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()


def ensure_outputs():
    out = Path("outputs"); out.mkdir(exist_ok=True)
    (out/"figures").mkdir(exist_ok=True, parents=True)
    return out

def load_sms(path: str = None) -> pd.DataFrame:
    if path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = Path("data/smsspamcollection.zip"); Path("data").mkdir(exist_ok=True, parents=True)
        if not zip_path.exists():
            urlretrieve(url, zip_path.as_posix())
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["label","text"], encoding="utf-8")
    else:
        df = pd.read_csv(path)
        assert {"label","text"} <= set(df.columns), "CSV must have columns: label,text"
    df["label"] = df["label"].map({"ham":0, "spam":1}).astype(int) if df["label"].dtype == object else df["label"].astype(int)
    return df

def make_pipelines():
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), min_df=2, max_df=0.95)
    logit = Pipeline([("tfidf", vectorizer), ("clf", LogisticRegression(max_iter=5000, solver="liblinear", C=1.0))])
    linsvm = Pipeline([("tfidf", vectorizer), ("clf", LinearSVC(C=1.0))])
    linsvm_cal = Pipeline([("tfidf", vectorizer), ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=5, method="sigmoid"))])
    rf = Pipeline([("tfidf", vectorizer), ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42))])
    return {"logistic": logit, "linear_svm": linsvm, "linear_svm_cal": linsvm_cal, "random_forest": rf}

def _get_scores(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=1)
    if y_score is not None:
        try: roc = roc_auc_score(y_true, y_score)
        except ValueError: roc = np.nan
        try: ap = average_precision_score(y_true, y_score)
        except ValueError: ap = np.nan
    else:
        roc, ap = np.nan, np.nan
    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": ap}

def _predict_scores(model: Pipeline, X):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        y_score = 1/(1+np.exp(-s))
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:,1]
    else:
        y_score = None
    y_pred = model.predict(X)
    return y_pred, y_score

def bootstrap_testset_metrics(model, X_test, y_test, B=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    metrics = []; n = len(y_test)
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        y_true_b = y_test[idx]; X_b = X_test.iloc[idx]
        y_pred_b, y_score_b = _predict_scores(model, X_b)
        metrics.append(_get_scores(y_true_b, y_pred_b, y_score_b))
    return pd.DataFrame(metrics)

def bootstrap_refit(model_factory, X, y, test_size=0.2, B=200, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y); metrics = []
    for _ in range(B):
        train_idx = rng.randint(0, n, size=n)
        mask = np.ones(n, dtype=bool); mask[train_idx] = False
        test_idx = np.where(mask)[0]
        if len(test_idx) < max(30, int(0.05*n)):
            from sklearn.model_selection import train_test_split
            tr_idx, te_idx = train_test_split(np.arange(n), test_size=test_size, stratify=y, random_state=rng.randint(0,10**9))
            train_idx, test_idx = tr_idx, te_idx
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred, y_score = _predict_scores(model, X_te)
        metrics.append(_get_scores(y_te, y_pred, y_score))
    return pd.DataFrame(metrics)

def ci_from_samples(samples, alpha=0.05):
    lo = np.percentile(samples, 100*(alpha/2))
    hi = np.percentile(samples, 100*(1 - alpha/2))
    return float(np.mean(samples)), float(lo), float(hi)

def main():
    outdir = ensure_outputs()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_sms(path)
    X = df["text"]; y = df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

    models = make_pipelines()
    rows = []
    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        pipe.fit(X_tr, y_tr)
        y_pred, y_score = _predict_scores(pipe, X_te)
        base = _get_scores(y_te, y_pred, y_score)
        print("Holdout:", base)

        df_bs_test = bootstrap_testset_metrics(pipe, X_te, y_te, B=1000, random_state=42)
        df_bs_test.to_csv(outdir / f"{name}_bootstrap_testset.csv", index=False)

        df_bs_refit = bootstrap_refit(lambda: make_pipelines()[name], X, y, B=200, random_state=1337)
        df_bs_refit.to_csv(outdir / f"{name}_bootstrap_refit.csv", index=False)

        for label, df_bs in [("testset", df_bs_test), ("refit", df_bs_refit)]:
            for metric in ["accuracy","f1","roc_auc","pr_auc"]:
                vals = df_bs[metric].dropna().values
                if len(vals)==0: continue
                mean, lo, hi = ci_from_samples(vals)
                rows.append({"model": name, "bootstrap": label, "metric": metric, "mean": mean, "ci95_lo": lo, "ci95_hi": hi})

                plt.figure()
                plt.hist(vals, bins=40)
                plt.title(f"{name} — {metric} ({label})\nmean={mean:.3f}, 95% CI [{lo:.3f},{hi:.3f}]")
                plt.xlabel(metric); plt.ylabel("count"); plt.tight_layout()
                plt.savefig(outdir/"figures"/f"{name}_{metric}_{label}_hist.png", dpi=160); plt.close()

    pd.DataFrame(rows).to_csv(outdir/"bootstrap_summary.csv", index=False)
    print("Saved:", outdir/"bootstrap_summary.csv")

if __name__ == "__main__":
    main()

