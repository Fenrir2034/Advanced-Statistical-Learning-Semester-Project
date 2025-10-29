#!/usr/bin/env python3
# add near the top of each script
import argparse, yaml
from pathlib import Path
import joblib
import pandas as pd
from src.preprocessing import load_or_download, stratified_split
from src.metrics import get_scores
# --- ensure repo root is on sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

MODELS_DIR = Path("outputs/models")

def _predict_scores(model, X):
    if hasattr(model, "decision_function"):
        import numpy as np
        s = model.decision_function(X); y_score = 1/(1+np.exp(-s))
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:,1]
    else:
        y_score = None
    y_pred = model.predict(X)
    return y_pred, y_score

def main():
    df = load_or_download()
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=0.2, random_state=123)

    rows = []
    for path in MODELS_DIR.glob("*.joblib"):
        name = path.stem
        model = joblib.load(path)
        y_pred, y_score = _predict_scores(model, X_te)
        scores = get_scores(y_te, y_pred, y_score)
        rows.append({"model": name, **scores})

    out = Path("outputs")/"evaluation_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()

