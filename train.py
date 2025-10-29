#!/usr/bin/env python3
import argparse, yaml
from pathlib import Path
import joblib
from src.preprocessing import load_or_download, stratified_split
from src.models import make_pipelines
from src.metrics import get_scores, plot_roc_pr

def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()



OUT_MODELS = Path("outputs/models"); OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_FIGS = Path("outputs/figures"); OUT_FIGS.mkdir(parents=True, exist_ok=True)

def _predict_scores(model, X):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # map to (0,1)
        import numpy as np
        y_score = 1/(1+np.exp(-s))
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:,1]
    else:
        y_score = None
    y_pred = model.predict(X)
    return y_pred, y_score

def main():
    df = load_or_download()
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=0.2, random_state=123)
    models = make_pipelines()

    for name, pipe in models.items():
        print(f"\n=== Training {name} ===")
        pipe.fit(X_tr, y_tr)
        y_pred, y_score = _predict_scores(pipe, X_te)
        scores = get_scores(y_te, y_pred, y_score)
        print("Holdout:", scores)

        joblib.dump(pipe, OUT_MODELS / f"{name}.joblib")

        if y_score is not None:
            plot_roc_pr(y_te, y_score, OUT_FIGS / f"{name}")

if __name__ == "__main__":
    main()

