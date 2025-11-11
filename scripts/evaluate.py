#!/usr/bin/env python3
# add near the top of each script
import argparse, yaml # for config
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

def load_config(): # load configuration from YAML file: that's where we keep parameters
    ap = argparse.ArgumentParser() # argument parser
    ap.add_argument("--config", default="config/default.yaml") # config file argument
    args = ap.parse_args() # parse arguments
    with open(args.config, "r") as f: # open config file
        return yaml.safe_load(f) # load YAML content

CFG = load_config() # load config at module level

MODELS_DIR = Path("outputs/models") # directory where models are stored

def _predict_scores(model, X): # get predictions and scores from model
    if hasattr(model, "decision_function"): # if model has decision_function
        import numpy as np # import numpy for calculations
        s = model.decision_function(X); y_score = 1/(1+np.exp(-s)) # convert to probabilities via sigmoid
    elif hasattr(model, "predict_proba"):   # if model has predict_proba
        y_score = model.predict_proba(X)[:,1] # get positive class probabilities
    else: # otherwise
        y_score = None # no scores available
    y_pred = model.predict(X) # get class predictions
    return y_pred, y_score # return predictions and scores

def main(): # main evaluation function
    df = load_or_download()
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=0.2, random_state=123)

    rows = [] # list to store evaluation results
    for path in MODELS_DIR.glob("*.joblib"):
        name = path.stem
        model = joblib.load(path)
        y_pred, y_score = _predict_scores(model, X_te)
        scores = get_scores(y_te, y_pred, y_score)
        rows.append({"model": name, **scores})

    out = Path("outputs")/"evaluation_summary.csv" # output CSV path
    pd.DataFrame(rows).to_csv(out, index=False) # save results to CSV
    print("Saved:", out) # print confirmation

if __name__ == "__main__":
    main()

