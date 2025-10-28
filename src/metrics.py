import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

def get_scores(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    if y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc = np.nan
        try:
            pr = average_precision_score(y_true, y_score)
        except ValueError:
            pr = np.nan
    else:
        roc, pr = np.nan, np.nan
    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": pr}

def plot_roc_pr(y_true, y_score, out_prefix: str):
    if y_score is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=160); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=160); plt.close()

