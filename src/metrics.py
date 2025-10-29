# src/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

# Map common binary label names to {0,1}
_LABEL_MAP = {
    "ham": 0, "spam": 1,
    "benign": 0, "malignant": 1,
    "negative": 0, "positive": 1,
    "no": 0, "yes": 1,
    0: 0, 1: 1, True: 1, False: 0
}

def _to_binary(y):
    """Convert labels to {0,1} if needed."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        return (y != 0).astype(int)
    out = np.empty_like(y, dtype=int)
    for i, v in enumerate(y):
        out[i] = _LABEL_MAP.get(v, 0)
    return out

def get_scores(y_true, y_pred, y_score=None):
    """Compute Accuracy, F1, ROC AUC, PR AUC (handles both string & numeric labels)."""
    y_true_bin = _to_binary(y_true)
    y_pred_bin = _to_binary(y_pred)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)

    roc, ap = np.nan, np.nan
    if y_score is not None:
        try:
            roc = roc_auc_score(y_true_bin, y_score)
        except Exception:
            pass
        try:
            ap = average_precision_score(y_true_bin, y_score)
        except Exception:
            pass
    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": ap}

def plot_roc_pr(y_true, y_score, out_prefix):
    """Save ROC and PR curves to disk."""
    y_true_bin = _to_binary(y_true)
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    prec, rec, _ = precision_recall_curve(y_true_bin, y_score)

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=160)
    plt.close()

    # Precision-Recall curve
    plt.figure()
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=160)
    plt.close()
