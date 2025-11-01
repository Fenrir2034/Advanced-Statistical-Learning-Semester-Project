#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/sms_spam.csv")

# --- Load dataset ---

df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Shared TF-IDF + SVD preproc ---
vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                             ngram_range=(1, 2), min_df=2, max_df=0.95)
svd = TruncatedSVD(n_components=200, random_state=42)
scaler = StandardScaler()

def make_features(X_fit, X_apply):
    X_fit_vec = vectorizer.fit_transform(X_fit)
    X_apply_vec = vectorizer.transform(X_apply)
    X_fit_red = svd.fit_transform(X_fit_vec)
    X_apply_red = svd.transform(X_apply_vec)
    X_fit_scaled = scaler.fit_transform(X_fit_red)
    X_apply_scaled = scaler.transform(X_apply_red)
    return X_fit_scaled, X_apply_scaled

X_train_prep, X_test_prep = make_features(X_train, X_test)

# --- Train models ---
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

lda.fit(X_train_prep, y_train)
qda.fit(X_train_prep, y_train)

# --- Predict and compute confusion matrices ---
lda_cm = confusion_matrix(y_test, lda.predict(X_test_prep))
qda_cm = confusion_matrix(y_test, qda.predict(X_test_prep))

# --- Save figures ---
fig_dir = Path("Advanced-Statistical-Learning-Semester-Project/outputs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

for model_name, cm in [("lda", lda_cm), ("qda", qda_cm)]:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot(cmap="Purples", values_format="d", colorbar=False)
    plt.title(f"{model_name.upper()} Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(""outputs/f"{model_name}_cm.png", dpi=160) # Save figure
    plt.close()

print("Saved LDA and QDA confusion matrices to: outputs/figures")
