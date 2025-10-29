#!/usr/bin/env python3
"""
Visualize LDA projection before and after SMOTE on SMS spam data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import matplotlib as plt
plt.use("Agg")

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
df = pd.read_csv("data/sms_spam.csv")
X_text = df["text"]
y = df["label"]

# Convert labels to numeric if needed
if y.dtype == object:
    y = y.map({"ham": 0, "spam": 1}).astype(int)

# ----------------------------------------------------
# 2. Vectorize
# ----------------------------------------------------
vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.95)
X = vectorizer.fit_transform(X_text)

# ----------------------------------------------------
# 3. LDA on original data
# ----------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X.toarray(), y)

plt.figure(figsize=(6, 4))
plt.scatter(X_lda[y==0], [1]*sum(y==0), c="blue", label="ham", alpha=0.6, s=10)
plt.scatter(X_lda[y==1], [1.02]*sum(y==1), c="red", label="spam", alpha=0.6, s=10)
plt.title("LDA: Linear Discriminants Analysis (original data)")
plt.xlabel("First Discriminant")
plt.ylabel("")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/lda_original.png", dpi=160)
plt.close()

# ----------------------------------------------------
# 4. Apply SMOTE and LDA again
# ----------------------------------------------------
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

lda_sm = LinearDiscriminantAnalysis(n_components=1)
X_lda_sm = lda_sm.fit_transform(X_sm.toarray(), y_sm)

plt.figure(figsize=(6, 4))
plt.scatter(X_lda_sm[y_sm==0], [0]*sum(y_sm==0), c="blue", label="ham", alpha=0.6, s=10)
plt.scatter(X_lda_sm[y_sm==1], [0.02]*sum(y_sm==1), c="red", label="spam", alpha=0.6, s=10)
plt.title("LDA Projection of SMS Data (SMOTEd data)")
plt.xlabel("LDA Component 1")
plt.ylabel("")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/lda_smote.png", dpi=160)
plt.close()

print("✅ Saved LDA projections:")
print(" - outputs/figures/lda_original.png")
print(" - outputs/figures/lda_smote.png")
