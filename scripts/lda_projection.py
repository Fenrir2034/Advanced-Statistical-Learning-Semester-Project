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
df = pd.read_csv("data/sms_spam.csv") # load SMS spam dataset
X_text = df["text"] # feature: text messages
y = df["label"] # label: spam/ham

# Convert labels to numeric if needed
if y.dtype == object: # if labels are strings
    y = y.map({"ham": 0, "spam": 1}).astype(int) # map to 0 and 1

# ----------------------------------------------------
# 2. Vectorize
# ----------------------------------------------------
vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.95) # TF-IDF vectorizer
X = vectorizer.fit_transform(X_text) # fit and transform text data

# ----------------------------------------------------
# 3. LDA on original data
# ----------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=1)    # LDA projection
X_lda = lda.fit_transform(X.toarray(), y) # fit and transform data

plt.figure(figsize=(6, 4)) # create figure
plt.scatter(X_lda[y==0], [1]*sum(y==0), c="blue", label="ham", alpha=0.6, s=10) # plot ham points
plt.scatter(X_lda[y==1], [1.02]*sum(y==1), c="red", label="spam", alpha=0.6, s=10) # plot spam points
plt.title("LDA: Linear Discriminants Analysis (original data)") # title
plt.xlabel("First Discriminant") # x-axis label
plt.ylabel("")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/lda_original.png", dpi=160)
plt.close()

# ----------------------------------------------------
# 4. Apply SMOTE and LDA again
# ----------------------------------------------------
smote = SMOTE(random_state=42)  # SMOTE for balancing classes
X_sm, y_sm = smote.fit_resample(X, y) # apply SMOTE

lda_sm = LinearDiscriminantAnalysis(n_components=1)   # LDA projection
X_lda_sm = lda_sm.fit_transform(X_sm.toarray(), y_sm) # fit and transform SMOTEd data

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

print(" Saved LDA projections:")
print(" - outputs/figures/lda_original.png")
print(" - outputs/figures/lda_smote.png")
