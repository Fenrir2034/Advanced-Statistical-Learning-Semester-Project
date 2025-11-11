from imblearn.pipeline import Pipeline  # <-- not sklearn.pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path

# --- data ---
df = pd.read_csv(Path("data/sms_spam.csv"))
X = df["text"].astype(str)
y = (df["label"].astype(str).str.lower().str.strip() == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- pipeline with SMOTE inside (imblearn) ---
pipe = Pipeline(steps=[ # pipeline steps
    ("tfidf", TfidfVectorizer( #    text to numeric
        lowercase=True, stop_words="english", # remove stop words
        ngram_range=(1,2), max_df=0.95, min_df=2 # n-grams and filtering
    )),
    ("smote", SMOTE(random_state=42)),           # works because it's imblearn.Pipeline
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)) # SVM with RBF kernel: the RBF kernel can handle non-linear decision boundaries
]) # the pipeline first vectorizes the text, then applies SMOTE to balance classes, and finally fits the SVM model
# the RBF kernel is a popular choice for SVMs when the relationship between features and classes is non-linear

param_grid = { # hyperparameter grid for tuning
    "svm__C": [0.5, 1, 3, 10],
    "svm__gamma": [1e-3, 3e-3, 1e-2]
}

grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1) # grid search for hyperparameter tuning
grid.fit(X_train, y_train) # fit model

print("Best params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("\n=== SVM (RBF) report ===")
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))
