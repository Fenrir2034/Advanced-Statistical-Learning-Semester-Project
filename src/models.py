from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

def vectorizer():
    return TfidfVectorizer(lowercase=True, strip_accents="unicode",
                           ngram_range=(1,2), min_df=2, max_df=0.95)

def make_pipelines():
    vec = vectorizer()
    logit = Pipeline([("tfidf", vec),
                      ("clf", LogisticRegression(max_iter=5000, solver="liblinear", C=1.0))])
    linsvm = Pipeline([("tfidf", vec),
                       ("clf", LinearSVC(C=1.0))])
    linsvm_cal = Pipeline([("tfidf", vec),
                           ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=5, method="sigmoid"))])
    rf = Pipeline([("tfidf", vec),
                   ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42))])
    return {
        "logistic": logit,
        "linear_svm": linsvm,
        "linear_svm_cal": linsvm_cal,
        "random_forest": rf,
    }

