#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="advanced-statistical-learning"
PY_VER="3.10"

echo "==> Project root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

have_cmd () { command -v "$1" >/dev/null 2>&1; }

USE_MAMBA=0
if have_cmd mamba; then
  USE_MAMBA=1
elif ! have_cmd conda; then
  echo "No mamba/conda found. Falling back to python venv."
fi

# --------------------------------------------------------------------
# 1) Create or update environment with full dependencies
# --------------------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  echo "==> Ensuring mamba env '${ENV_NAME}' has all required packages..."
  if mamba env list | grep -q "${ENV_NAME}"; then
    echo "Environment exists. Installing/updating required packages..."
    mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
    mamba run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  else
    echo "Creating environment from scratch..."
    mamba create -y -n "${ENV_NAME}" python=${PY_VER}
    mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
    mamba run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  fi

elif have_cmd conda; then
  echo "==> Ensuring conda env '${ENV_NAME}' has all required packages..."
  if conda env list | grep -q "${ENV_NAME}"; then
    echo "Environment exists. Installing/updating required packages..."
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  else
    echo "Creating environment from scratch..."
    conda create -y -n "${ENV_NAME}" python=${PY_VER}
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  fi

else
  echo "==> Fallback: python venv + pip"
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
fi


# --------------------------------------------------------------------
# 2) Ensure config/default.yaml exists
# --------------------------------------------------------------------
mkdir -p config
if [[ ! -f config/default.yaml ]]; then
  cat > config/default.yaml <<'YAML'
project:
  data_csv: "data/sms_spam.csv"
  out_dir: "outputs"
  seed: 42
  test_size: 0.2
YAML
  echo "==> Created config/default.yaml"
fi

# --------------------------------------------------------------------
# 3) Download UCI SMS Spam dataset if missing
# --------------------------------------------------------------------
mkdir -p data
if [[ ! -f data/sms_spam.csv ]]; then
  echo "==> Downloading SMS Spam dataset..."
  if [[ $USE_MAMBA -eq 1 ]]; then RUN="mamba run -n ${ENV_NAME}"; elif have_cmd conda; then RUN="conda run -n ${ENV_NAME}"; else RUN="python"; fi
  $RUN python - <<'PY'
import zipfile, io, urllib.request, pandas as pd, pathlib
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
pathlib.Path("data").mkdir(parents=True, exist_ok=True)
zip_bytes = urllib.request.urlopen(url).read()
zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
with zf.open("SMSSpamCollection") as f:
    df = pd.read_csv(f, sep="\t", header=None, names=["label","text"], encoding="utf-8")
df.to_csv("data/sms_spam.csv", index=False)
print("Saved data/sms_spam.csv")
PY
fi

# --------------------------------------------------------------------
# 4) Outputs
# --------------------------------------------------------------------
mkdir -p outputs/models outputs/figures

# --------------------------------------------------------------------
# 5) Run LDA/QDA + SVM pipelines
# --------------------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  RUN="mamba run -n ${ENV_NAME}"
elif have_cmd conda; then
  RUN="conda run -n ${ENV_NAME}"
else
  RUN=""   # already in venv
fi

export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=xcb

echo "==> Running LDA/QDA text pipeline..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/qda-lda_after_tifdf-pca.py" || true

echo "==> Running SVM (RBF) text pipeline..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/TF-IDF_SVM_RBF.py" || true

echo "==> DONE."
echo "Figures  -> outputs/figures/"
echo "Models   -> outputs/models/"
# --- 6) Visualize QDA discriminant scores ------------------------------
#from sklearn.preprocessing import LabelEncoder

#print("==> Generating QDA projection plots...")

# Encode classes as 0/1 explicitly
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(X_train_num, y_train_enc)

# QDA does not have a transform(), but we can use the log-posterior ratios
qda_scores_train = qda.predict_proba(X_train_num)[:, 1]  # probability of "spam"
qda_scores_test  = qda.predict_proba(X_test_num)[:, 1]

plt.figure(figsize=(7,4))
plt.hist(qda_scores_train[y_train_enc==0], bins=30, alpha=0.6, label="ham (train)", density=True)
plt.hist(qda_scores_train[y_train_enc==1], bins=30, alpha=0.6, label="spam (train)", density=True)
plt.title("QDA discriminant projection (train)")
plt.xlabel("Posterior probability of 'spam'")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "qda_train_hist.png", dpi=150)

plt.figure(figsize=(7,4))
plt.hist(qda_scores_test[y_test_enc==0], bins=30, alpha=0.6, label="ham (test)", density=True)
plt.hist(qda_scores_test[y_test_enc==1], bins=30, alpha=0.6, label="spam (test)", density=True)
plt.title("QDA discriminant projection (test)")
plt.xlabel("Posterior probability of 'spam'")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "qda_test_hist.png", dpi=150)

print(" - outputs/figures/qda_train_hist.png")
print(" - outputs/figures/qda_test_hist.png")
