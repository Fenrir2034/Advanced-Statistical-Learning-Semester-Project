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
