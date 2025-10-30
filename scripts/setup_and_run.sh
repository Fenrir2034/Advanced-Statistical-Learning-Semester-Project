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

# 1) env ----------------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  echo "==> Ensuring mamba env '${ENV_NAME}' has all required packages..."
  if mamba env list | grep -q "${ENV_NAME}"; then
    mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
    mamba run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  else
    mamba create -y -n "${ENV_NAME}" python=${PY_VER}
    mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
    mamba run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  fi
elif have_cmd conda; then
  echo "==> Ensuring conda env '${ENV_NAME}' has all required packages..."
  if conda env list | grep -q "${ENV_NAME}"; then
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
      numpy pandas scikit-learn imbalanced-learn matplotlib seaborn pyyaml joblib tqdm
  else
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

# 2) config -------------------------------------------------------------
mkdir -p config
if [[ ! -f config/default.yaml ]]; then
  cat > config/default.yaml <<'YAML'
project:
  data_csv: "data/sms_spam.csv"
  out_dir: "outputs"
  seed: 42
  test_size: 0.2
  bootstrap_test_iters: 1000
  bootstrap_refit_iters: 200
YAML
  echo "==> Created config/default.yaml"
fi

# 3) data ---------------------------------------------------------------
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

# 4) outputs ------------------------------------------------------------
mkdir -p outputs/models outputs/figures outputs/reports

# 5) choose runner ------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  RUN="mamba run -n ${ENV_NAME}"
elif have_cmd conda; then
  RUN="conda run -n ${ENV_NAME}"
else
  RUN=""   # venv
fi

export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=xcb

# 6) run pipelines ------------------------------------------------------
echo "==> Running LDA/QDA text pipeline..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/qda-lda_after_tifdf-pca.py" || true

echo "==> Running SVM (RBF) text pipeline..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/TF-IDF_SVM_RBF.py" || true

echo "==> Training (multi-model)..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/train.py" --config "config/default.yaml" || true

echo "==> Evaluation summary..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/evaluate.py" --config "config/default.yaml" || true

echo "==> Bootstrap metrics..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "scripts/bootstrap_eval.py" --config "config/default.yaml" || true

echo "==> DONE."
echo "Figures  -> outputs/figures/"
echo "Models   -> outputs/models/"
echo "Reports  -> outputs/reports/"
