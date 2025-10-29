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
# 1) Create/update environment
# --------------------------------------------------------------------
ENV_YML=""
for f in environment.yml environement-requirements.yml env.yml; do
  [[ -f "$f" ]] && ENV_YML="$f" && break
done

if [[ $USE_MAMBA -eq 1 ]]; then
  echo "==> Using mamba env: ${ENV_NAME}"
  if [[ -n "$ENV_YML" ]]; then
    mamba env update -n "${ENV_NAME}" -f "${ENV_YML}" || mamba env create -n "${ENV_NAME}" -f "${ENV_YML}" || true
  else
    mamba create -y -n "${ENV_NAME}" python=${PY_VER}
    mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
    if [[ -f requirements.txt ]]; then
      mamba run -n "${ENV_NAME}" pip install -r requirements.txt
    else
      mamba run -n "${ENV_NAME}" pip install numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
    fi
  fi
elif have_cmd conda; then
  echo "==> Using conda env: ${ENV_NAME}"
  if [[ -n "$ENV_YML" ]]; then
    conda env update -n "${ENV_NAME}" -f "${ENV_YML}" || conda env create -n "${ENV_NAME}" -f "${ENV_YML}" || true
  else
    conda create -y -n "${ENV_NAME}" python=${PY_VER}
    conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
    if [[ -f requirements.txt ]]; then
      conda run -n "${ENV_NAME}" pip install -r requirements.txt
    else
      conda run -n "${ENV_NAME}" pip install numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
    fi
  fi
else
  echo "==> Fallback: python venv + pip"
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  [[ -f requirements.txt ]] && pip install -r requirements.txt || pip install numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
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
  bootstrap_test_iters: 200
  bootstrap_refit_iters: 100
YAML
  echo "==> Created config/default.yaml"
fi

# --------------------------------------------------------------------
# 3) Data: download UCI SMS if missing
# --------------------------------------------------------------------
mkdir -p data
if [[ ! -f data/sms_spam.csv ]]; then
  echo "==> Downloading SMS Spam dataset and converting to CSV..."
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
# 5) Run training & bootstrap via env-runner (no activation)
# --------------------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  RUN="mamba run -n ${ENV_NAME}"
elif have_cmd conda; then
  RUN="conda run -n ${ENV_NAME}"
else
  RUN=""   # already in venv
fi

export PYTHONHASHSEED=42

echo "==> Running training..."
$RUN python scripts/train.py --config config/default.yaml

echo "==> Running bootstrap evaluation..."
$RUN python scripts/bootstrap_eval.py --config config/default.yaml

echo "==> DONE."
echo "Models   -> outputs/models/"
echo "Figures  -> outputs/figures/"
echo "Bootstrap-> outputs/bootstrap_summary.csv"

# --------------------------------------------------------------------
# 5) Run training & bootstrap via env-runner (no activation)
# --------------------------------------------------------------------
if [[ $USE_MAMBA -eq 1 ]]; then
  RUN="mamba run -n ${ENV_NAME}"
elif have_cmd conda; then
  RUN="conda run -n ${ENV_NAME}"
else
  RUN=""   # already in venv
fi

export PYTHONHASHSEED=42

echo "==> Running training..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/train.py --config config/default.yaml

echo "==> Running bootstrap evaluation..."
$RUN env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/bootstrap_eval.py --config config/default.yaml

echo '==> DONE.'
echo 'Models   -> outputs/models/'
echo 'Figures  -> outputs/figures/'
echo 'Bootstrap-> outputs/bootstrap_summary.csv'
