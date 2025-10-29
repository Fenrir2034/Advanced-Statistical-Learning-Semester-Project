#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="advanced-statistical-learning"
PY_VER="3.10"

echo "==> Project root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

# --------------------------------------------------------------------
# 0) Helper: detect mamba/conda
# --------------------------------------------------------------------
have_cmd () { command -v "$1" >/dev/null 2>&1; }

USE_MAMBA=0
if have_cmd mamba; then
  USE_MAMBA=1
elif have_cmd conda; then
  USE_MAMBA=0
else
  echo "No mamba/conda found. Will fallback to python -m venv + pip."
fi

# --------------------------------------------------------------------
# 1) Create environment
# --------------------------------------------------------------------
ENV_YML=""
for f in environment.yml environement-requirements.yml env.yml; do
  if [[ -f "$f" ]]; then ENV_YML="$f"; break; fi
done

if [[ $USE_MAMBA -eq 1 || -n "$(command -v conda || true)" ]]; then
  echo "==> Using $( [[ $USE_MAMBA -eq 1 ]] && echo mamba || echo conda ) env: ${ENV_NAME}"
  if [[ -n "$ENV_YML" ]]; then
    echo "==> Found environment file: ${ENV_YML}"
    if [[ $USE_MAMBA -eq 1 ]]; then
      mamba env update -n "${ENV_NAME}" -f "${ENV_YML}" || mamba env create -n "${ENV_NAME}" -f "${ENV_YML}" || true
    else
      conda env update -n "${ENV_NAME}" -f "${ENV_YML}" || conda env create -n "${ENV_NAME}" -f "${ENV_YML}" || true
    fi
  else
    echo "==> No environment.yml found; creating minimal env with core deps"
    if [[ $USE_MAMBA -eq 1 ]]; then
      mamba create -y -n "${ENV_NAME}" python=${PY_VER}
      mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
      mamba run -n "${ENV_NAME}" pip install -r requirements.txt || mamba run -n "${ENV_NAME}" pip install \
        numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
    else
      conda create -y -n "${ENV_NAME}" python=${PY_VER}
      conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
      conda run -n "${ENV_NAME}" pip install -r requirements.txt || conda run -n "${ENV_NAME}" pip install \
        numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
    fi
  fi
  # Activate
  if [[ $USE_MAMBA -eq 1 ]]; then
    # shellcheck disable=SC1091
    source "$(mamba info --base)/etc/profile.d/conda.sh"
  else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  conda activate "${ENV_NAME}"
else
  echo "==> Fallback: python venv + pip"
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
  else
    pip install numpy pandas scikit-learn matplotlib pyyaml joblib tqdm
  fi
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
# 3) Ensure data exists (download UCI zip and convert to CSV if needed)
# --------------------------------------------------------------------
mkdir -p data
if [[ ! -f data/sms_spam.csv ]]; then
  echo "==> Downloading SMS Spam dataset and converting to CSV..."
  python - <<'PY'
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
# 4) Ensure outputs folders
# --------------------------------------------------------------------
mkdir -p outputs/models outputs/figures

# --------------------------------------------------------------------
# 5) Pin seed in environment (optional, models already use SEED from YAML)
# --------------------------------------------------------------------
export PYTHONHASHSEED=42

# --------------------------------------------------------------------
# 6) Run training
# --------------------------------------------------------------------
echo "==> Running training..."
python scripts/train.py --config config/default.yaml

# --------------------------------------------------------------------
# 7) Run bootstrap evaluation
# --------------------------------------------------------------------
echo "==> Running bootstrap evaluation..."
python scripts/bootstrap_eval.py --config config/default.yaml

echo "==> DONE."
echo "Models   -> outputs/models/"
echo "Figures  -> outputs/figures/"
echo "Bootstrap-> outputs/bootstrap_summary.csv"
