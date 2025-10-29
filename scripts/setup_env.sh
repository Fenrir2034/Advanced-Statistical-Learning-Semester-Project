#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="advanced-statistical-learning"

if command -v mamba >/dev/null 2>&1; then
  echo "[setup] Using mamba"
  mamba env update -n "$ENV_NAME" -f environment-requirements.yml || \
  mamba env create -n "$ENV_NAME" -f environment-requirements.yml || true
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  pip install -r requirements.txt
else
  echo "[setup] mamba not found; falling back to conda+pip"
  conda env update -n "$ENV_NAME" -f environment-requirements.yml || \
  conda env create -n "$ENV_NAME" -f environment-requirements.yml || true
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  pip install -r requirements.txt
fi

python -c "import sklearn, numpy, pandas, matplotlib, seaborn, yaml; print('[setup] Python deps OK')"
echo "[setup] Done. Activate via: conda activate $ENV_NAME"

