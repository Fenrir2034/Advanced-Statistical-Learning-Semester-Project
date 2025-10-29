#!/usr/bin/env bash
set -euo pipefail

# Allow custom config path (default to config/default.yaml)
CONF="${1:-config/default.yaml}"

echo "[run] Using config: $CONF"

# 1) Ensure data exists (download from OpenML if needed)
CSV_PATH=$(python - <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["project"]["data_csv"])
PY
"$CONF")

if [ ! -f "$CSV_PATH" ]; then
  echo "[run] Dataset not found at $CSV_PATH. Downloading..."
  python src/download_data.py --out "$CSV_PATH"
else
  echo "[run] Found dataset at $CSV_PATH"
fi

# 2) Train + save models/plots/metrics
python src/train.py --config "$CONF"

# 3) Bootstrap eval
python src/bootstrap_eval.py --config "$CONF"

# 4) Final evaluation + plots
python src/evaluate.py --config "$CONF"

echo "[run] All steps completed. See outputs/ for artifacts."

