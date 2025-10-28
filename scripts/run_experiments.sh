#!/usr/bin/env bash
set -euo pipefail
python train.py
python evaluate.py
python bootstrap_eval.py
echo "All experiments finished. Check outputs/"

