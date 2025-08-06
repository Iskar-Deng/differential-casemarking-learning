#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

run() {
  local MODE="$1"
  local STRAT="$2"
  echo ">>> Running: mode=${MODE}, strategy=${STRAT}"
  python -m perturbation.perturb_with_model \
    --mode "${MODE}" \
    --strategy "${STRAT}" 2>&1
}

run rule      full
