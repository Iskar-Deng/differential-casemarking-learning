#!/bin/bash
set -e

CONFIGS=(
  rule_A_only
  heuristic_A+P
)

for cfg in "${CONFIGS[@]}"; do
  echo ">>> Training: $cfg"
  python mistral/train.py --config mistral/conf/user_main/${cfg}.yaml
done
