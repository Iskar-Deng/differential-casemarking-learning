#!/usr/bin/env bash
set -e

RUNS=(rule_A_only rule_A+P rule_A+P_with_invalid rule_full rule_none rule_none_with_invalid)

for r in "${RUNS[@]}"; do
  echo ">>> Running $r"
  python -m evaluation.eval_ppl --run-id "$r" --out-dir results
done
