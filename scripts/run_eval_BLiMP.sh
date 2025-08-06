#!/bin/bash

DATASET="regular_plural_subject_verb_agreement_1"

INPUT_DIR="evaluation/BLiMP_perturbed/${DATASET}"

OUT_DIR="results_blimp/${DATASET}"

mkdir -p "$OUT_DIR"

declare -A RUNS
RUNS["rule_A+P"]="rule_A+P"
RUNS["rule_A+P_with_invalid"]="rule_A+P"
RUNS["rule_A_only"]="rule_A_only"
RUNS["rule_full"]="rule_full"
RUNS["heuristic_A+P"]="heuristic_A+P"
RUNS["rule_none"]=""
RUNS["rule_none_with_invalid"]=""

for RUN_ID in "${!RUNS[@]}"; do
  SUFFIX="${RUNS[$RUN_ID]}"
  if [[ -z "$SUFFIX" ]]; then
    JSONL="evaluation/BLiMP_raw/${DATASET}.jsonl"
  else
    JSONL="${INPUT_DIR}/${DATASET}_${SUFFIX}.jsonl"
  fi

  if [[ ! -f "$JSONL" ]]; then
    echo "[!] Missing input: $JSONL"
    continue
  fi

  echo "→ Running $RUN_ID on $DATASET"
  python -m evaluation.eval_minipairs \
    --run-id "$RUN_ID" \
    --jsonl "$JSONL" \
    --out-dir "$OUT_DIR"
done
