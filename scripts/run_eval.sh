#!/bin/bash

RUN_IDS=(
  "heuristic_A+P"
)

JSONL_PATH="evaluation/casemarking/heuristic_A+P/cbt_minimal_pairs.jsonl"

for RUN_ID in "${RUN_IDS[@]}"
do
  OUT_DIR="heuristic_A+P_pairs/${RUN_ID}"
  echo "Running: $RUN_ID"
  python -m evaluation.eval_minipairs \
    --run-id "$RUN_ID" \
    --jsonl "$JSONL_PATH" \
    --out-dir "$OUT_DIR"
done
