#!/bin/bash

INPUT_DIR="evaluation/BLiMP_raw"
OUTPUT_BASE="evaluation/BLiMP_perturbed"

DATASETS=(
  "animate_subject_trans"
  "regular_plural_subject_verb_agreement_1"
  "transitive"
)

declare -A CONFIGS
CONFIGS["rule_A_only"]="rule A_only"
CONFIGS["rule_A+P"]="rule A+P"
CONFIGS["rule_full"]="rule full"
CONFIGS["heuristic_A+P"]="heuristic A+P"

for DATASET in "${DATASETS[@]}"; do
  JSONL="${INPUT_DIR}/${DATASET}.jsonl"

  if [[ ! -f "$JSONL" ]]; then
    echo "Warning: File not found: $JSONL"
    continue
  fi

  echo ""
  echo "=== Processing dataset: $DATASET ==="

  for CONFIG_NAME in "${!CONFIGS[@]}"; do
    IFS=" " read -r MODE STRATEGY <<< "${CONFIGS[$CONFIG_NAME]}"
    OUT_DIR="${OUTPUT_BASE}/${DATASET}"
    mkdir -p "$OUT_DIR"
    OUT_PATH="${OUT_DIR}"

    if [[ -f "$OUT_PATH" ]]; then
      echo "→ Skipping existing: $OUT_PATH"
      continue
    fi

    echo "→ Running: MODE=$MODE | STRATEGY=$STRATEGY"
    echo "  Output: $OUT_PATH"

    python -m evaluation.perturb_blimp_pairs \
      --blimp "$JSONL" \
      --out "$OUT_PATH" \
      --mode "$MODE" \
      --strategy "$STRATEGY"
  done
done
