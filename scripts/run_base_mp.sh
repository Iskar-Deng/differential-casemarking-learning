#!/bin/bash

CHECKPOINT_DIR="checkpoints"
JSONL_PATH="evaluation/casemarking/rule_full/cbt.jsonl_minimal_pairs.jsonl"
OUT_DIR="results_raw_mp"

echo "Scanning checkpoints in: $CHECKPOINT_DIR"
for RUN_ID in $(ls "$CHECKPOINT_DIR"); do
    if [ -d "$CHECKPOINT_DIR/$RUN_ID" ]; then
        echo ""
        echo "Running eval_minipairs for: $RUN_ID"
        python -m evaluation.eval_minipairs \
            --run-id "$RUN_ID" \
            --jsonl "$JSONL_PATH" \
            --out-dir "$OUT_DIR"

        if [ $? -ne 0 ]; then
            echo "Evaluation failed for $RUN_ID"
        fi
    fi
done
