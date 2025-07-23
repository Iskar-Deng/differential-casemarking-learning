#!/bin/bash

set -e

MODES=("rule" "heuristic")
STRATEGIES=("A+P" "A_only" "P_only")

for MODE in "${MODES[@]}"; do
  for STRAT in "${STRATEGIES[@]}"; do
    OUTFILE="data/perturbed_model/${MODE}_${STRAT}_affected.txt"
    
    if [ -f "$OUTFILE" ]; then
      echo "Skipping mode=$MODE strategy=$STRAT (already exists)"
      continue
    fi

    echo "Running mode=$MODE strategy=$STRAT"
    python -m perturbation.perturb_with_model --mode "$MODE" --strategy "$STRAT"
  done
done
