#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 6: Inject animacy-based case markers ==="
python -m perturbation.perturb_with_model --mode rule --strategy A+P
python -m perturbation.perturb_with_model --mode heuristic --strategy A+P
python -m perturbation.perturb_with_model --mode rule --strategy A_only
python -m perturbation.perturb_with_model --mode rule --strategy P_only
python -m perturbation.perturb_with_model --mode none --strategy A+P
python -m perturbation.perturb_with_model --mode full --strategy A+P

echo "=== ALL DONE ==="
