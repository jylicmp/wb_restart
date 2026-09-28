#!/bin/bash
# Pass site resource flags to sbatch; this does not launch a production run.
set -euo pipefail
: "${WB_REPO:?}" "${WB_ENV:?}" "${WB_ORIGINAL_ENV:?}" "${WB_WORK:?}" "${WB_SOURCE:?}" "${WB_TB_FILE:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_USAGE_STATS_ENABLED=0
mkdir -p "$WB_WORK"
cd "$WB_WORK"
retry_args=()
if [[ -n "${WB_SAMPLE_INDICES:-}" ]]; then retry_args=(--indices "$WB_SAMPLE_INDICES"); fi
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/validate_crse.py" \
  --source "$WB_SOURCE" --work "$WB_WORK/crse-sample" \
  --config "$WB_REPO/examples/crse_config.py" \
  --original-package "$WB_ORIGINAL_ENV/lib/python3.12/site-packages/wannierberri" \
  --baseline "$WB_REPO/provenance/baseline.json" --cpus 4 "${retry_args[@]}"
echo CRSE_VALIDATION_PASSED
