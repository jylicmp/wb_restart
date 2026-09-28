#!/bin/bash
# Independent existing sample copies only. Resource flags are provided to sbatch.
set -euo pipefail
: "${WB_REPO:?}" "${WB_ENV:?}" "${WB_WORK:?}" "${WB_PARTIAL_SOURCE:?}" "${WB_TB_FILE:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WB_WORK"
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/validate_partial_export.py" \
  --source "$WB_PARTIAL_SOURCE" --work "$WB_WORK/export-check" \
  --config "$WB_REPO/examples/crse_config.py"
echo PARTIAL_EXPORT_VALIDATION_PASSED
