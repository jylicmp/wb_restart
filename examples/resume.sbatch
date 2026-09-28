#!/bin/bash
# Production template only; no automatic submission. Set site resource flags.
set -euo pipefail
: "${WB_REPO:?}" "${WB_ENV:?}" "${WB_TB_FILE:?}" "${WB_SOURCE:?}" "${WB_CHECKPOINT:?}" "${WB_OUTPUT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_USAGE_STATS_ENABLED=0
# Protect the original task tree, including paths hidden behind symlinks.
"$WB_ENV/bin/python" -B -c 'import os; from pathlib import Path; source=Path(os.environ["WB_SOURCE"]).resolve().parent; paths=[Path(os.environ[k]).resolve() for k in ("WB_CHECKPOINT","WB_OUTPUT_DIR")]; assert all(p!=source and source not in p.parents for p in paths), "Original task directory is read-only"'
mkdir -p "$WB_OUTPUT_DIR"
cd "$WB_OUTPUT_DIR"
# Single-node continuation: Ray starts here and uses the CPUs in this allocation.
# For multiple nodes, start an allocation-local Ray cluster before this command.
"$WB_ENV/bin/python" -B "$WB_REPO/scripts/restart_run.py" \
  --config "$WB_REPO/examples/crse_config.py" --checkpoint "$WB_CHECKPOINT" \
  --recover --on-corrupt recompute --until-iteration 20 \
  --output "$WB_OUTPUT_DIR/CrSe_Qorb_GaoXiao"
