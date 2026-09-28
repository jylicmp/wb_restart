#!/bin/bash
# Pass partition, QOS, nodes, CPUs, memory, time and log paths to sbatch.
set -euo pipefail
: "${WB_REPO:?}" "${WB_ENV:?}" "${WB_ORIGINAL_ENV:?}" "${WB_WORK:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_USAGE_STATS_ENABLED=0
mkdir -p "$WB_WORK"
cd "$WB_WORK"
"$WB_ENV/bin/python" -B -W ignore::ResourceWarning -m unittest discover -v -s "$WB_REPO/tests"
"$WB_ORIGINAL_ENV/bin/python" -B "$WB_REPO/tests/baseline_case.py" "$WB_WORK/original"
"$WB_ENV/bin/python" -B "$WB_REPO/tests/baseline_case.py" "$WB_WORK/new"
"$WB_ENV/bin/python" -B -c 'import numpy as np, json; a=np.load("original/tensors.npz"); b=np.load("new/tensors.npz"); [np.testing.assert_allclose(a[k],b[k],rtol=1e-10,atol=1e-12) for k in a.files]; print(json.dumps({k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files}))'
"$WB_ENV/bin/python" -B "$WB_REPO/tests/ray_case.py" "$WB_WORK/ray-case"
echo SMALL_VALIDATION_PASSED
