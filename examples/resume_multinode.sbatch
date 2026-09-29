#!/bin/bash
# Resource flags and private paths are supplied at submission.
set -euo pipefail
: "${WB_ENV:?}" "${WB_REPO:?}" "${WB_TEST_ROOT:?}" "${WB_SOURCE:?}" "${WB_TB_FILE:?}"
export PATH="$WB_ENV/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAY_USAGE_STATS_ENABLED=0
unset PYTHONPATH RAY_ADDRESS ip_head redis_password
control="$WB_TEST_ROOT/_restart_test"
cd "$WB_TEST_ROOT"
"$WB_ENV/bin/python" -B -c 'import json,os; from pathlib import Path; p=Path(os.environ["WB_TEST_ROOT"]).resolve(); original=Path(os.environ["WB_SOURCE"]).resolve().parent; assert p!=original and original not in p.parents; assert json.loads((p/"_restart_test/copy-state.json").read_text())["status"]=="complete"'
mkdir -p "$WB_TEST_ROOT/restart-results"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} -eq 4 ]]
head_ip=$(srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 -w "${nodes[0]}" hostname -I | awk '{print $1}')
# Ray reserves 10002-19999 for worker processes by default. Keep the GCS port
# outside that range while deriving a stable, job-specific value.
port=$((20000 + SLURM_JOB_ID % 10000))
address="$head_ip:$port"
ray_tmp="/tmp/wbi-$SLURM_JOB_ID"
steps=()
cleanup() {
  status=$?
  printf '%s\n' "$status" > "$control/exit-code"
  if ((${#steps[@]})); then kill "${steps[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 -w "${nodes[0]}" \
  "$WB_ENV/bin/ray" start --head --node-ip-address="$head_ip" --port="$port" \
  --num-cpus=30 --num-gpus=0 --object-store-memory=34359738368 \
  --include-dashboard=false --temp-dir="$ray_tmp" --disable-usage-stats --block \
  > "$control/ray-head-$SLURM_JOB_ID.log" 2>&1 &
steps+=("$!")
"$WB_ENV/bin/python" -B -c 'import socket,sys,time; host,port=sys.argv[1],int(sys.argv[2]); deadline=time.monotonic()+120
while True:
 try:
  with socket.create_connection((host,port),timeout=1):break
 except OSError:
  if time.monotonic()>deadline:raise TimeoutError("Ray head did not listen")
  time.sleep(1)' "$head_ip" "$port"
for node in "${nodes[@]:1}"; do
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 -w "$node" \
    "$WB_ENV/bin/ray" start --address="$address" --num-cpus=32 --num-gpus=0 \
    --object-store-memory=34359738368 --temp-dir="$ray_tmp" --disable-usage-stats --block \
    > "$control/ray-$node-$SLURM_JOB_ID.log" 2>&1 &
  steps+=("$!")
done
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/check_ray_cluster.py" --address "$address" \
  --hosts "${nodes[@]}" --output "$control/cluster.json"
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/restart_run.py" \
  --config "$WB_REPO/examples/crse_config.py" --checkpoint "$WB_TEST_ROOT/_tmp_wb" \
  --recover --on-corrupt recompute --until-iteration 1 --ray-address "$address" \
  --output "$WB_TEST_ROOT/restart-results/CrSe_Qorb_GaoXiao" \
  --partial-interval 900 --partial-output "$WB_TEST_ROOT/partial" \
  2>&1 | tee "$control/restart-$SLURM_JOB_ID.log"
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/validate_integration.py" --root "$WB_TEST_ROOT"
echo INTEGRATION_VALIDATION_PASSED
