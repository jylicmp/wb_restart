#!/bin/bash
set -euo pipefail
: "${WB_ENV:?}" "${WB_REPO:?}" "${WB_COPY_SOURCE:?}" "${WB_TEST_ROOT:?}"
cd "$WB_REPO/.local"
"$WB_ENV/bin/python" -u -B "$WB_REPO/scripts/copy_live_tree.py" \
  --source "$WB_COPY_SOURCE" --target "$WB_TEST_ROOT" --workers 4
echo LIVE_COPY_VERIFIED
