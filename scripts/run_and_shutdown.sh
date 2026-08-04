#!/usr/bin/env bash
# run_and_shutdown.sh — run a command, then self-TERMINATE the Lambda instance.
#
# IMPORTANT: OS `shutdown`/`poweroff` does NOT stop Lambda billing — only
# terminating the instance (via the Cloud API) does. This script does that.
# Outputs written under data/ live on the persistent filesystem, so termination
# loses nothing.
#
# Usage (from the repo root, AFTER `source scripts/setup_box.sh`):
#   export LAMBDA_API_KEY=<your lambda cloud api key>     # dashboard → API keys
#   export INSTANCE_ID=<this instance's id>               # see lookup below
#   nohup bash scripts/run_and_shutdown.sh \
#         python scripts/01_generate_traces.py --n-questions 300 --k 8 \
#         > run.log 2>&1 &
#   tail -f run.log      # watch a moment, Ctrl-C to detach, then sleep
#
# Find your INSTANCE_ID:
#   curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances
#
# MAX_HOURS caps runtime as a hang safety-net (default 3): even if the job wedges,
# the box terminates after this long instead of billing all night.

set -u
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY (dashboard → API keys)}"
: "${INSTANCE_ID:?set INSTANCE_ID (curl .../v1/instances to find it)}"
MAX_HOURS="${MAX_HOURS:-3}"

if [ "$#" -eq 0 ]; then
  echo "ERROR: pass the command to run, e.g. python scripts/01_generate_traces.py ..." >&2
  exit 2
fi

echo "=== run start $(date -u) | cap ${MAX_HOURS}h | cmd: $* ==="
# -k 30s: if the job ignores SIGTERM at the cap, force-kill 30s later.
timeout -k 30s "${MAX_HOURS}h" "$@"
code=$?
echo "=== job exited code=$code at $(date -u) ==="

echo "=== terminating instance $INSTANCE_ID ==="
curl -s -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -u "$LAMBDA_API_KEY:" \
  -H "Content-Type: application/json" \
  -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
echo
echo "=== terminate request sent $(date -u) — verify in the dashboard when you wake ==="
