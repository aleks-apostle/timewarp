#!/usr/bin/env bash
set -euo pipefail

DB=./timewarp.db
BLOBS=./blobs

if [[ $# -lt 2 ]]; then
  echo "Usage: examples/diff_demo.sh <run_a> <run_b>"
  exit 2
fi

RUN_A=$1
RUN_B=$2

python -m timewarp $DB $BLOBS diff $RUN_A $RUN_B --window 7 --json

