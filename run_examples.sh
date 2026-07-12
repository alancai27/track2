#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p io/input io/output
cp examples/tasks.json io/input/tasks.json

if [[ "${1:-}" == "docker" ]]; then
  docker run --rm --env-file .env \
    -v "$PWD/io/input:/input" -v "$PWD/io/output:/output" \
    ghcr.io/alancai27/amd_track2:latest
else
  set -a; [[ -f .env ]] && source .env; set +a
  INPUT_PATH=io/input/tasks.json OUTPUT_PATH=io/output/results.json \
    python app/entrypoint.py
fi

echo "----- results -----"
python -m json.tool io/output/results.json
