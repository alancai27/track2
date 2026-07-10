#!/usr/bin/env bash
# Local end-to-end test on the 3 example clips.
# Usage: put GROQ_API_KEY (and optionally VISION_MODEL / STYLE_MODEL)
# in .env, then:  ./run_examples.sh          (python, no docker)
#                                 ./run_examples.sh docker    (built image)
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p io/input io/output
cat > io/input/tasks.json <<'EOF'
[
  {"task_id":"v1","video_url":"https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4","styles":["formal","sarcastic","humorous_tech","humorous_non_tech"]},
  {"task_id":"v2","video_url":"https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4","styles":["formal","sarcastic","humorous_tech","humorous_non_tech"]},
  {"task_id":"v3","video_url":"https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4","styles":["formal","sarcastic","humorous_tech","humorous_non_tech"]}
]
EOF

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
