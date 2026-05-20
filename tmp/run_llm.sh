#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hostname> <query>"
  exit 1
fi

HOST="$1"
shift
QUERY="$*"
MODEL_ID="mlx-community/Kimi-K2-Thinking"
ENCODED_MODEL_ID=$(
  python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$MODEL_ID"
)

if ! AWAIT_RESPONSE=$(curl -fsS --max-time 35 \
  "http://$HOST:52415/instance/await?model_id=$ENCODED_MODEL_ID&timeout_seconds=30" |
  awk '/^data: / { sub(/^data: /, ""); print; exit }'); then
  echo "No placed instance found for $MODEL_ID" >&2
  exit 1
fi

if ! printf '%s' "$AWAIT_RESPONSE" |
  python3 -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("type") == "ready" else 1)'; then
  echo "No placed instance found for $MODEL_ID" >&2
  exit 1
fi

curl -sN -X POST "http://$HOST:52415/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"$MODEL_ID\",
        \"stream\": true,
        \"messages\": [{ \"role\": \"user\",   \"content\": \"$QUERY\"}]
      }" |
  grep --line-buffered '^data:' |
  grep --line-buffered -v 'data: \[DONE\]' |
  cut -d' ' -f2- |
  jq -r --unbuffered '.choices[].delta.content // empty' |
  awk '{ORS=""; print; fflush()} END {print "\n"}'
