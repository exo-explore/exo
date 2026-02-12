#!/bin/bash
# Usage: ./download_glm5_shard.sh <start> <end> [local_dir]
#
# Split across 4 Macs:
#   Mac 1: ./download_glm5_shard.sh 1 71
#   Mac 2: ./download_glm5_shard.sh 72 141
#   Mac 3: ./download_glm5_shard.sh 142 212
#   Mac 4: ./download_glm5_shard.sh 213 282

set -euo pipefail

START=${1:?Usage: $0 <start> <end> [local_dir]}
END=${2:?Usage: $0 <start> <end> [local_dir]}
LOCAL_DIR="${3:-GLM-5}"

INCLUDES=()
for i in $(seq "$START" "$END"); do
  INCLUDES+=(--include "$(printf 'model-%05d-of-00282.safetensors' "$i")")
done

echo "Downloading safetensors $START-$END to $LOCAL_DIR"
hf download zai-org/GLM-5 "${INCLUDES[@]}" --local-dir "$LOCAL_DIR"
