#!/usr/bin/env bash
set -euo pipefail

[ $# -eq 0 ] && { echo "Usage: $0 host1 [host2 ...]"; exit 1; }

[ -z "$(git status --porcelain)" ] || { echo "Uncommitted changes"; exit 1; }
commit=$(git rev-parse HEAD)
git fetch -q
git branch -r --contains "$commit" | grep -q origin || { echo "Not pushed to origin"; exit 1; }

echo "Deploying $commit to $# hosts..."

for host; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 "$host@$host" "
    cd exo && 
    git fetch -q &&
    git checkout -q $commit && 
    exec uv run tests/headless_runner.py
  " &
done

for host; do
  echo "Waiting for $host..."
  until curl -sf "http://$host:8000/models" >/dev/null 2>&1; do sleep 1; done
done

uv run tests/start_distributed_test.py "$@"
