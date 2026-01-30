#!/usr/bin/env bash
set -euo pipefail

[ $# -eq 0 ] && {
  echo "Usage: $0 host1 [host2 ...]"
  exit 1
}

[ -z "$(git status --porcelain)" ] || {
  echo "Uncommitted changes"
  exit 1
}
commit=$(git rev-parse HEAD)
git fetch -q origin
git branch -r --contains "$commit" | grep -qE '^\s*origin/' || {
  echo "Not pushed to origin"
  exit 1
}
echo "Deploying $commit to $# hosts..."

pids=""
trap 'xargs -r kill 2>/dev/null <<<"$pids" || true' EXIT INT TERM
colours=($'\e[31m' $'\e[32m' $'\e[33m' $'\e[34m')
reset=$'\e[0m'
i=0

for host; do
  colour=${colours[i++ % 4]}
  ssh -t -o BatchMode=yes -o ServerAliveInterval=30 "$host@$host" "/usr/bin/env bash -lc '
    set -euo pipefail
    cd exo 
    git fetch -q origin
    git checkout -q $commit
    nix develop -c uv sync
    uv run exo
    '" 2>&1 | sed -u "s/^/${colour}[${host}]${reset}/" &
  pids+=" $!"
done
wait
