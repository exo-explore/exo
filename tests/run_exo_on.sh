#!/usr/bin/env bash
set -euo pipefail

[ $# -lt 1 ] && {
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
hosts=("$@")
cleanup() {
  for host in "${hosts[@]}"; do
    ssh -T -o BatchMode=yes "$host@$host" "pkill -f bin/exo" &
  done
  wait
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap 'cleanup' EXIT INT TERM

colours=($'\e[31m' $'\e[32m' $'\e[33m' $'\e[34m')
reset=$'\e[0m'
i=0
for host; do
  colour=${colours[i++ % 4]}
  ssh -T -o BatchMode=yes -o ServerAliveInterval=30 "$host@$host" \
    "EXO_LIBP2P_NAMESPACE=$commit /nix/var/nix/profiles/default/bin/nix run github:exo-explore/exo/$commit" |&
    awk -v p="${colour}[${host}]${reset}" '{ print p $0; fflush() }' &
done

for host; do
  echo "Waiting for $host..."
  until curl -sf "http://$host:52415/models" &>/dev/null; do sleep 1; done
done
wait
