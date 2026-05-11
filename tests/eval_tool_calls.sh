#!/usr/bin/env bash

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
hosts=("$@")
cleanup() {
  for host in "${hosts[@]}"; do
    ssh -T -o BatchMode=yes "$host@$host" "pkill -f bin/exo" &
  done
  sleep 1
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap 'cleanup' EXIT INT TERM

for host; do
  ssh -T -o BatchMode=yes -o ServerAliveInterval=30 "$host@$host" \
    "EXO_LIBP2P_NAMESPACE=$commit /nix/var/nix/profiles/default/bin/nix build github:exo-explore/exo/$commit" &
done
wait
for host; do
  ssh -T -o BatchMode=yes -o ServerAliveInterval=30 "$host@$host" \
    "EXO_LIBP2P_NAMESPACE=$commit /nix/var/nix/profiles/default/bin/nix run github:exo-explore/exo/$commit" &>/dev/null &
done

for host; do
  echo "Waiting for $host..." 1>&2
  until curl -sf "http://$host:52415/models" &>/dev/null; do sleep 1; done
done

echo "Waiting 30s for cluster setup" 1>&2
sleep 30
echo "EXO loaded" 1>&2
eval_runner="${hosts[0]}"
mkdir -p "./bench/$commit"
nix run .#exo-get-all-models-on-cluster -- "$eval_runner" | while IFS= read -r model; do
  echo "running eval for $model" 1>&2
  ssh -Tn -o BatchMode=yes -o ServerAliveInterval=30 "$eval_runner@$eval_runner" \
    "/nix/var/nix/profiles/default/bin/nix run github:exo-explore/exo/$commit#exo-eval-tool-calls -- --model $model --stdout" \
    >>"./bench/$commit/${model//\//--}-eval.json"
  echo
done
