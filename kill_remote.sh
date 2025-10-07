#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Args & prerequisites
###############################################################################
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [hosts_file]" >&2
  exit 1
fi
HOSTS_FILE=${1:-hosts.txt}

###############################################################################
# Load hosts.txt (works on macOS Bash 3.2 and Bash 4+)
###############################################################################
if [[ ! -f "$HOSTS_FILE" ]]; then
  echo "Error: $HOSTS_FILE not found"
  exit 1
fi

if builtin command -v mapfile >/dev/null 2>&1; then
  mapfile -t HOSTS <"$HOSTS_FILE"
else
  HOSTS=()
  while IFS= read -r h; do
    [[ -n "$h" ]] && HOSTS+=("$h")
  done <"$HOSTS_FILE"
fi
[[ ${#HOSTS[@]} -gt 0 ]] || {
  echo "No hosts found in $HOSTS_FILE"
  exit 1
}

###############################################################################
# Helper – run a remote command and capture rc/stderr/stdout
###############################################################################
ssh_opts=(-o StrictHostKeyChecking=no
  -o LogLevel=ERROR)

run_remote() { # $1 host   $2 command
  local host=$1 cmd=$2 rc
  if ssh "${ssh_opts[@]}" "$host" "$cmd"; then
    rc=0
  else
    rc=$?
  fi
  return $rc
}

###############################################################################
# Kill exo everywhere (parallel)
###############################################################################
echo "=== Killing exo on ${#HOSTS[@]} host(s) ==="
fail=0
for h in "${HOSTS[@]}"; do
  (
    run_remote "$h" 'pkill -f exo || true'
  ) || fail=1 &
done
wait
((fail == 0)) || {
  echo "❌ Some hosts could not be reached—check SSH access."
  exit 1
}
echo "✓ exo processes killed on all reachable hosts."