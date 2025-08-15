#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Args & prerequisites
###############################################################################
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <PASSWORD> [hosts_file]" >&2 ; exit 1
fi
PASSWORD=$1
HOSTS_FILE=${2:-hosts.json}

for prog in jq sshpass; do
  command -v "$prog" >/dev/null ||
    { echo "Error: $prog not installed."; exit 1; }
done

###############################################################################
# Load hosts.json (works on macOS Bash 3.2 and Bash 4+)
###############################################################################
if builtin command -v mapfile >/dev/null 2>&1; then
  mapfile -t HOSTS < <(jq -r '.[]' "$HOSTS_FILE")
else
  HOSTS=()
  while IFS= read -r h; do HOSTS+=("$h"); done < <(jq -r '.[]' "$HOSTS_FILE")
fi
[[ ${#HOSTS[@]} -gt 0 ]] || { echo "No hosts found in $HOSTS_FILE"; exit 1; }

###############################################################################
# Helper – run a remote command and capture rc/stderr/stdout
###############################################################################
ssh_opts=(-o StrictHostKeyChecking=no
          -o NumberOfPasswordPrompts=1   # allow sshpass to answer exactly once
          -o LogLevel=ERROR)

run_remote () {                  # $1 host   $2 command
  local host=$1 cmd=$2 rc
  if sshpass -p "$PASSWORD" ssh "${ssh_opts[@]}" "$host" "$cmd"; then
    rc=0
  else
    rc=$?
  fi
  return $rc
}

###############################################################################
# Phase 1 – kill exo everywhere (parallel)
###############################################################################
echo "=== Stage 1: killing exo on ${#HOSTS[@]} host(s) ==="
fail=0
for h in "${HOSTS[@]}"; do
  (
    run_remote "$h" 'pkill -f exo || true'
  ) || fail=1 &
done
wait
(( fail == 0 )) || { echo "❌ Some hosts could not be reached—check password or SSH access."; exit 1; }
echo "✓ exo processes killed on all reachable hosts."

###############################################################################
# Phase 2 – start new exo processes (parallel, with sudo -S)
###############################################################################
echo "=== Stage 2: starting new exo processes ==="
fail=0
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}

  # one liner that pre-caches sudo and then runs the script
  if [[ $i -eq 0 ]]; then
    remote_cmd="cd ~/exo && ./run.sh -c"
  else
    remote_cmd="cd ~/exo && ./run.sh -rc"
  fi

  ( run_remote "$h" "$remote_cmd" ) || fail=1 &
done
wait
(( fail == 0 )) && echo "🎉 Deployment finished!" || \
  { echo "⚠️  Some starts failed—see above."; exit 1; }
