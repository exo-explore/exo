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
((fail == 0)) || {
  echo "❌ Some hosts could not be reached—check SSH access."
  exit 1
}
echo "✓ exo processes killed on all reachable hosts."
#
###############################################################################
# Phase 2 – cleanup database files (parallel)
###############################################################################
echo "=== Stage 2: cleaning up database files ==="
fail=0
for h in "${HOSTS[@]}"; do
  (
    run_remote "$h" 'rm -f ~/.exo/*db* || true'
  ) || fail=1 &
done
wait
((fail == 0)) || {
  echo "❌ Some hosts failed database cleanup."
  exit 1
}
echo "✓ Database files cleaned on all hosts."

###############################################################################
# Phase 3 – start new exo processes in Terminal windows (parallel)
###############################################################################
echo "=== Stage 3: starting new exo processes ==="
fail=0
for h in "${HOSTS[@]}"; do
  # Use osascript to open Terminal windows on remote Mac
  remote_cmd="osascript -e \"tell app \\\"Terminal\\\" to do script \\\"cd ~/exo-v2; nix develop --command uv run exo\\\"\""

  (run_remote "$h" "$remote_cmd") || fail=1 &
done
wait
((fail == 0)) && echo "🎉 Deployment finished!" || {
  echo "⚠️  Some starts failed—see above."
  exit 1
}
