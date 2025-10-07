#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Args & prerequisites
###############################################################################
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <git_command> [git_args...]" >&2
  echo "Examples:" >&2
  echo "  $0 pull" >&2
  echo "  $0 checkout main" >&2
  echo "  $0 status" >&2
  echo "  $0 fetch --all" >&2
  exit 1
fi

GIT_CMD="$*" # All args form the git command
HOSTS_FILE=${HOSTS_FILE:-hosts.txt}

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
# Run git command on remote hosts (parallel)
###############################################################################
echo ""
echo "=== Running 'git $GIT_CMD' on ${#HOSTS[@]} remote host(s) ==="
fail=0
for h in "${HOSTS[@]}"; do
  (
    echo "→ Running on $h..."
    if run_remote "$h" "cd ~/exo-v2 && git $GIT_CMD"; then
      echo "  ✓ $h: success"
    else
      echo "  ❌ $h: failed"
      exit 1
    fi
  ) || fail=1 &
done
wait

echo ""
if ((fail == 0)); then
  echo "🎉 Git command executed successfully on all hosts!"
else
  echo "⚠️  Some hosts failed—see above."
  exit 1
fi
