#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Args & prerequisites
###############################################################################
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <PASSWORD> <git_command> [git_args...]" >&2
  echo "Examples:" >&2
  echo "  $0 mypassword pull" >&2
  echo "  $0 mypassword checkout main" >&2
  echo "  $0 mypassword status" >&2
  echo "  $0 mypassword fetch --all" >&2
  exit 1
fi

PASSWORD=$1
shift  # Remove password from args
GIT_CMD="$*"  # Remaining args form the git command
HOSTS_FILE=${HOSTS_FILE:-hosts.json}

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
# Run git command locally
###############################################################################
echo "=== Running 'git $GIT_CMD' locally ==="
if (cd ~/exo && git $GIT_CMD); then
  echo "✓ Local git command succeeded"
else
  echo "❌ Local git command failed"
  exit 1
fi

###############################################################################
# Run git command on remote hosts (parallel)
###############################################################################
echo ""
echo "=== Running 'git $GIT_CMD' on ${#HOSTS[@]} remote host(s) ==="
fail=0
for h in "${HOSTS[@]}"; do
  (
    echo "→ Running on $h..."
    if run_remote "$h" "cd ~/exo && git $GIT_CMD"; then
      echo "  ✓ $h: success"
    else
      echo "  ❌ $h: failed"
      exit 1
    fi
  ) || fail=1 &
done
wait

echo ""
if (( fail == 0 )); then
  echo "🎉 Git command executed successfully on all hosts!"
else
  echo "⚠️  Some hosts failed—see above."
  exit 1
fi