#!/usr/bin/env bash
# bulk_scp.sh — Sync a local repo to many hosts, respecting .gitignore and continuing even if
# some hosts fail. Tested on macOS Bash 3.x.
#
# ------------ User-tunable variables ------------
LOCAL_DIR="."      # Local directory you want to send
REMOTE_DIR="~/exo-v2"         # Destination directory on the remote machines
HOSTS_FILE="hosts.json"       # JSON array of hosts (["user@ip", ...])
# ------------ End of user-tunable section -------

set -uo pipefail   # Treat unset vars as error; fail pipelines, but we handle exit codes ourselves

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <password>" >&2
  exit 1
fi
PASSWORD="$1"

# Dependency checks
for cmd in sshpass jq rsync git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

# Verify hosts file exists
if [ ! -f "$HOSTS_FILE" ]; then
  echo "Error: Hosts file '$HOSTS_FILE' not found." >&2
  exit 1
fi

# Build a temporary exclude file containing every Git‑ignored path
EXCLUDE_FILE=$(mktemp)
trap 'rm -f "$EXCLUDE_FILE"' EXIT

if git -C "$LOCAL_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$LOCAL_DIR" ls-files -z -o -i --exclude-standard \
    | tr '\0' '\n' > "$EXCLUDE_FILE"
else
  # Fallback: just use top‑level .gitignore if present
  [ -f "$LOCAL_DIR/.gitignore" ] && cat "$LOCAL_DIR/.gitignore" > "$EXCLUDE_FILE"
fi

# Iterate over hosts — process substitution keeps stdin free for rsync/ssh
while IFS= read -r TARGET || [ -n "$TARGET" ]; do
  [ -z "$TARGET" ] && continue  # skip blanks
  echo "\n—— Syncing $LOCAL_DIR → $TARGET:$REMOTE_DIR ——"

#   # Ensure remote directory exists (ignore failure but report)
#   if ! sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$TARGET" "mkdir -p $REMOTE_DIR" </dev/null; then
#     echo "✗ Failed to create $REMOTE_DIR on $TARGET" >&2
#     continue  # move on to next host
#   fi

  # Rsync with checksums; redirect stdin so rsync/ssh can't eat host list
  if sshpass -p "$PASSWORD" rsync -azc --delete --exclude-from="$EXCLUDE_FILE" \
       -e "ssh -o StrictHostKeyChecking=no" \
       "$LOCAL_DIR/" "$TARGET:$REMOTE_DIR/" </dev/null; then
    echo "✓ Success: $TARGET"
  else
    echo "✗ Failed:  $TARGET" >&2
  fi

done < <(jq -r '.[]' "$HOSTS_FILE")
