#!/usr/bin/env nix-shell
#!nix-shell -i bash -p mktorrent -p python3Packages.huggingface-hub -p git -p git-lfs
set -euo pipefail
set -x

MODEL="$1"

mkdir -p "$MODEL"

# Step 1: Clone/fetch the repo and get the hash of head
mkdir -p "$MODEL"
if test -d "$MODEL/git"; then
  # Assert that the origin is correct
  git -C "$MODEL/git" fetch
else
  git clone "https://huggingface.co/$MODEL" "$MODEL/git"
fi

HASH=$(git -C "$MODEL/git" rev-parse origin/main)
LARGE_FILES=$(git -C "$MODEL/git" lfs ls-files --all --name-only)

SMALL_DIR="$MODEL/$HASH-small"
LARGE_DIR="$MODEL/$HASH-large"
mkdir -p "$SMALL_DIR" "$LARGE_DIR"

# Step 2: Prepare files. Two torrents: one for large files and one for metadata.
git -C "$MODEL/git" archive "$HASH" | tar -x -C "$SMALL_DIR"
echo "$LARGE_FILES" | xargs -I{} rm "$SMALL_DIR/{}"

echo "$LARGE_FILES" | xargs hf download "$MODEL" --revision "$HASH" --local-dir "$LARGE_DIR" --cache-dir "$(realpath .cache)" --include
if test -d "$LARGE_DIR/.cache"; then
  echo ".cache created against our wishes, deleting it..."
  rm -r "$LARGE_DIR/.cache"
fi

# Step 3: Create both torrents
mkdir -p "torrents/$MODEL/"
SMALL_TORRENT_PATH="torrents/$MODEL/${HASH}.small.torrent"
LARGE_TORRENT_PATH="torrents/$MODEL/${HASH}.large.torrent"

mktorrent "$SMALL_DIR/" --output="$SMALL_TORRENT_PATH" \
  -n "$HASH" \
  --web-seed="https://huggingface.co/$MODEL/raw/" \
  --no-date \
  --announce="udp://tracker.opentrackr.org:1337/announce"
  # --private

mktorrent "$LARGE_DIR/" --output="$LARGE_TORRENT_PATH" \
  -n "$HASH" \
  --web-seed="https://huggingface.co/$MODEL/resolve/" \
  --piece-length=24 \
  --no-date \
  --announce="udp://tracker.opentrackr.org:1337/announce"
  # --private

echo "Successfully created torrent files in:"
echo "$SMALL_TORRENT_PATH"
echo "$LARGE_TORRENT_PATH"
