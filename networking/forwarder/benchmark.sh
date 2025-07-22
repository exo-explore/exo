#!/usr/bin/env bash
set -euo pipefail

NUM_RECORDS="${1:-10000}"
BATCH_SIZE="${2:-100}"

echo "Running burst benchmark with $NUM_RECORDS records in batches of $BATCH_SIZE..."

# Build the forwarder binary
BIN_PATH="$(pwd)/forwarder_bin"
BUILD_TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t forwarder-build)"
export TMPDIR="$BUILD_TMPDIR"

pushd . >/dev/null
go build -o "$BIN_PATH" .
popd >/dev/null

# Temporary workspace
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t forwarder-burst)"
SRC_DB="$TMP_DIR/src.db"
DST_DB="$TMP_DIR/dst.db"
TABLE="records"
TOPIC="burst_topic_$$"

# Cleanup function
cleanup() {
  echo "Cleaning up…"
  kill "${PID1:-}" "${PID2:-}" 2>/dev/null || true
  wait "${PID1:-}" "${PID2:-}" 2>/dev/null || true
  rm -rf "$TMP_DIR" "$BIN_PATH" "$BUILD_TMPDIR"
}
trap cleanup EXIT

# Create databases with WAL mode
sqlite3 "$SRC_DB" <<SQL
.timeout 5000
PRAGMA journal_mode=WAL;
SQL

sqlite3 "$DST_DB" <<SQL
.timeout 5000
PRAGMA journal_mode=WAL;
SQL

# Start forwarder nodes
"$BIN_PATH" -node-id node1 "sqlite:${SRC_DB}:${TABLE}|libp2p:${TOPIC}" >"$TMP_DIR/node1.log" 2>&1 &
PID1=$!

"$BIN_PATH" -node-id node2 "libp2p:${TOPIC}|sqlite:${DST_DB}:${TABLE}" >"$TMP_DIR/node2.log" 2>&1 &
PID2=$!

# Give nodes time to start
sleep 3

echo "Inserting $NUM_RECORDS records in batches of $BATCH_SIZE..."
START_NS=$(date +%s%N)

# Insert records in batches for high throughput
for batch_start in $(seq 1 $BATCH_SIZE $NUM_RECORDS); do
  batch_end=$((batch_start + BATCH_SIZE - 1))
  if [ $batch_end -gt $NUM_RECORDS ]; then
    batch_end=$NUM_RECORDS
  fi
  
  # Build values for batch insert
  values=""
  for i in $(seq $batch_start $batch_end); do
    if [ -n "$values" ]; then
      values="$values,"
    fi
    values="$values('seednode','seedpath',$i,datetime('now'),'{}')"
  done
  
  # Insert batch
  sqlite3 -cmd ".timeout 5000" "$SRC_DB" \
    "INSERT INTO ${TABLE} (source_node_id, source_path, source_row_id, source_timestamp, data) VALUES $values;"
  
  # Small delay to prevent overwhelming
  sleep 0.01
done

echo "Waiting for destination to catch up..."

# Wait for completion
while true; do
  dest_count=$(sqlite3 -cmd ".timeout 5000" "$DST_DB" "SELECT IFNULL(COUNT(*),0) FROM ${TABLE};" 2>/dev/null || echo 0)
  if [[ "$dest_count" -ge "$NUM_RECORDS" ]]; then
    break
  fi
  echo "Progress: $dest_count / $NUM_RECORDS"
  sleep 1
done

END_NS=$(date +%s%N)
DURATION_NS=$((END_NS-START_NS))
THROUGHPUT=$(echo "scale=2; $NUM_RECORDS*1000000000/$DURATION_NS" | bc)

echo "Forwarded $NUM_RECORDS records in $(printf '%.2f' "$(echo "$DURATION_NS/1000000000" | bc -l)") seconds — $THROUGHPUT records/s"

# Show some logs
echo ""
echo "=== Node1 Log (last 10 lines) ==="
tail -10 "$TMP_DIR/node1.log"
echo ""
echo "=== Node2 Log (last 10 lines) ==="
tail -10 "$TMP_DIR/node2.log"