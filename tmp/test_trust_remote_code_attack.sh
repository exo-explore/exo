#!/usr/bin/env bash
# Test that models added via API get trust_remote_code=false
# Run this against a running exo instance.
# Usage: ./test_trust_remote_code_attack.sh [host:port]

set -uo pipefail

HOST="${1:-localhost:52415}"
MODEL_ID="KevTheHermit/security-testing"
CUSTOM_CARDS_DIR="$HOME/.exo/custom_model_cards"
CARD_FILE="$CUSTOM_CARDS_DIR/KevTheHermit--security-testing.toml"

echo "=== Test: trust_remote_code attack via API ==="
echo "Target: $HOST"
echo ""

# Clean up RCE proof from previous runs
rm -f /tmp/exo-rce-proof.txt

# Step 0: Clean up any stale card from previous runs
if [ -f "$CARD_FILE" ]; then
  echo "[0] Removing stale card from previous run ..."
  curl -s -X DELETE \
    "http://$HOST/models/custom/$(python3 -c 'import urllib.parse; print(urllib.parse.quote("'"$MODEL_ID"'", safe=""))')" >/dev/null
  rm -f "$CARD_FILE"
  echo "    Done"
  echo ""
fi

# Step 1: Add the malicious model via API
echo "[1] Adding model via POST /models/add ..."
ADD_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://$HOST/models/add" \
  -H "Content-Type: application/json" \
  -d "{\"model_id\":\"$MODEL_ID\"}")
HTTP_CODE=$(echo "$ADD_RESPONSE" | tail -1)
BODY=$(echo "$ADD_RESPONSE" | sed '$d')
echo "    HTTP $HTTP_CODE"

if [ "$HTTP_CODE" -ge 400 ]; then
  echo "    Model add failed (HTTP $HTTP_CODE) — that's fine if model doesn't exist on HF."
  echo "    Response: $BODY"
  echo ""
  echo "RESULT: Model was rejected at add time. Attack blocked."
  exit 0
fi

# Step 2: Verify the saved TOML has trust_remote_code = false
echo ""
echo "[2] Checking saved model card TOML ..."
if [ ! -f "$CARD_FILE" ]; then
  echo "    FAIL: Card file not found at $CARD_FILE"
  exit 1
fi

if grep -q 'trust_remote_code = false' "$CARD_FILE"; then
  echo "    SAFE: trust_remote_code = false (fix is active)"
else
  echo "    VULNERABLE: trust_remote_code is not false — remote code WILL be trusted"
fi
echo "    Contents:"
cat "$CARD_FILE"

# Step 3: Place the instance
echo ""
echo "[3] Attempting POST /place_instance ..."
PLACE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://$HOST/place_instance" \
  -H "Content-Type: application/json" \
  -d "{\"model_id\":\"$MODEL_ID\"}")
PLACE_CODE=$(echo "$PLACE_RESPONSE" | tail -1)
PLACE_BODY=$(echo "$PLACE_RESPONSE" | sed '$d')
echo "    HTTP $PLACE_CODE"
echo "    Response: $PLACE_BODY"

# Step 3b: Send a chat completion to actually trigger tokenizer loading
echo ""
echo "[3b] Sending chat completion to trigger tokenizer load ..."
CHAT_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 30 -X POST "http://$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":1}")
CHAT_CODE=$(echo "$CHAT_RESPONSE" | tail -1)
CHAT_BODY=$(echo "$CHAT_RESPONSE" | sed '$d')
echo "    HTTP $CHAT_CODE"
echo "    Response: $CHAT_BODY"
echo ""
echo "[3c] Checking for RCE proof ..."
sleep 5
if [ -f /tmp/exo-rce-proof.txt ]; then
  echo "    VULNERABLE: Remote code executed!"
  echo "    Contents:"
  cat /tmp/exo-rce-proof.txt
else
  echo "    SAFE: /tmp/exo-rce-proof.txt does not exist — remote code was NOT executed"
fi

# Step 4: Clean up — delete instance and custom model
echo ""
echo "[4] Cleaning up ..."

# Find and delete any instance for this model
INSTANCE_ID=$(curl -s "http://$HOST/state" | python3 -c "
import sys, json
state = json.load(sys.stdin)
for iid, wrapper in state.get('instances', {}).items():
    for tag, inst in wrapper.items():
        sa = inst.get('shardAssignments', {})
        if sa.get('modelId', '') == '$MODEL_ID':
            print(iid)
            sys.exit(0)
" 2>/dev/null || true)

if [ -n "$INSTANCE_ID" ]; then
  echo "    Deleting instance $INSTANCE_ID ..."
  curl -s -X DELETE "http://$HOST/instance/$INSTANCE_ID" >/dev/null
  echo "    Done"
else
  echo "    No instance found to delete"
fi

echo "    Deleting custom model card ..."
curl -s -X DELETE \
  "http://$HOST/models/custom/$(python3 -c 'import urllib.parse; print(urllib.parse.quote("'"$MODEL_ID"'", safe=""))')" >/dev/null
echo "    Done"

echo ""
echo "=== DONE ==="
