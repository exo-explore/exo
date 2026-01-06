#!/usr/bin/env bash

set -euo pipefail

query() {
  tailscale status | awk -v find="$1" '$2 == find { print $1 }'
}

if [[ $# -lt 2 ]]; then
  echo "USAGE: $0 <test kind> [host1] [host2] ..."
  exit 1
fi


kind=$1
shift

test_kinds="ring jaccl"

if ! echo "$test_kinds" | grep -q "$kind"; then
  printf "%s is not a known test kind.\nCurrent test kinds are %s" "$kind" "$test_kinds"
  exit 1
fi

hostnames=("$@")
weaved=()
ips=()
for name in "${hostnames[@]}"; do
  ip=$(query "$name")
  ips+=("$ip")
  weaved+=("$name" "$ip")
done

devs_raw=$(printf "[\"%s\", \"%s\"], " "${weaved[@]}")
devs="[${devs_raw%, }]"

for i in "${!ips[@]}"; do  
  { 
    req="{
      \"model_id\": \"llama-3.2-1b\",
      \"devs\": ${devs},
      \"kind\": \"inference\"
     }"
    echo "req $req"
    curl -sN \
      -X POST "http://${ips[$i]}:52415/${kind}" \
      -H "Content-Type: application/json" -d "$req" \
    2>&1 | sed "s/^/\n${hostnames[$i]}@${ips[$i]}: /" || echo "curl to ${hostnames[$i]} failed"
  } &
done

wait
