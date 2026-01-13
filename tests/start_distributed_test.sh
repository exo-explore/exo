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

model_ids=("qwen3-30b" "gpt-oss-120b-MXFP4-Q8" "kimi-k2-thinking")

for model_id in "${model_ids[@]}"; do
  for i in "${!ips[@]}"; do  
    { 
      req="{
        \"model_id\": \"${model_id}\",
        \"devs\": ${devs},
        \"kind\": \"inference\"
       }"
      echo "req $req"
      curl -sN \
        -X POST "http://${ips[$i]}:52415/${kind}" \
        -H "Content-Type: application/json" -d "$req" \
      2>&1 | sed "s/^/\n${hostnames[$i]}@${ips[$i]}: /" || echo "curl to ${hostnames[$i]} failed" && exit 1
    } &
  done
  wait
done

