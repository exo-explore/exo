#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-gx10-de89}"
PORT="${2:-52415}"
NUM_REQUESTS="${3:-4}"
MODEL="${4:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "Sending $NUM_REQUESTS parallel requests to $HOST:$PORT ($MODEL) with ~32k token prompts..."
echo

tmpdir=$(mktemp -d)
pids=()
for i in $(seq 1 "$NUM_REQUESTS"); do
  (
    python3 -c "
import json, sys, time, urllib.request

import random
random.seed($i * 9999)
topics = [
    'mathematics', 'philosophy', 'religion', 'culture', 'astronomy',
    'biology', 'music', 'architecture', 'literature', 'physics',
    'chemistry', 'geology', 'psychology', 'economics', 'linguistics',
]
random.shuffle(topics)
sentences = []
for j in range(95):
    t1, t2, t3 = topics[j % len(topics)], topics[(j+3) % len(topics)], topics[(j+7) % len(topics)]
    sentences.append(
        f'In the field of {t1}, the number {$i * 1000 + j} holds particular significance '
        f'when examining its relationship to {t2} and {t3}. Scholars have long debated '
        f'whether the patterns observed in iteration {j} of this analysis reveal deeper '
        f'structural connections between seemingly unrelated disciplines. The evidence '
        f'from experiment {$i * 7 + j * 13} suggests that cross-domain numerical '
        f'correlations emerge at scale {j * $i}, challenging conventional assumptions '
        f'about the independence of these fields. ')
prompt = ' '.join(sentences) + f' Summarize the key finding about the number {$i}.'
payload = json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 1,
    'stream': True,
}).encode()
req = urllib.request.Request(
    'http://$HOST:$PORT/v1/chat/completions',
    data=payload,
    headers={'Content-Type': 'application/json'},
)
t0 = time.perf_counter()
try:
    resp = urllib.request.urlopen(req, timeout=300)
    first_byte = None
    for line in resp:
        if first_byte is None:
            first_byte = time.perf_counter()
        line = line.decode().strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            break
    ttft = (first_byte or time.perf_counter()) - t0
    prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
    tps = prompt_tokens / ttft
    print(f'request $i: TTFT={ttft:.2f}s  ~{int(prompt_tokens)} prompt tokens  ~{int(tps)} tok/s prefill')
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f'request $i: FAILED after {elapsed:.2f}s — {e}', file=sys.stderr)
    sys.exit(1)
" >"$tmpdir/$i" 2>&1
  ) &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

for i in $(seq 1 "$NUM_REQUESTS"); do
  cat "$tmpdir/$i"
done
rm -rf "$tmpdir"
