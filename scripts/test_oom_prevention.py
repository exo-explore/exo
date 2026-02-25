#!/usr/bin/env python3
# Test OOM prevention by sending increasingly large contexts to exo.
#
# Usage:
#   1. Start exo with a model loaded (in another terminal):
#      uv run exo
#
#   2. Run this script:
#      uv run python scripts/test_oom_prevention.py
#
#   3. To force the OOM check to trigger sooner, lower the threshold:
#      EXO_MEMORY_THRESHOLD=0.5 uv run exo
#      Then run this script -- it should trigger much earlier.

import json
import sys
import time

import httpx

BASE_URL = "http://localhost:52415"
MODEL = "mlx-community/Llama-3.3-70B-Instruct-4bit"


def send_chat(
    messages: list[dict[str, str]], max_tokens: int = 100, stream: bool = True
) -> str | None:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        return _send_streaming(payload)
    return _send_non_streaming(payload)


def _send_streaming(payload: dict[str, object]) -> str | None:
    collected = ""
    error = None

    with httpx.stream(
        "POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if "error" in chunk:
                    err = chunk["error"]
                    error = (
                        err.get("message", str(err))
                        if isinstance(err, dict)
                        else str(err)
                    )
                    break
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                collected += delta.get("content", "")
            except json.JSONDecodeError:
                pass

    if error:
        return f"ERROR: {error}"
    return collected


def _send_non_streaming(payload: dict[str, object]) -> str | None:
    resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    if resp.status_code != 200:
        return f"ERROR (HTTP {resp.status_code}): {resp.text}"
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def make_large_message(token_count_approx: int) -> str:
    return "buffalo " * token_count_approx


def test_oom_prevention() -> None:
    print("=" * 60)
    print("OOM Prevention Test")
    print("=" * 60)

    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        models = resp.json()
        print(
            f"\nConnected to exo. Models: {[m['id'] for m in models.get('data', [])]}"
        )
    except Exception as e:
        print(f"\nCannot connect to exo at {BASE_URL}: {e}")
        print("Start exo first: uv run exo")
        sys.exit(1)

    print("\n--- Test 1: Small request (should succeed) ---")
    result = send_chat(
        [{"role": "user", "content": "Say hello in exactly 5 words."}],
        max_tokens=50,
    )
    print(f"Response: {result}")
    if result and result.startswith("ERROR"):
        print("FAIL: Small request should not fail")
        sys.exit(1)
    print("PASS")

    sizes = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 120000]
    max_gen = 8000

    print("\n--- Test 2: Escalating context sizes ---")
    print(f"{'Size':>10} | {'Max Gen':>8} | {'Result':>10} | Details")
    print("-" * 70)

    for size in sizes:
        large_msg = make_large_message(size)
        messages = [
            {
                "role": "user",
                "content": f"Summarize this in one sentence:\n{large_msg}",
            },
        ]

        t0 = time.time()
        try:
            result = send_chat(messages, max_tokens=max_gen)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"{size:>10} | {max_gen:>8} | {'EXCEPTION':>10} | {e}")
            continue

        if result is None:
            print(f"{size:>10} | {max_gen:>8} | {'NONE':>10} | No response")
        elif "not enough memory" in result.lower() or "ERROR" in result:
            print(f"{size:>10} | {max_gen:>8} | {'OOM CAUGHT':>10} | {result[:80]}...")
            print(
                f"\nOOM prevention triggered at ~{size} prompt tokens "
                f"+ {max_gen} gen tokens"
            )
            print("SUCCESS: The system caught the OOM before crashing!")
            return
        else:
            preview = result[:60].replace("\n", " ")
            print(
                f"{size:>10} | {max_gen:>8} | {'OK':>10} | "
                f"{preview}... ({elapsed:.1f}s)"
            )

    print("\nNOTE: OOM prevention did not trigger at any tested size.")
    print("Either your machine has enough memory, or try:")
    print("  EXO_MEMORY_THRESHOLD=0.5 uv run exo")


def test_with_low_threshold() -> None:
    # Requires: EXO_MEMORY_THRESHOLD=0.3 uv run exo
    print("\n--- Test 3: Forced OOM (requires EXO_MEMORY_THRESHOLD=0.3) ---")
    result = send_chat(
        [
            {
                "role": "user",
                "content": "Write a long essay about the history of computing.",
            }
        ],
        max_tokens=16000,
    )
    if result and ("not enough memory" in result.lower() or "ERROR" in result):
        print(f"PASS: OOM prevention triggered: {result[:100]}...")
    else:
        preview = (result or "")[:100]
        print(
            f"Did not trigger. Is EXO_MEMORY_THRESHOLD set low enough? Got: {preview}"
        )


if __name__ == "__main__":
    test_oom_prevention()
    if "--force" in sys.argv:
        test_with_low_threshold()
