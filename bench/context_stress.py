#!/usr/bin/env python3
"""Context length stress test with quantized KV cache.

Sends progressively larger prompts to test the context ceiling.
Uses needle-in-a-haystack to verify coherence.

Usage:
  uv run python3 bench/context_stress.py --base-url http://192.168.86.201:52415
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx

# Filler paragraphs about various programming topics (to pad context)
FILLER_TOPICS = [
    "The observer pattern is a software design pattern in which an object, named the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes, usually by calling one of their methods. It is mainly used for implementing distributed event handling systems. The observer pattern is also a key part in the familiar model-view-controller architectural pattern.",
    "A binary search tree is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree. The time complexity of operations on the binary search tree is directly proportional to the height of the tree.",
    "Garbage collection is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program but is no longer referenced. Garbage collection was invented by American computer scientist John McCarthy around 1959 to simplify manual memory management in Lisp.",
    "MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster. A MapReduce program is composed of a map procedure that performs filtering and sorting and a reduce method that performs a summary operation.",
    "The CAP theorem states that any distributed data store can provide only two of the following three guarantees: consistency, availability, and partition tolerance. When a network partition failure happens, it must be decided whether to cancel the operation and thus decrease the availability but ensure consistency or to proceed with the operation and thus provide availability but risk inconsistency.",
    "Functional programming is a programming paradigm where programs are constructed by applying and composing functions. It is a declarative programming paradigm in which function definitions are trees of expressions that map values to other values, rather than a sequence of imperative statements.",
    "A hash table is a data structure that implements an associative array or dictionary. It is an abstract data type that maps keys to values. A hash table uses a hash function to compute an index, also called a hash code, into an array of buckets or slots, from which the desired value can be found.",
    "Consensus algorithms are fundamental to distributed computing. They allow multiple processes to agree on a single value even in the presence of failures. Paxos and Raft are two well-known consensus algorithms used in practice for building reliable distributed systems.",
    "The actor model is a mathematical model of concurrent computation that treats actor as the universal primitive of concurrent computation. In response to a message it receives, an actor can make local decisions, create more actors, send more messages, and determine how to respond to the next message received.",
    "B-trees are self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in logarithmic time. The B-tree generalizes the binary search tree, allowing for nodes with more than two children. B-trees are well suited for storage systems that read and write relatively large blocks of data, such as databases and file systems.",
    "Type theory is the academic study of type systems. A type system is a syntactic method for enforcing levels of abstraction in programs. Type theory was created to avoid paradoxes in a variety of formal logics and rewrite systems. Two well-known type theories that can serve as mathematical foundations are Alonzo Church's typed lambda calculus and Per Martin-Lof's intuitionistic type theory.",
    "Event sourcing is a software architecture pattern in which changes to application state are stored as a sequence of events. Instead of storing just the current state, the full history of actions is stored. The current state can be reconstructed by replaying the events from the beginning of time.",
]

NEEDLE = "The secret code for project Nightingale is: FALCON-MERCURY-7749."


def build_prompt(target_tokens: int) -> tuple[str, str]:
    """Build a prompt of approximately target_tokens size.
    Returns (prompt_text, expected_answer).
    """
    # Rough estimate: 1 token ≈ 4 chars
    target_chars = target_tokens * 4

    # Place the needle randomly in the middle third
    paragraphs: list[str] = []
    char_count = 0

    needle_placed = False
    needle_position = random.randint(target_chars // 3, 2 * target_chars // 3)

    while char_count < target_chars:
        if not needle_placed and char_count >= needle_position:
            paragraphs.append(NEEDLE)
            char_count += len(NEEDLE)
            needle_placed = True
        else:
            topic = random.choice(FILLER_TOPICS)
            paragraphs.append(topic)
            char_count += len(topic) + 2  # +2 for newlines

    if not needle_placed:
        mid = len(paragraphs) // 2
        paragraphs.insert(mid, NEEDLE)

    filler = "\n\n".join(paragraphs)

    prompt = (
        "I'm going to give you a very long document. Read it carefully. "
        "At the end, I'll ask you a question about a specific detail buried in the text.\n\n"
        "--- BEGIN DOCUMENT ---\n\n"
        f"{filler}\n\n"
        "--- END DOCUMENT ---\n\n"
        "Question: What is the secret code for project Nightingale? "
        "Answer with just the code, nothing else."
    )

    return prompt, "FALCON-MERCURY-7749"


async def stress_test(base_url: str, target_tokens: int) -> dict:
    """Run a single stress test at target_tokens context size."""
    print(f"\n{'='*60}")
    print(f"Testing {target_tokens:,} token context")
    print(f"{'='*60}")

    prompt, expected = build_prompt(target_tokens)
    prompt_chars = len(prompt)
    est_tokens = prompt_chars // 4
    print(f"Prompt: {prompt_chars:,} chars (~{est_tokens:,} tokens)")

    body = {
        "model": "mlx-community/Qwen3.5-397B-A17B-4bit",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "stream": True,
        "enable_thinking": False,
        "max_tokens": 128,
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    response_chunks: list[str] = []
    usage: dict = {}

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=body,
                timeout=600.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        # Print prefill progress
                        if "prefill_progress" in line:
                            try:
                                prog = json.loads(line.split(" ", 1)[1])
                                chunk = prog.get("PrefillProgressChunk", {})
                                processed = chunk.get("processed_tokens", 0)
                                total = chunk.get("total_tokens", 0)
                                if total > 0:
                                    elapsed = time.perf_counter() - start
                                    print(f"  Prefill: {processed:,}/{total:,} tokens ({elapsed:.1f}s)", end="\r", flush=True)
                            except (json.JSONDecodeError, IndexError):
                                pass
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        response_chunks.append(content)

        except Exception as e:
            print(f"\n  ERROR: {e}")
            return {
                "target_tokens": target_tokens,
                "error": str(e),
            }

    end = time.perf_counter()
    response = "".join(response_chunks)
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    total_s = end - start
    prompt_tokens = usage.get("prompt_tokens", est_tokens)

    found_needle = expected.lower() in response.lower()

    print(f"\n  Prompt tokens: {prompt_tokens:,}")
    print(f"  TTFT (prefill): {ttft/1000:.1f}s")
    print(f"  Response: {response[:200]!r}")
    print(f"  Needle found: {'YES' if found_needle else 'NO'}")
    print(f"  Total: {total_s:.1f}s")

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "ttft_ms": ttft,
        "response": response,
        "needle_found": found_needle,
        "total_s": total_s,
    }


async def check_memory(host: str) -> str:
    """Get memory usage from a Studio via SSH."""
    import subprocess
    result = subprocess.run(
        ["ssh", host, "memory_pressure | head -1; sysctl hw.memsize | head -1"],
        capture_output=True, text=True, timeout=5,
    )
    return result.stdout.strip()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.86.201:52415")
    parser.add_argument("--targets", default="50000,80000,100000,120000,150000",
                        help="Comma-separated token targets")
    args = parser.parse_args()

    targets = [int(t) for t in args.targets.split(",")]

    print("Context Length Stress Test (4-bit quantized KV cache)")
    print(f"Targets: {[f'{t:,}' for t in targets]}")

    results = []
    for target in targets:
        result = await stress_test(args.base_url, target)
        results.append(result)

        if result.get("error"):
            print(f"\n  FAILED at {target:,} tokens — stopping.")
            break

        # Brief pause between tests
        await asyncio.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tokens':>10} {'Prefill':>10} {'Needle':>8} {'Status':>10}")
    print("-" * 45)
    for r in results:
        if r.get("error"):
            print(f"{r['target_tokens']:>10,} {'':>10} {'':>8} {'FAILED':>10}")
        else:
            ttft_s = r['ttft_ms'] / 1000
            needle = "YES" if r['needle_found'] else "NO"
            print(f"{r['prompt_tokens']:>10,} {ttft_s:>9.1f}s {needle:>8} {'OK':>10}")


if __name__ == "__main__":
    asyncio.run(main())
