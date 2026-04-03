#!/usr/bin/env python3
"""Simple quant comparison — sends prompts to whatever model is loaded.

Does NOT manage instances. Load the model yourself, then run this.

Usage:
  uv run python3 bench/quant_compare.py --base-url http://192.168.86.201:52415 --label nvfp4
  uv run python3 bench/quant_compare.py --base-url http://192.168.86.201:52415 --label 4bit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

# Qwen3.5-397B model-card params for thinking mode + webui defaults
COMMON_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": True,
    "logprobs": True,
    "top_logprobs": 5,
    "enable_thinking": True,
    "max_tokens": 131072,
    # no seed — let it be non-deterministic
}

PROMPTS: list[dict[str, str]] = [
    {
        "id": "algo_impl",
        "name": "Algorithm Implementation",
        "system": "You are an expert Python programmer. Write clean, well-typed, production-quality code.",
        "user": (
            "Implement a persistent (immutable) red-black tree in Python with type hints. "
            "Support insert, lookup, and in-order traversal. The tree should be fully "
            "immutable — insert returns a new tree. Include a brief complexity analysis "
            "as a docstring."
        ),
    },
    {
        "id": "debug_fix",
        "name": "Debug and Fix",
        "system": "You are a senior engineer debugging production code. Be precise and thorough.",
        "user": (
            "This async Python function has a subtle bug that causes data loss under concurrency. "
            "Find the bug, explain it, and provide the fix:\n\n"
            "```python\n"
            "import asyncio\n"
            "from dataclasses import dataclass, field\n\n"
            "@dataclass\n"
            "class BatchProcessor:\n"
            "    batch: list[str] = field(default_factory=list)\n"
            "    batch_size: int = 10\n"
            "    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)\n\n"
            "    async def add(self, item: str) -> None:\n"
            "        async with self._lock:\n"
            "            self.batch.append(item)\n"
            "            if len(self.batch) >= self.batch_size:\n"
            "                await self._flush()\n\n"
            "    async def _flush(self) -> None:\n"
            "        items = self.batch\n"
            "        self.batch = []\n"
            "        # Simulate sending batch to external service\n"
            "        await asyncio.sleep(0.1)\n"
            "        print(f'Flushed {len(items)} items')\n"
            "```"
        ),
    },
    {
        "id": "arch_design",
        "name": "Architecture Design",
        "system": "You are a systems architect. Provide concrete, actionable designs with code.",
        "user": (
            "Design a distributed rate limiter for a multi-region API gateway. Requirements:\n"
            "- Sliding window algorithm with 1-second granularity\n"
            "- Must handle 100K+ req/s per region with <1ms p99 latency overhead\n"
            "- Eventually consistent across regions (allow brief over-limit)\n"
            "- Graceful degradation if the backing store (Redis) is unavailable\n\n"
            "Provide the core Python implementation with Redis commands, the cross-region "
            "sync protocol, and the local fallback strategy. Explain tradeoffs."
        ),
    },
    {
        "id": "refactor",
        "name": "Complex Refactor",
        "system": "You are refactoring legacy code. Preserve behavior exactly while improving structure.",
        "user": (
            "Refactor this legacy callback-based Node.js code into modern async/await TypeScript "
            "with proper error handling and types. The refactored code must be a drop-in replacement.\n\n"
            "```javascript\n"
            "function fetchUserData(userId, callback) {\n"
            "  db.query('SELECT * FROM users WHERE id = ?', [userId], function(err, rows) {\n"
            "    if (err) return callback(err);\n"
            "    if (!rows.length) return callback(null, null);\n"
            "    var user = rows[0];\n"
            "    db.query('SELECT * FROM orders WHERE user_id = ?', [userId], function(err, orders) {\n"
            "      if (err) return callback(err);\n"
            "      user.orders = orders;\n"
            "      orders.forEach(function(order) {\n"
            "        db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id],\n"
            "          function(err, items) {\n"
            "            if (err) return callback(err);\n"
            "            order.items = items;\n"
            "          });\n"
            "      });\n"
            "      // BUG: callback fires before forEach queries complete\n"
            "      setTimeout(function() { callback(null, user); }, 100);\n"
            "    });\n"
            "  });\n"
            "}\n"
            "```"
        ),
    },
    {
        "id": "code_review",
        "name": "Code Review & Security Audit",
        "system": "You are a security-focused code reviewer. Find all issues, rank by severity.",
        "user": (
            "Review this FastAPI endpoint for bugs, security issues, and performance problems. "
            "Rank findings by severity (Critical/High/Medium/Low).\n\n"
            "```python\n"
            "from fastapi import FastAPI, Request\n"
            "from pydantic import BaseModel\n"
            "import sqlite3\n"
            "import os\n"
            "import pickle\n"
            "import hashlib\n\n"
            "app = FastAPI()\n"
            "DB_PATH = os.getenv('DB_PATH', 'app.db')\n\n"
            "class UserUpdate(BaseModel):\n"
            "    name: str\n"
            "    role: str\n"
            "    preferences: str  # base64-encoded pickled dict\n\n"
            "@app.put('/users/{user_id}')\n"
            "async def update_user(user_id: int, body: UserUpdate, request: Request):\n"
            "    token = request.headers.get('Authorization', '').removeprefix('Bearer ')\n"
            "    token_hash = hashlib.md5(token.encode()).hexdigest()\n"
            "    \n"
            "    conn = sqlite3.connect(DB_PATH)\n"
            "    cursor = conn.cursor()\n"
            "    \n"
            "    # Verify token\n"
            "    cursor.execute(f\"SELECT user_id FROM sessions WHERE token_hash = '{token_hash}'\")\n"
            "    session = cursor.fetchone()\n"
            "    if not session:\n"
            "        return {'error': 'unauthorized'}\n"
            "    \n"
            "    # Deserialize preferences\n"
            "    import base64\n"
            "    prefs = pickle.loads(base64.b64decode(body.preferences))\n"
            "    \n"
            "    # Update user\n"
            "    cursor.execute(\n"
            "        f\"UPDATE users SET name = '{body.name}', role = '{body.role}', \"\n"
            "        f\"preferences = '{json.dumps(prefs)}' WHERE id = {user_id}\"\n"
            "    )\n"
            "    conn.commit()\n"
            "    conn.close()\n"
            "    return {'status': 'updated'}\n"
            "```"
        ),
    },
]


@dataclass
class PromptResult:
    prompt_id: str
    prompt_name: str
    model: str
    thinking: str = ""
    response: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    decode_tok_per_s: float = 0.0
    error: str | None = None


async def detect_model(client: httpx.AsyncClient, base_url: str) -> str:
    """Get the model ID from current running instance."""
    resp = await client.get(f"{base_url}/state", timeout=10.0)
    state = resp.json()
    for _iid, inst in state.get("instances", {}).items():
        for _k, v in inst.items():
            if isinstance(v, dict):
                sa = v.get("shardAssignments") or v.get("shard_assignments") or {}
                mid = sa.get("modelId") or sa.get("model_id")
                if mid:
                    return mid
    raise RuntimeError("No model instance found. Load a model first.")


async def run_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: dict[str, str],
) -> PromptResult:
    result = PromptResult(
        prompt_id=prompt["id"],
        prompt_name=prompt["name"],
        model=model,
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        **COMMON_PARAMS,
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    chunks_text: list[str] = []
    chunks_thinking: list[str] = []
    usage: dict = {}
    token_count = 0

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

                # Thinking: exo uses "reasoning_content"
                rc = delta.get("reasoning_content")
                if rc:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks_thinking.append(rc)
                    token_count += 1

                # Response content
                c = delta.get("content")
                if c:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunks_text.append(c)
                    token_count += 1

                # Print progress dot every 500 tokens
                if token_count % 500 == 0 and token_count > 0:
                    elapsed = time.perf_counter() - start
                    print(f"    {token_count} tokens, {elapsed:.0f}s...", flush=True)

    except Exception as e:
        result.error = str(e)
        return result

    end = time.perf_counter()

    result.thinking = "".join(chunks_thinking)
    result.response = "".join(chunks_text)
    result.total_ms = (end - start) * 1000
    result.first_token_ms = (first_token_time - start) * 1000 if first_token_time else 0

    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)
    result.thinking_tokens = usage.get("thinking_tokens", usage.get("reasoning_tokens", 0))
    result.total_tokens = usage.get("total_tokens", 0)

    # If usage didn't report thinking_tokens, estimate from chunk count
    if result.thinking_tokens == 0 and chunks_thinking:
        result.thinking_tokens = len(chunks_thinking)

    decode_time_s = end - (first_token_time or start)
    if decode_time_s > 0 and result.completion_tokens > 0:
        result.decode_tok_per_s = result.completion_tokens / decode_time_s

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompts against loaded model")
    parser.add_argument("--base-url", default="http://192.168.86.201:52415")
    parser.add_argument("--label", required=True, help="Label for this run (e.g. 'nvfp4' or '4bit')")
    parser.add_argument("--output-dir", default="bench/quant_compare_results")
    parser.add_argument("--prompts", default=None, help="Comma-separated prompt IDs (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS
    if args.prompts:
        ids = set(args.prompts.split(","))
        prompts = [p for p in PROMPTS if p["id"] in ids]

    async with httpx.AsyncClient() as client:
        model = await detect_model(client, args.base_url)
        print(f"Detected model: {model}")
        print(f"Label: {args.label}")
        print(f"Prompts: {len(prompts)}")
        print(f"Params: temp={COMMON_PARAMS['temperature']}, top_p={COMMON_PARAMS['top_p']}, "
              f"thinking=on, max_tokens={COMMON_PARAMS['max_tokens']}")
        print()

        results: list[PromptResult] = []
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {prompt['name']}...", flush=True)
            result = await run_prompt(client, args.base_url, model, prompt)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                think_tok = result.thinking_tokens
                reply_tok = result.completion_tokens - think_tok if result.completion_tokens else len(result.response.split())
                print(f"  Thinking: {think_tok} tokens ({len(result.thinking)} chars)")
                print(f"  Response: {reply_tok} tokens ({len(result.response)} chars)")
                print(f"  Decode: {result.decode_tok_per_s:.1f} tok/s | TTFT: {result.first_token_ms:.0f}ms | Total: {result.total_ms/1000:.1f}s")

            results.append(result)
            print()

        # Save JSON
        json_path = output_dir / f"results_{args.label}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Saved JSON: {json_path}")

        # Save individual markdown files
        for r in results:
            if r.error:
                continue
            md_path = output_dir / f"{r.prompt_id}_{args.label}.md"
            with open(md_path, "w") as f:
                f.write(f"# {r.prompt_name} — {args.label}\n\n")
                f.write(f"**Model:** `{r.model}`\n\n")
                f.write(f"| Metric | Value |\n|---|---|\n")
                f.write(f"| Thinking tokens | {r.thinking_tokens} |\n")
                f.write(f"| Completion tokens | {r.completion_tokens} |\n")
                f.write(f"| Decode tok/s | {r.decode_tok_per_s:.1f} |\n")
                f.write(f"| TTFT | {r.first_token_ms:.0f}ms |\n")
                f.write(f"| Total time | {r.total_ms/1000:.1f}s |\n\n")
                if r.thinking:
                    f.write("## Thinking\n\n")
                    f.write("```\n")
                    f.write(r.thinking)
                    f.write("\n```\n\n")
                f.write("## Response\n\n")
                f.write(r.response)
                f.write("\n")
            print(f"Saved: {md_path}")

        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
