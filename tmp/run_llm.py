#!/usr/bin/env python3
import argparse
import json
import sys

import requests


def stream_chat(host: str, query: str) -> None:
    url = f"http://{host}:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        # "model": "mlx-community/Llama-3_3-Nemotron-Super-49B-v1_5-mlx-4Bit",
        "stream": True,
        "messages": [{"role": "user", "content": query}],
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # SSE lines look like: "data: {...}" or "data: [DONE]"
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        print(content, end="", flush=True)

    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream chat completions from a local server."
    )
    parser.add_argument("host", help="Hostname (without protocol), e.g. localhost")
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a text file whose contents will be used as the query",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text (if not using -f/--file). All remaining arguments are joined with spaces.",
    )

    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                query = f.read().strip()
        except OSError as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.query:
        query = " ".join(args.query)
    else:
        parser.error("You must provide either a query or a file (-f/--file).")

    stream_chat(args.host, query)


if __name__ == "__main__":
    main()
