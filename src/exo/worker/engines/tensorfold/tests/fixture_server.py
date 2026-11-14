#!/usr/bin/env python3
"""Protocol fixture only: this executable never loads weights or runs GPU inference."""

import argparse
import json
import os
import socketserver
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast, override

# macOS nodes publish Metal memory, not CUDA memory, so the fixture follows the host.
NATIVE_BACKEND = "metal" if sys.platform == "darwin" else "cuda"
_record_lock = threading.Lock()
_model = ""
_requests = 0
_active = 0


def frames(request: dict[str, Any], model: str) -> Iterator[bytes]:
    messages = cast(list[dict[str, object]], request.get("messages", []))
    prompt = " ".join(str(message.get("content", "")) for message in messages)
    count = min(
        cast(int, request.get("max_tokens", 8)), 8 if "fixture:slow" in prompt else 3
    )
    identifiers = list(range(301, 301 + count))
    request_id = "chatcmpl-protocol-fixture"
    pieces = ["Answer ", "café ", "✓", " four", " five", " six", " seven", " eight"]
    finish = "length" if count < 3 else "stop"

    for index, token in enumerate(identifiers):
        if index and "fixture:slow" in prompt:
            time.sleep(0.3)
        raw = {
            "object": "tensorfold.token.chunk",
            "version": 1,
            "id": request_id,
            "model": model,
            "offset": index + (1 if "fixture:bad-offset" in prompt else 0),
            "token_ids": [token],
        }
        yield event(raw)
        delta: dict[str, Any] = {"content": pieces[index]}
        if "fixture:reasoning" in prompt and index == 0:
            delta = {"reasoning_content": "Checking the result."}
        if "fixture:tool" in prompt:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_fixture" if index == 0 else None,
                        "type": "function" if index == 0 else None,
                        "function": {
                            "name": "weather" if index == 0 else None,
                            "arguments": ['{"city":', '"London"', "}"][index],
                        },
                    }
                ]
            }
            finish = "tool_calls"
        yield event(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                "tensorfold_stream": {
                    "version": 1,
                    "token_count": index + 1,
                    "token_id": token,
                },
            }
        )
        if "fixture:truncate" in prompt:
            return
        if "fixture:malformed" in prompt:
            yield b"data: {broken json\n\n"
            return
    yield event(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "tensorfold_stream": {
                "version": 1,
                "token_count": count,
                "token_id": identifiers[-1] if identifiers else None,
            },
        }
    )
    yield event(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": count
                + (1 if "fixture:bad-usage" in prompt else 0),
                "total_tokens": 11 + count,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {
                    "reasoning_tokens": 1 if "fixture:reasoning" in prompt else 0
                },
            },
            "tensorfold": {
                "token_ids": identifiers,
                "tokens_per_second": 30.0,
                "prefill_seconds": 0.01,
            },
            "tensorfold_stream": {
                "version": 1,
                "token_count": count,
                "token_id": identifiers[-1] if identifiers else None,
            },
        }
    )
    yield b"data: [DONE]\n\n"


def event(value: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def record(event: str, **values: object) -> None:
    if destination := os.environ.get("EXO_TENSORFOLD_FIXTURE_RECORD"):
        with _record_lock, Path(destination).open("a") as output:
            output.write(
                json.dumps({"event": event, "pid": os.getpid(), **values}) + "\n"
            )


def capabilities() -> dict[str, object]:
    return {
        "schema": 1,
        "engine": "zig",
        "version": "1.2.7-protocol-fixture",
        "chip": ("apple" if NATIVE_BACKEND == "metal" else "nvidia")
        + "-protocol-fixture",
        "backends": [NATIVE_BACKEND],
        "families": {"qwen3": ["mlx-q4g64"]},
        "token_stream_versions": [1],
        "serve": {
            "flags": {
                flag: {}
                for flag in ("--unix-socket", "--name", "--parallel", "--context")
            }
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    def respond(self, value: dict[str, object], status: int = 200) -> None:
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.respond(
                {
                    "status": "ok",
                    "model": _model,
                    "warming": False,
                    "max_batch_size": 2,
                    "backend": NATIVE_BACKEND,
                    "version": "1.2.7-protocol-fixture",
                    "memory": {"peak": 1234},
                    "live": {"connections": _active, "waiting": 0},
                }
            )
        elif self.path == "/v1/models":
            self.respond(
                {
                    "data": [
                        {
                            "id": _model,
                            "capabilities": {
                                name: True
                                for name in (
                                    "text_generation",
                                    "tools",
                                    "reasoning",
                                    "logprobs",
                                    "penalties",
                                    "thinking_switch",
                                    "thinking_applicable",
                                )
                            }
                            | {"greedy_only": False},
                            "context_length": 4096,
                        }
                    ]
                }
            )
        else:
            self.respond({"error": {"message": "Unknown fixture endpoint"}}, 404)

    def do_POST(self) -> None:
        global _requests, _active
        if self.path != "/v1/chat/completions":
            self.respond({"error": {"message": "Unknown fixture endpoint"}}, 404)
            return
        request = cast(
            dict[str, Any],
            json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
        )
        messages = cast(list[dict[str, object]], request.get("messages", []))
        prompt = " ".join(str(message.get("content", "")) for message in messages)
        record("request", request=request)
        if "fixture:http-error" in prompt:
            self.respond({"error": {"message": "Injected native rejection"}}, 400)
            return
        if "fixture:crash" in prompt:
            record("crash")
            os._exit(23)
        _requests += 1
        _active += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for frame in frames(request, _model):
                for offset in range(0, len(frame), 17):
                    self.wfile.write(frame[offset : offset + 17])
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            record("cancelled", prompt=prompt)
        else:
            record("finished", prompt=prompt)
        finally:
            _active -= 1
            self.close_connection = True


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False


def main() -> None:
    global _model
    if sys.argv[1:] == ["capabilities", "--json"]:
        print(json.dumps(capabilities()))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("model_path")
    parser.add_argument("--name", required=True)
    parser.add_argument("--unix-socket", required=True)
    parser.add_argument("--parallel", default="2")
    parser.add_argument("--context", default="4096")
    arguments = parser.parse_args()
    _model = cast(str, arguments.name)
    socket_path = Path(cast(str, arguments.unix_socket))
    record(
        "start",
        socket=str(socket_path),
        model=_model,
        model_path=cast(str, arguments.model_path),
        context=int(cast(str, arguments.context)),
        parallel=int(cast(str, arguments.parallel)),
    )
    try:
        with Server(str(socket_path), Handler) as server:
            server.serve_forever(poll_interval=0.05)
    finally:
        socket_path.unlink(missing_ok=True)
        record("exit")


if __name__ == "__main__":
    main()
