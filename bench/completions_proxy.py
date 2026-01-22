# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Proxy that converts /v1/completions requests to /v1/chat/completions.

Used by exo_eval to support lm_eval tasks that require the completions API.
"""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Tasks that require the completions API (loglikelihood-based)
# These cannot work with chat completions because they need prompt token logprobs
COMPLETIONS_REQUIRED_TASKS: set[str] = {
    # Multiple choice / loglikelihood tasks
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "sciq",
    "siqa",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "winogrande",
    "boolq",
    "lambada",
    "lambada_openai",
    "logiqa",
    "logiqa2",
    # Add more as needed
}

# Task prefixes that indicate completions are required
COMPLETIONS_REQUIRED_PREFIXES: tuple[str, ...] = (
    "mmlu_",  # mmlu subtasks
    "arc_",  # arc subtasks
    "hellaswag_",
    "winogrande_",
)


def tasks_require_completions(tasks: list[str]) -> bool:
    """Check if any of the tasks require the completions API."""
    for task in tasks:
        task_lower = task.lower()
        if task_lower in COMPLETIONS_REQUIRED_TASKS:
            return True
        for prefix in COMPLETIONS_REQUIRED_PREFIXES:
            if task_lower.startswith(prefix):
                return True
    return False


def find_free_port() -> int:
    """Find a free port to use for the proxy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_proxy_app(upstream_url: str) -> FastAPI:
    """Create a FastAPI app that proxies completions to chat completions."""

    app = FastAPI()

    def convert_completions_to_chat_request(
        completions_req: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert a /v1/completions request to /v1/chat/completions format."""
        prompt = completions_req.get("prompt", "")

        # Handle prompt as string or list of strings
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        chat_req: dict[str, Any] = {
            "model": completions_req.get("model", ""),
            "messages": [{"role": "user", "content": prompt}],
            "stream": completions_req.get("stream", False),
        }

        # Map common parameters
        for param in (
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            if param in completions_req:
                chat_req[param] = completions_req[param]

        # Handle logprobs - completions uses int, chat uses bool + top_logprobs
        logprobs = completions_req.get("logprobs")
        if logprobs is not None and logprobs > 0:
            chat_req["logprobs"] = True
            chat_req["top_logprobs"] = logprobs
        elif logprobs is not None:
            chat_req["logprobs"] = True

        return chat_req

    def convert_chat_to_completions_response(
        chat_resp: dict[str, Any],
        echo: bool = False,
        prompt: str = "",
    ) -> dict[str, Any]:
        """Convert a /v1/chat/completions response to /v1/completions format."""
        choices = []

        for chat_choice in chat_resp.get("choices", []):
            message = chat_choice.get("message", {})
            text = message.get("content", "") or ""

            # Build logprobs in completions format
            logprobs_data = None
            chat_logprobs = chat_choice.get("logprobs")

            if chat_logprobs and chat_logprobs.get("content"):
                tokens: list[str] = []
                token_logprobs: list[float] = []
                top_logprobs: list[dict[str, float]] = []
                text_offset: list[int] = []

                offset = 0
                for item in chat_logprobs["content"]:
                    tokens.append(item["token"])
                    token_logprobs.append(item["logprob"])

                    # Convert top_logprobs list to dict format
                    top_lp_dict: dict[str, float] = {}
                    for top_item in item.get("top_logprobs", []):
                        top_lp_dict[top_item["token"]] = top_item["logprob"]
                    top_logprobs.append(top_lp_dict)

                    text_offset.append(offset)
                    offset += len(item["token"])

                logprobs_data = {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                    "text_offset": text_offset,
                }

            # If echo was requested, prepend prompt to text
            if echo:
                text = prompt + text

            choices.append(
                {
                    "text": text,
                    "index": chat_choice.get("index", 0),
                    "logprobs": logprobs_data,
                    "finish_reason": chat_choice.get("finish_reason"),
                }
            )

        return {
            "id": chat_resp.get("id", ""),
            "object": "text_completion",
            "created": chat_resp.get("created", 0),
            "model": chat_resp.get("model", ""),
            "choices": choices,
            "usage": chat_resp.get("usage"),
        }

    def convert_chat_stream_chunk_to_completions(
        chunk: dict[str, Any],
        echo: bool = False,
        prompt: str = "",
        is_first: bool = False,
    ) -> dict[str, Any]:
        """Convert a streaming chat completion chunk to completions format."""
        choices = []

        for chat_choice in chunk.get("choices", []):
            delta = chat_choice.get("delta", {})
            text = delta.get("content", "") or ""

            # If echo and first chunk, prepend prompt
            if echo and is_first:
                text = prompt + text

            # Build logprobs in completions format
            logprobs_data = None
            chat_logprobs = chat_choice.get("logprobs")

            if chat_logprobs and chat_logprobs.get("content"):
                tokens: list[str] = []
                token_logprobs: list[float] = []
                top_logprobs: list[dict[str, float]] = []

                for item in chat_logprobs["content"]:
                    tokens.append(item["token"])
                    token_logprobs.append(item["logprob"])

                    top_lp_dict: dict[str, float] = {}
                    for top_item in item.get("top_logprobs", []):
                        top_lp_dict[top_item["token"]] = top_item["logprob"]
                    top_logprobs.append(top_lp_dict)

                logprobs_data = {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                }

            choices.append(
                {
                    "text": text,
                    "index": chat_choice.get("index", 0),
                    "logprobs": logprobs_data,
                    "finish_reason": chat_choice.get("finish_reason"),
                }
            )

        return {
            "id": chunk.get("id", ""),
            "object": "text_completion",
            "created": chunk.get("created", 0),
            "model": chunk.get("model", ""),
            "choices": choices,
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(request: Request):
        body = await request.json()

        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        echo = body.get("echo", False)
        stream = body.get("stream", False)

        chat_request = convert_completions_to_chat_request(body)
        logger.debug(f"Proxying to {upstream_url}/v1/chat/completions")

        async with httpx.AsyncClient(timeout=300.0, http2=False) as client:
            if stream:

                async def generate() -> AsyncGenerator[str, None]:
                    is_first = True
                    async with client.stream(
                        "POST",
                        f"{upstream_url}/v1/chat/completions",
                        json=chat_request,
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break

                                try:
                                    chunk = json.loads(data)
                                    converted = (
                                        convert_chat_stream_chunk_to_completions(
                                            chunk,
                                            echo=echo,
                                            prompt=prompt,
                                            is_first=is_first,
                                        )
                                    )
                                    is_first = False
                                    yield f"data: {json.dumps(converted)}\n\n"
                                except json.JSONDecodeError:
                                    continue

                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                response = await client.post(
                    f"{upstream_url}/v1/chat/completions",
                    json=chat_request,
                )
                chat_response = response.json()

                if "error" in chat_response:
                    return JSONResponse(chat_response, status_code=response.status_code)

                completions_response = convert_chat_to_completions_response(
                    chat_response, echo=echo, prompt=prompt
                )
                return JSONResponse(completions_response)

    @app.get("/v1/models", response_model=None)
    async def models():
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{upstream_url}/v1/models")
            return JSONResponse(response.json())

    return app


class CompletionsProxy:
    """Manages a completions proxy server lifecycle."""

    def __init__(self, upstream_host: str, upstream_port: int):
        self.upstream_url = f"http://{upstream_host}:{upstream_port}"
        self.port = find_free_port()
        self.host = "127.0.0.1"
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        """Start the proxy server in the background."""
        app = create_proxy_app(self.upstream_url)
        config = Config()
        config.bind = [f"{self.host}:{self.port}"]
        config.accesslog = None  # Suppress access logs

        self._shutdown_event = asyncio.Event()

        async def run_server() -> None:
            await serve(app, config, shutdown_trigger=self._shutdown_event.wait)  # type: ignore[arg-type]

        self._task = asyncio.create_task(run_server())

        # Wait a bit for server to start
        await asyncio.sleep(0.5)
        logger.info(f"Completions proxy started on {self.base_url}")

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info("Completions proxy stopped")


@asynccontextmanager
async def completions_proxy_context(
    upstream_host: str, upstream_port: int
) -> AsyncIterator[CompletionsProxy]:
    """Context manager for running the completions proxy."""
    proxy = CompletionsProxy(upstream_host, upstream_port)
    await proxy.start()
    try:
        yield proxy
    finally:
        await proxy.stop()


@contextmanager
def run_completions_proxy(
    upstream_host: str, upstream_port: int
) -> Generator[CompletionsProxy, None, None]:
    """Synchronous context manager that runs proxy in a subprocess."""
    import subprocess
    import sys
    import time

    port = find_free_port()
    upstream_url = f"http://{upstream_host}:{upstream_port}"

    # Start proxy as subprocess
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import asyncio
import sys
from bench.completions_proxy import create_proxy_app
from hypercorn.asyncio import serve
from hypercorn.config import Config

async def main():
    print(f"Proxy starting: 127.0.0.1:{port} -> {upstream_url}", file=sys.stderr, flush=True)
    app = create_proxy_app("{upstream_url}")
    config = Config()
    config.bind = ["127.0.0.1:{port}"]
    config.accesslog = "-"  # Log to stderr
    config.errorlog = "-"
    await serve(app, config)

asyncio.run(main())
""",
        ],
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )

    # Create a proxy object with the right base_url
    class ProxyInfo:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

        @property
        def base_url(self) -> str:
            return f"http://{self.host}:{self.port}"

    proxy = ProxyInfo("127.0.0.1", port)

    # Wait for server to start
    time.sleep(1.0)
    logger.info(f"Completions proxy started on {proxy.base_url} -> {upstream_url}")

    try:
        yield proxy  # type: ignore[misc]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Completions proxy stopped")
