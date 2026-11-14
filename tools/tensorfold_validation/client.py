import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from http.client import HTTPResponse
from typing import cast

Object = dict[str, object]


def object_value(value: object) -> Object:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    candidate = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in candidate):
        raise ValueError("Expected JSON string keys")
    return cast(Object, candidate)


def decode_object(encoded: str | bytes) -> Object:
    return object_value(cast(object, json.loads(encoded)))


def array_value(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return cast(list[object], value)


def token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Missing or invalid authoritative token usage")
    return value


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str | None
    timeout: float

    def open(self, path: str, payload: Object | None = None) -> HTTPResponse:
        headers = {
            "Accept": "application/json" if payload is None else "text/event-stream"
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path, data=data, headers=headers
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            return cast(HTTPResponse, opener.open(request, timeout=self.timeout))
        except urllib.error.HTTPError as error:
            raise ValueError(
                f"HTTP {error.code} from {path}: {error.read(2048).decode(errors='replace')}"
            ) from error

    def get(self, path: str) -> Object:
        with self.open(path) as response:
            return decode_object(response.read(8 * 1024 * 1024))


def stream_events(response: HTTPResponse, deadline: float) -> Iterator[Object | str]:
    pending: list[str] = []
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("The completion exceeded the total request deadline")
        line = response.readline(1024 * 1024 + 1)
        if len(line) > 1024 * 1024:
            raise ValueError("SSE line exceeds the validation limit")
        if not line:
            if pending:
                raise ValueError("Truncated SSE event")
            return
        decoded = line.decode().rstrip("\r\n")
        if decoded.startswith("data:"):
            pending.append(decoded[5:].lstrip(" "))
        elif not decoded and pending:
            value = "\n".join(pending)
            pending.clear()
            yield value if value == "[DONE]" else decode_object(value)


def output_text(event: Object) -> str:
    parts: list[str] = []
    for item in array_value(event.get("choices", [])):
        delta = object_value(object_value(item).get("delta", {}))
        for name in ("content", "reasoning_content"):
            value = delta.get(name)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def request_body(model: str, prompt: str, max_tokens: int) -> Object:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
    }


@dataclass(frozen=True)
class Trial:
    first_output_milliseconds: float
    elapsed_milliseconds: float
    completion_tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    output_sha256: str
    output_chunks: int

    def json(self) -> Object:
        return cast(Object, asdict(self))


def measure(endpoint: Endpoint, model: str, prompt: str, max_tokens: int) -> Trial:
    started = time.monotonic()
    first_output: float | None = None
    text: list[str] = []
    usage: Object | None = None
    finishes: list[str] = []
    done = False
    with endpoint.open(
        "/v1/chat/completions", request_body(model, prompt, max_tokens)
    ) as response:
        if "text/event-stream" not in (response.getheader("Content-Type") or ""):
            raise ValueError("Expected an SSE response")
        for event in stream_events(response, started + endpoint.timeout):
            if event == "[DONE]":
                done = True
                break
            if not isinstance(event, dict):
                raise ValueError("Unexpected stream payload")
            if "error" in event:
                raise ValueError(f"The completion failed: {event['error']}")
            output = output_text(event)
            if output:
                first_output = first_output or time.monotonic()
                text.append(output)
            if event.get("usage") is not None:
                usage = object_value(event["usage"])
            for item in array_value(event.get("choices", [])):
                finish = object_value(item).get("finish_reason")
                if isinstance(finish, str):
                    finishes.append(finish)
    elapsed = time.monotonic() - started
    if not done or first_output is None or usage is None or len(finishes) != 1:
        raise ValueError(
            "Incomplete completion: need output, authoritative usage, one finish, and [DONE]"
        )
    prompt_tokens = token_count(usage.get("prompt_tokens"))
    completion_tokens = token_count(usage.get("completion_tokens"))
    if token_count(usage.get("total_tokens")) != prompt_tokens + completion_tokens:
        raise ValueError("Token usage does not add up")
    if completion_tokens == 0 or completion_tokens > max_tokens:
        raise ValueError("Completion token count is outside the requested bound")
    if finishes[0] not in {"stop", "length"}:
        raise ValueError(
            "Unexpected finish reason for the plain text qualification prompt"
        )
    return Trial(
        first_output_milliseconds=(first_output - started) * 1000,
        elapsed_milliseconds=elapsed * 1000,
        completion_tokens_per_second=completion_tokens / elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finishes[0],
        output_sha256=hashlib.sha256("".join(text).encode()).hexdigest(),
        output_chunks=len(text),
    )


def disconnect_after_output(endpoint: Endpoint, model: str, prompt: str) -> None:
    started = time.monotonic()
    with endpoint.open(
        "/v1/chat/completions", request_body(model, prompt, 512)
    ) as response:
        for event in stream_events(response, started + endpoint.timeout):
            if isinstance(event, dict) and output_text(event):
                return
    raise ValueError("No output arrived before the cancellation probe ended")
