import hashlib
import json

from exo.reasoning_proxy._helpers import as_dict, dict_get_str


def _canonical_tool_calls(
    tool_calls: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if not tool_calls:
        return []
    result: list[dict[str, object]] = []
    for tc in tool_calls:
        entry: dict[str, object] = {}
        if "id" in tc:
            entry["id"] = tc["id"]
        fn = as_dict(tc.get("function"))
        if fn is not None:
            entry["function"] = {
                "name": dict_get_str(fn, "name") or "",
                "arguments": dict_get_str(fn, "arguments") or "",
            }
        if "type" in tc:
            entry["type"] = tc["type"]
        result.append(entry)
    return result


def hash_openai_assistant(
    content: str | list[object] | None,
    tool_calls: list[dict[str, object]] | None,
) -> str:
    """Deterministic hash of an OpenAI assistant message's observable surface.

    Canonicalizes None content to "" and tool_calls to a minimal id/function shape
    so trivial shape differences between client render and our re-emit don't miss.
    """
    shape: dict[str, object] = {
        "content": content if content is not None else "",
        "tool_calls": _canonical_tool_calls(tool_calls),
    }
    payload = json.dumps(
        shape, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_claude_assistant(content_blocks: list[dict[str, object]]) -> str:
    """Deterministic hash of a Claude assistant message's observable surface.

    Skips thinking blocks (we're hashing what the *client sends back*, which typically
    omits thinking) and normalizes tool_use blocks to id/name/input.
    """
    normalized: list[dict[str, object]] = []
    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            normalized.append(
                {"type": "text", "text": dict_get_str(block, "text") or ""}
            )
        elif btype == "tool_use":
            normalized.append(
                {
                    "type": "tool_use",
                    "id": dict_get_str(block, "id") or "",
                    "name": dict_get_str(block, "name") or "",
                    "input": block.get("input")
                    if block.get("input") is not None
                    else {},
                }
            )
    payload = json.dumps(
        normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
