"""
Tool call detection and parsing for MLX generation.

Based on MLX-LM tool calling patterns from:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/tool_use.py
"""

import json
import re
from typing import Any

from exo.worker.runner.bootstrap import logger


def detect_tool_call_format(model_id: str) -> tuple[str, str]:
    """
    Detect the tool call format based on the model.

    Different models use different formats:
    - Qwen models: <tool_call>...</tool_call>
    - Llama 3.1+: <|python_tag|>...<|eom_id|>
    - Other models may use different patterns

    Returns:
        tuple[str, str]: (opening_tag, closing_tag)
    """
    model_id_lower = model_id.lower()

    if "qwen" in model_id_lower:
        return "<tool_call>", "</tool_call>"
    elif "llama" in model_id_lower or "mistral" in model_id_lower:
        # Llama 3.1+ uses python_tag format
        return "<|python_tag|>", "<|eom_id|>"
    else:
        # Default to Qwen-style tags
        return "<tool_call>", "</tool_call>"


def parse_tool_calls(
    text: str,
    model_id: str,
) -> list[dict[str, Any]] | None:
    """
    Parse tool calls from generated text.

    Args:
        text: The generated text that may contain tool calls
        model_id: The model identifier to determine parsing format

    Returns:
        list of tool call dicts with 'id', 'type', 'function' keys,
        or None if no tool calls detected
    """
    tool_open, tool_close = detect_tool_call_format(model_id)

    # Find all tool call blocks
    tool_calls: list[dict[str, Any]] = []

    # Use regex to find tool call blocks
    if "<|python_tag|>" in tool_open:
        # Llama 3.1+ format: <|python_tag|>{"name": "func", "parameters": {...}}<|eom_id|>
        pattern = r"<\|python_tag\|>(.*?)<\|eom_id\|>"
        matches = re.finditer(pattern, text, re.DOTALL)

        for idx, match in enumerate(matches):
            tool_call_json = match.group(1).strip()

            try:
                tool_call_data = json.loads(tool_call_json)

                # Llama format uses "parameters" instead of "arguments"
                tool_calls.append({
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_call_data.get("name", ""),
                        "arguments": json.dumps(tool_call_data.get("parameters", tool_call_data.get("arguments", {}))),
                    }
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue
    else:
        # Qwen format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        start_idx = 0
        while True:
            start = text.find(tool_open, start_idx)
            if start == -1:
                break

            end = text.find(tool_close, start)
            if end == -1:
                break

            tool_call_json = text[start + len(tool_open):end].strip()

            try:
                tool_call_data = json.loads(tool_call_json)

                # Convert to OpenAI format
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_call_data.get("name", ""),
                        "arguments": json.dumps(tool_call_data.get("arguments", {})),
                    }
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")

            start_idx = end + len(tool_close)

    return tool_calls if tool_calls else None


def extract_text_without_tool_calls(
    text: str,
    model_id: str,
) -> str:
    """
    Extract the text content without tool call markers.

    Args:
        text: The generated text that may contain tool calls
        model_id: The model identifier to determine parsing format

    Returns:
        Text with tool call markers removed
    """
    tool_open, tool_close = detect_tool_call_format(model_id)

    if "<|python_tag|>" in tool_open:
        # Remove Llama 3.1+ style python_tag calls
        pattern = r"<\|python_tag\|>.*?<\|eom_id\|>"
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()
    else:
        # Remove Qwen style tool calls
        result = text
        while True:
            start = result.find(tool_open)
            if start == -1:
                break
            end = result.find(tool_close, start)
            if end == -1:
                break
            result = result[:start] + result[end + len(tool_close):]

        return result.strip()
