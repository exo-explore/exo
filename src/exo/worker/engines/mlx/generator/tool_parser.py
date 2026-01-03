"""
Tool call detection and parsing for MLX generation.

Based on MLX-LM tool calling patterns from:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/tool_use.py
"""

import json
import re
from typing import Any, cast

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

    # First, try to find raw JSON tool calls (for parallel tool calling)
    # Pattern: {"name": "func_name", "parameters": {...} } or { "name": ... }
    # Use a simple approach: find all {"name": or { "name": strings and try to parse from there
    idx = 0
    search_pos = 0
    while True:
        # Find the start of a potential tool call (try both with and without space)
        start1 = text.find('{"name":', search_pos)
        start2 = text.find('{ "name":', search_pos)

        # Use whichever appears first (or -1 if neither found)
        if start1 == -1 and start2 == -1:
            break
        elif start1 == -1:
            start = start2
        elif start2 == -1:
            start = start1
        else:
            start = min(start1, start2)

        # Try to find the matching closing brace by counting braces
        brace_count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if end > start:
            potential_json = text[start:end]
            try:
                tool_data = cast(dict[str, Any], json.loads(potential_json))
                if "name" in tool_data and "parameters" in tool_data:
                    tool_calls.append({
                        "id": f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": cast(str, tool_data["name"]),
                            "arguments": json.dumps(cast(dict[str, Any], tool_data["parameters"])),
                        }
                    })
                    idx += 1
            except json.JSONDecodeError:
                pass

        search_pos = start + 1

    # If we found raw JSON tool calls, return them
    if tool_calls:
        return tool_calls

    # Otherwise, use regex to find tool call blocks with tags
    if "<|python_tag|>" in tool_open:
        # Llama 3.1+ format: <|python_tag|>{"name": "func", "parameters": {...}}<|eom_id|>
        pattern = r"<\|python_tag\|>(.*?)<\|eom_id\|>"
        matches = re.finditer(pattern, text, re.DOTALL)

        for idx, match in enumerate(matches):
            tool_call_json = match.group(1).strip()

            try:
                tool_call_data = cast(dict[str, Any], json.loads(tool_call_json))

                # Llama format uses "parameters" instead of "arguments"
                name = cast(str, tool_call_data.get("name", ""))
                default_params = cast(dict[str, Any], tool_call_data.get("arguments", {}))
                parameters = cast(dict[str, Any], tool_call_data.get("parameters", default_params))
                tool_calls.append({
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(parameters),
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
                tool_call_data = cast(dict[str, Any], json.loads(tool_call_json))

                # Convert to OpenAI format
                name = cast(str, tool_call_data.get("name", ""))
                arguments = cast(dict[str, Any], tool_call_data.get("arguments", {}))
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")

            start_idx = end + len(tool_close)

    return tool_calls if tool_calls else None


def normalize_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Normalize tool call arguments to fix common LLM mistakes.

    Best practices for 2025:
    - Add sensible defaults for missing required parameters
    - Auto-fix type mismatches (string "true" -> boolean true)
    - Validate before execution to prevent errors reaching user

    Args:
        tool_calls: List of tool calls with potentially incomplete arguments
        tool_definitions: Optional tool definitions from request to check schemas

    Returns:
        List of tool calls with normalized arguments
    """
    # Helper to find tool definition for a function
    def get_tool_schema(func_name: str) -> dict[str, Any] | None:
        if not tool_definitions:
            return None
        for tool_def in tool_definitions:
            func = cast(dict[str, Any], tool_def.get("function", {}))
            tool_name = cast(str, func.get("name", ""))
            if tool_name == func_name:
                return cast(dict[str, Any], func.get("parameters", {}))
        return None

    normalized: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        normalized_call = dict(tool_call)  # Shallow copy
        function_data = cast(dict[str, Any], tool_call.get("function", {}))
        function_name = cast(str, function_data.get("name", ""))

        try:
            arguments_str = cast(str, function_data.get("arguments", "{}"))
            arguments = cast(dict[str, Any], json.loads(arguments_str))

            # Get tool schema if available
            tool_schema = get_tool_schema(function_name)

            # Normalize based on function name
            if function_name == "bash":
                # Add missing required 'description' field with sensible default
                # Only if schema defines description as required or if no schema available
                should_add_description = True
                if tool_schema:
                    required_fields = cast(list[str], tool_schema.get("required", []))
                    # Only add if description is in required fields
                    should_add_description = "description" in required_fields

                if should_add_description and "description" not in arguments and "command" in arguments:
                    cmd = cast(str, arguments["command"])
                    # Generate description from command (first 50 chars)
                    arguments["description"] = f"Execute: {cmd[:50]}"
                    logger.info(f"[NORMALIZATION] Added default description for bash command: {cmd[:30]}")

            elif function_name == "edit":
                # Fix type mismatch: string "true"/"false" -> boolean
                replace_all = arguments.get("replaceAll")
                if isinstance(replace_all, str):
                    arguments["replaceAll"] = replace_all.lower() == "true"
                    logger.info(f"[NORMALIZATION] Fixed replaceAll type: {replace_all} -> {arguments['replaceAll']}")

            elif function_name == "write":
                # Ensure filePath is present - ADD it while keeping 'file' for compatibility
                if "filePath" not in arguments and "file" in arguments:
                    # Copy 'file' to 'filePath' without removing 'file'
                    # This ensures compatibility with tools expecting either field
                    arguments["filePath"] = arguments["file"]
                    logger.info(f"[NORMALIZATION] Added 'filePath' from 'file' for write (keeping both)")

            elif function_name == "read":
                # Ensure filePath is present - ADD it while keeping 'file' for compatibility
                if "filePath" not in arguments and "file" in arguments:
                    # Copy 'file' to 'filePath' without removing 'file'
                    # This ensures compatibility with tools expecting either field
                    arguments["filePath"] = arguments["file"]
                    logger.info(f"[NORMALIZATION] Added 'filePath' from 'file' for read (keeping both)")

            # Update the normalized call
            normalized_call["function"] = {
                "name": function_name,
                "arguments": json.dumps(arguments)
            }
            normalized_call["id"] = tool_call.get("id", f"call_{len(normalized)}")
            normalized_call["type"] = "function"

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to normalize tool call {function_name}: {e}")
            # Keep original if normalization fails
            normalized_call = tool_call

        normalized.append(normalized_call)

    return normalized


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
