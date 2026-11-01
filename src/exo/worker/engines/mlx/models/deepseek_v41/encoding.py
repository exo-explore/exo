# type: ignore
"""
DeepSeek-V4.1 Encoding

Vendored from DeepSeek's reference prompt encoder for DeepSeek-V4.1 (the encoder the model
card names as the prompt format; the repository's chat_template.jinja is a community port).

Source:   https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding/encoding.py
Revision: 314113f8a7dadb388215396ba5fbee2c610a04d6 ("fix(encoding): preserve tool image order and
          current-turn thinking", 2026-10-16), git blob 3c7c357dc9ae3ed37adacb8572a2e9498b7d1bcb.
          deepseek-ai/DeepSeek-V4.1-Flash-1017 carries the same encoder.
Vendored: 2026-10-30.
Licence:  MIT (notice below).

Local changes, none of which change behaviour:
- formatted with ruff (line wrapping, quotes, import order);
- this docstring replaces the original module docstring;
- `# type: ignore` for the whole file and two `# noqa` comments (E722, N806);
- in `render_message`, an `if`/`elif` pair with the same body merged into one condition.

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_TAG_PATTERN = re.compile(r"<image>(.*?)</image>", re.DOTALL)

# Task special tokens for internal classification tasks
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

# ============================================================
# Templates
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

response_format_template: str = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"

tool_output_template: str = "<tool_result>{content}</tool_result>"

# ============================================================
# Utility Functions
# ============================================================


def to_json(value: Any) -> str:
    """Serialize a value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except:  # noqa: E722
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    """Extract function definitions with namespace-qualified names."""
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = (
                namespace["description"] + "\n" + (function.get("description") or "")
            )
        functions.append(function)
    return functions


def _split_tool_name(
    name: str, namespace: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Split a qualified name and validate any explicit namespace."""
    prefix, separator, bare_name = name.partition("::")
    if separator:
        assert namespace in (None, prefix), (
            f"Conflicting tool namespaces: {namespace} != {prefix}"
        )
        namespace, name = prefix, bare_name
    assert "::" not in name, f"Tool name must not contain '::': {name}"
    assert namespace is None or "::" not in namespace, (
        f"Tool namespace must not contain '::': {namespace}"
    )
    return namespace, name


def _tool_name_for_encoding(tool: Dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = _split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tool_calls_from_openai_format(tool_calls):
    """Convert OpenAI-format tool calls to internal format."""
    calls = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        namespace, name = _split_tool_name(
            function["name"], tool_call.get("namespace") or function.get("namespace")
        )
        call = {"name": name, "arguments": function["arguments"]}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def tool_calls_to_openai_format(tool_calls):
    """Convert internal tool calls to OpenAI format."""
    calls = []
    for tool_call in tool_calls:
        call = {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        if tool_call.get("namespace") is not None:
            call["namespace"] = tool_call["namespace"]
        calls.append(call)
    return calls


def decode_dsml_to_arguments(
    tool_name: str, tool_args: Dict[str, Tuple[str, str]]
) -> Dict[str, str]:
    """
    Decode DSML parameters back to a tool call dict.

    Args:
        tool_name: Name of the tool.
        tool_args: Dict mapping param_name -> (value, is_string_flag).

    Returns:
        Dict with "name", "arguments" (JSON string), and optional "namespace".
    """

    def _decode_value(key: str, value: str, string: str):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = (
        "{"
        + ", ".join(
            [_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]
        )
        + "}"
    )
    namespace, name = _split_tool_name(tool_name)
    tool_call = dict(name=name, arguments=tool_args_json)
    if namespace is not None:
        tool_call["namespace"] = namespace
    return tool_call


# ============================================================
# Preprocessing
# ============================================================


def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge tool messages into the preceding user message using content_blocks format.

    DeepSeek-V4.1 does not have a standalone "tool" role; instead, tool results
    are encoded as <tool_result> blocks within user messages.
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            # Convert tool message to a user message with tool_result block
            content = msg.get("content", "")
            if msg.get("content_blocks") and not isinstance(msg.get("content"), str):
                content = msg["content_blocks"]
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
            # Merge into previous message if it's already a user (merged tool)
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append(
                    {
                        "role": "user",
                        "content_blocks": [tool_block],
                    }
                )
        elif role == "user":
            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content = msg.get("content", "")
                content_blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                # Preserve structured content and all message-level metadata.
                new_msg = msg
                new_msg["content_blocks"] = content_blocks
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sort tool_result blocks within user messages by the order of tool_calls
    in the preceding assistant message.
    """
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Vision Message Preprocessing
# ============================================================


def parse_tagged_text(text: str) -> Union[str, List[Dict[str, Any]]]:
    """Convert ``<image>path</image>`` text into standard content blocks."""
    matches = list(IMAGE_TAG_PATTERN.finditer(text))
    remaining = IMAGE_TAG_PATTERN.sub("", text)
    if "<image>" in remaining or "</image>" in remaining:
        raise ValueError("Malformed <image>path</image> tag")
    if not matches:
        return text

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            blocks.append({"type": "text", "text": text[cursor : match.start()]})
        path = match.group(1)
        if not path:
            raise ValueError("Image path must not be empty")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": path},
            }
        )
        cursor = match.end()
    if cursor < len(text):
        blocks.append({"type": "text", "text": text[cursor:]})
    return blocks


def _is_image_block(block: Dict[str, Any]) -> bool:
    """Return whether a content block is an OpenAI/Anthropic/internal image."""
    return isinstance(block, dict) and block.get("type") in ("image", "image_url")


def _extract_image(block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a supported image block into an internal image record."""
    record: Dict[str, Any] = {"type": "image"}
    if block.get("type") == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            record["url"] = image_url
        else:
            record["url"] = (image_url or {}).get("url", "")
    else:
        for key in ("source", "url", "data"):
            if key in block:
                record[key] = block[key]
    if not any(record.get(key) for key in ("source", "url", "data")):
        raise ValueError("Image block does not contain a valid source")
    return record


def _process_image_blocks(
    blocks: List[Any], image_placeholder: str = IMAGE_PLACEHOLDER
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Replace image blocks and collect their records in one ordered traversal."""
    new_blocks: List[Any] = []
    images: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if _is_image_block(block):
            new_blocks.append({"type": "text", "text": image_placeholder})
            images.append(_extract_image(block))
        elif block.get("type") == "tool_result" and isinstance(
            block.get("content"), list
        ):
            block = copy.copy(block)
            block["content"], nested_images = _process_image_blocks(
                block["content"], image_placeholder
            )
            new_blocks.append(block)
            images.extend(nested_images)
        elif block.get("type") == "text":
            text = block.get("text") or ""
            if IMAGE_PLACEHOLDER in text:
                raise ValueError(
                    f"Text block contains image placeholder '{IMAGE_PLACEHOLDER}': "
                    f"'{text[:100]}'. Images should be separate content blocks."
                )
            new_blocks.append(block)
        else:
            new_blocks.append(block)
    return new_blocks, images


def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    """Reject user-supplied image placeholder tokens in textual fields."""
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError(
            f"Message content contains image special token '{IMAGE_PLACEHOLDER}'. "
            "Images should be provided as image content blocks."
        )
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            f"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
        )


def process_image_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize image blocks and return their records in prompt order."""
    processed: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        _validate_no_image_sp_tokens(msg)

        if isinstance(msg.get("content"), list) and "content_blocks" not in msg:
            msg["content_blocks"] = msg.pop("content")

        if msg.get("content_blocks"):
            msg["content_blocks"], message_images = _process_image_blocks(
                msg["content_blocks"]
            )
            images.extend(message_images)
            if not isinstance(msg.get("content"), str):
                texts = [
                    block.get("text", "")
                    for block in msg["content_blocks"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                msg["content"] = "\n\n".join(texts)

        processed.append(msg)
    return processed, images


def _read_until_stop(
    index: int, text: str, stop: List[str]
) -> Tuple[int, str, Optional[str]]:
    """
    Read text from index until one of the stop strings is found.

    Returns:
        Tuple of (new_index, content_before_stop, matched_stop_string_or_None).
    """
    min_pos = len(text)
    matched_stop = None

    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s

    if matched_stop:
        content = text[index:min_pos]
        return min_pos + len(matched_stop), content, matched_stop
    else:
        content = text[index:]
        return len(text), content, None


# ============================================================
# V4.1 Special Tokens and DSML Tag Names
# ============================================================

SYSTEM_SP_TOKEN = "<｜System｜>"

tool_calls_block_name: str = " calls"
tool_call_tag_name: str = " invoke"
tool_parameter_tag_name: str = " parameter"

tool_call_template: str = '<{dsml_token}{tool_call_tag_name} name="{name}">\n{arguments}\n</{dsml_token}{tool_call_tag_name}>'
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)

# ============================================================
# Reasoning Effort (numeric budget)
# ============================================================

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 50,
    "high": 75,
    "max": 100,
}
DEFAULT_REASONING_EFFORT = "high"


def render_reasoning_effort(
    index: int,
    thinking_mode: str,
    effort: Union[str, int, None],
) -> str:
    """Render the V4.1 numeric reasoning effort prefix (thinking mode, index 0 only)."""
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    assert (
        type(effort) is int and 1 <= effort <= 100
    ) or effort in REASONING_EFFORT_MAPPINGS, (
        "Invalid reasoning effort for deepseek_v41: "
        f"{effort}, should be int within [1,100] or {list(REASONING_EFFORT_MAPPINGS)}"
    )
    if type(effort) is str:
        effort = REASONING_EFFORT_MAPPINGS[effort]
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_TEMPLATE.format(budget=effort)
    return ""


# ============================================================
# Tools rendering
# ============================================================

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    """Render tool schemas into the V4.1 system prompt format."""
    tools_json = [to_json(t) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        tc_block_name=tool_calls_block_name,
        tool_call_tag_name=tool_call_tag_name,
        tool_parameter_tag_name=tool_parameter_tag_name,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def encode_arguments_to_dsml(tool_call: Dict[str, Any]) -> str:
    """Encode tool call arguments into V4.1 DSML parameter format."""
    p_dsml_template = (
        '<{dsml_token}{tool_parameter_tag_name} name="{key}" string="{is_str}">'
        "{value}</{dsml_token}{tool_parameter_tag_name}>"
    )
    P_dsml_strs = []  # noqa: N806

    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict):
        # Tolerate JSON strings, including double-encoded ones.
        for _ in range(2):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    break
            else:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call["arguments"]}

    for k, v in arguments.items():
        P_dsml_strs.append(
            p_dsml_template.format(
                dsml_token=dsml_token,
                tool_parameter_tag_name=tool_parameter_tag_name,
                key=k,
                is_str="true" if isinstance(v, str) else "false",
                value=v if isinstance(v, str) else to_json(v),
            )
        )

    return "\n".join(P_dsml_strs)


# ============================================================
# Message Rendering
# ============================================================


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """
    Find the index of the last user message.

    V4.1 supports mid-conversation system messages, which count as user
    messages for the purposes of the assistant generation header.
    User messages containing only tool results do not start a new turn.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "user" or (role == "system" and idx > 0):
            content_blocks = messages[idx].get("content_blocks")
            if content_blocks and not any(
                block.get("type") in ("text", "image", "image_url")
                for block in content_blocks
            ):
                continue
            last_user_index = idx
            break
    return last_user_index


def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    """
    Render a single message at the given index into its V4.1 encoded string form.
    """
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], (
        f"Invalid thinking_mode `{thinking_mode}`"
    )

    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    # Reasoning effort prefix (thinking mode, index 0 only)
    reasoning_effort_prompt = render_reasoning_effort(
        index, thinking_mode, reasoning_effort
    )
    # System token leads the conversation when there is a reasoning effort prompt
    # or the first message is a system message.
    prompt = (
        SYSTEM_SP_TOKEN
        if index == 0 and (reasoning_effort_prompt or role == "system")
        else ""
    )
    prompt += reasoning_effort_prompt

    if role == "system":
        if index > 0:
            # Mid-conversation system message
            prompt += SYSTEM_SP_TOKEN
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "user":
        prompt += USER_SP_TOKEN

        # Handle content blocks (tool results mixed with text)
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(
            content=content
        )

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    tool_call_tag_name=tool_call_tag_name,
                    name=_tool_name_for_encoding(tc),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            tc_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        rc = reasoning_content or ""

        # Check if previous message has a task - if so, this is a task output (no thinking)
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (
                    thinking_template.format(reasoning_content=rc) + thinking_end_token
                )
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # Append transition tokens based on what follows
    if index + 1 < len(messages) and messages[index + 1].get("role") not in [
        "assistant",
        "latest_reminder",
    ]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        # Task special token for internal classification tasks
        assert task in VALID_TASKS, (
            f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        )
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            # Non-action tasks: append task sp token directly after the message
            prompt += task_sp_token
        else:
            # Action task: append Assistant + thinking token + action sp token
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_end_token
                if thinking_mode != "thinking"
                else thinking_start_token
            )
            prompt += task_sp_token

    elif role == "user" or (role == "system" and index > 0):
        # Normal generation: append Assistant + thinking token
        # (mid-conversation system messages also trigger the assistant header)
        prompt += ASSISTANT_SP_TOKEN
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= last_user_idx
        ):
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Main Encoding Function
# ============================================================


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop reasoning_content and non-essential messages before the last user message.
    Same as V4, but uses the V4.1 last-user definition (mid systems count).
    """
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)

    return result


def _encode_messages_text(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    """Encode preprocessed (text-only) messages into the V4.1 prompt format."""
    context = context if context else []

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    # Resolve drop_thinking: if any message has tools defined, don't drop thinking
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Union[str, int, None] = None,
    return_multi_modal_data: bool = False,
) -> Any:
    """Encode text or multimodal messages into the DeepSeek-V4.1 prompt format.

    Text-only calls return the prompt string. When return_multi_modal_data is
    true, the result is ``(prompt, media_data)``.
    """
    # Validate before merging can discard message-level textual fields.
    for msg in (context or []) + messages:
        _validate_no_image_sp_tokens(msg)

    # Collect images only after tool results are in their final prompt order.
    context = merge_tool_messages(context or [])
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    processed_context, _ = process_image_messages(context) if context else ([], [])
    processed_messages, images = process_image_messages(messages)
    prompt = _encode_messages_text(
        processed_messages,
        thinking_mode=thinking_mode,
        context=processed_context if processed_context else None,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )
    if return_multi_modal_data:
        return prompt, {"images": images}
    return prompt


def load_cases(input_file: str) -> List[Dict[str, Any]]:
    """Load one or more OpenAI-format conversation cases from JSON."""
    with open(input_file) as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = [data]
    elif data and isinstance(data[0], dict) and "role" in data[0]:
        data = [{"messages": data}]

    cases = []
    for case in data:
        messages = copy.deepcopy(case["messages"])
        if "tools" in case:
            if not messages:
                raise ValueError("A case with tools must contain at least one message")
            messages[0]["tools"] = case["tools"]
        cases.append(
            {
                "messages": messages,
                "context": case.get("context"),
                "thinking_mode": case.get("thinking_mode"),
                "reasoning_effort": case.get("reasoning_effort"),
            }
        )
    return cases


def encode_case(
    case: Dict[str, Any], thinking_mode: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """Encode one JSON case and return its current-turn image records."""
    prompt, media_data = encode_messages(
        case["messages"],
        thinking_mode=case.get("thinking_mode") or thinking_mode,
        context=case.get("context"),
        reasoning_effort=case.get("reasoning_effort"),
        return_multi_modal_data=True,
    )
    return prompt, media_data["images"]


# ============================================================
# Parsing (Decoding model output)
# ============================================================


def parse_tool_calls(
    index: int, text: str
) -> Tuple[int, Optional[str], List[Dict[str, str]]]:
    """
    Parse V4.1 DSML tool calls from text starting at the given index.

    Returns:
        Tuple of (new_index, last_stop_token, list_of_tool_call_dicts).
    """
    tool_calls: List[Dict[str, Any]] = []
    stop_token = None
    tool_calls_end_token = f"</{dsml_token}{tool_calls_block_name}>"
    tool_call_start_token = f"<{dsml_token}{tool_call_tag_name}"
    tool_call_end_token = f"</{dsml_token}{tool_call_tag_name}"
    tool_parameter_start_token = f"<{dsml_token}{tool_parameter_tag_name}"
    tool_parameter_end_token = f"/{dsml_token}{tool_parameter_tag_name}"

    while index < len(text):
        index, _, stop_token = _read_until_stop(
            index, text, [tool_call_start_token, tool_calls_end_token]
        )
        if _ != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{_}'")

        if stop_token == tool_calls_end_token:
            break

        if stop_token is None:
            raise ValueError("Missing special token in tool calls")

        index, tool_name_content, stop_token = _read_until_stop(
            index, text, [tool_parameter_start_token, tool_call_end_token]
        )

        p_tool_name = re.findall(
            r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL
        )
        if len(p_tool_name) != 1:
            raise ValueError(f"Tool name format error: '{tool_name_content}'")
        tool_name = p_tool_name[0]

        tool_args: Dict[str, Tuple[str, str]] = {}
        while stop_token == tool_parameter_start_token:
            index, param_content, stop_token = _read_until_stop(
                index, text, [tool_parameter_end_token]
            )

            param_kv = re.findall(
                r'^ name="(.*?)" string="(true|false)">(.*?)<$',
                param_content,
                flags=re.DOTALL,
            )
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]

            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)

            index, content, stop_token = _read_until_stop(
                index, text, [tool_parameter_start_token, tool_call_end_token]
            )
            if content != ">\n":
                raise ValueError(
                    f"Parameter format error: expected '>\\n' but got '{content}'"
                )

        tool_call = decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args)
        tool_calls.append(tool_call)

    return index, stop_token, tool_calls


def parse_message_from_completion_text(text: str, thinking_mode: str) -> Dict[str, Any]:
    """
    Parse a model completion text into a structured assistant message (V4.1 format).

    Returns:
        Dict with keys: "role", "content", "reasoning_content", "tool_calls".
        tool_calls are in OpenAI format.
    """
    summary_content, reasoning_content, tool_calls = "", "", []
    index, stop_token = 0, None
    tool_calls_start_token = f"\n\n<{dsml_token}{tool_calls_block_name}"

    is_thinking = thinking_mode == "thinking"
    is_tool_calling = False

    if is_thinking:
        index, content_delta, stop_token = _read_until_stop(
            index, text, [thinking_end_token, tool_calls_start_token]
        )
        reasoning_content = content_delta
        assert stop_token == thinking_end_token, (
            "Invalid thinking format: missing </think>"
        )

    index, content_delta, stop_token = _read_until_stop(
        index, text, [eos_token, tool_calls_start_token]
    )
    summary_content = content_delta
    if stop_token == tool_calls_start_token:
        is_tool_calling = True
    else:
        assert stop_token == eos_token, "Invalid format: missing EOS token"

    if is_tool_calling:
        index, stop_token, tool_calls = parse_tool_calls(index, text)

        index, tool_ends_text, stop_token = _read_until_stop(index, text, [eos_token])
        assert not tool_ends_text, "Unexpected content after tool calls"

    assert len(text) == index and stop_token in [eos_token, None], (
        "Unexpected content at end"
    )

    for sp_token in [
        bos_token,
        eos_token,
        thinking_start_token,
        thinking_end_token,
        dsml_token,
    ]:
        assert sp_token not in summary_content and sp_token not in reasoning_content, (
            f"Unexpected special token '{sp_token}' in content"
        )

    return {
        "role": "assistant",
        "content": summary_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls_to_openai_format(tool_calls),
    }
