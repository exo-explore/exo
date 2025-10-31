#!/usr/bin/env python3
"""
Simple standalone test for tool call parsing logic.
No imports from exo package - just test the pure function.
"""

import json
import re
import uuid
from typing import Optional, List, Dict

def parse_tool_calls(content: str) -> tuple:
    """
    Parse tool calls from model output in XML format.
    Standalone version for testing.
    """
    tool_calls = []

    # Find all tool call matches
    matches = list(re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL))

    if not matches:
        return None, None, None

    # Get content before first tool call
    first_match_start = matches[0].start()
    content_before = content[:first_match_start].strip() if first_match_start > 0 else None

    # Parse each tool call
    for match in matches:
        try:
            tool_call_json = json.loads(match.group(1))

            # Ensure arguments is a JSON string (not an object)
            if "arguments" in tool_call_json and isinstance(tool_call_json["arguments"], dict):
                tool_call_json["arguments"] = json.dumps(tool_call_json["arguments"])

            # Generate unique call ID
            call_id = f"call_{uuid.uuid4().hex[:24]}"

            # Format according to OpenAI spec
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call_json.get("name", ""),
                    "arguments": tool_call_json.get("arguments", "{}")
                }
            })
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool call JSON: {match.group(1)}")
            print(f"Error: {e}")
            continue

    if tool_calls:
        return content_before, tool_calls, "tool_calls"

    return None, None, None


def test_single_tool_call():
    """Test parsing a single tool call"""
    print("\n=== Test 1: Single Tool Call ===")

    content = """Let me check the weather for you.
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "celsius"}}
</tool_call>"""

    content_before, tool_calls, finish_reason = parse_tool_calls(content)

    print(f"✓ Content before: {repr(content_before)}")
    print(f"✓ Number of tool calls: {len(tool_calls) if tool_calls else 0}")
    print(f"✓ Finish reason: {finish_reason}")

    if tool_calls:
        for tc in tool_calls:
            print(f"✓ Tool call ID: {tc['id']}")
            print(f"✓ Function name: {tc['function']['name']}")
            print(f"✓ Arguments (type={type(tc['function']['arguments']).__name__}): {tc['function']['arguments']}")

            # Parse to verify JSON
            args = json.loads(tc['function']['arguments'])
            print(f"✓ Parsed arguments: {args}")

    assert content_before == "Let me check the weather for you."
    assert len(tool_calls) == 1
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["id"].startswith("call_")
    assert tool_calls[0]["type"] == "function"

    print("✅ PASS")


def test_parallel_tool_calls():
    """Test multiple parallel tool calls"""
    print("\n=== Test 2: Parallel Tool Calls ===")

    content = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Boston"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "SF"}}
</tool_call>"""

    content_before, tool_calls, finish_reason = parse_tool_calls(content)

    print(f"✓ Number of tool calls: {len(tool_calls) if tool_calls else 0}")

    assert tool_calls is not None
    assert len(tool_calls) == 3
    assert finish_reason == "tool_calls"

    # Verify each has unique ID
    ids = [tc["id"] for tc in tool_calls]
    assert len(ids) == len(set(ids)), "Tool call IDs should be unique"

    print("✅ PASS")


def test_no_tool_calls():
    """Test regular content without tools"""
    print("\n=== Test 3: No Tool Calls ===")

    content = "Hello! How can I help you today?"
    content_before, tool_calls, finish_reason = parse_tool_calls(content)

    assert content_before is None
    assert tool_calls is None
    assert finish_reason is None

    print("✅ PASS")


def test_dict_arguments_conversion():
    """Test that dict arguments are converted to JSON strings"""
    print("\n=== Test 4: Dict Arguments Conversion ===")

    content = """<tool_call>
{"name": "calculate", "arguments": {"a": 1, "b": 2}}
</tool_call>"""

    content_before, tool_calls, finish_reason = parse_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1

    args_value = tool_calls[0]["function"]["arguments"]
    print(f"✓ Arguments type: {type(args_value).__name__}")
    print(f"✓ Arguments value: {args_value}")

    assert isinstance(args_value, str), f"Arguments should be str, got {type(args_value)}"

    # Verify it's valid JSON
    parsed = json.loads(args_value)
    assert parsed["a"] == 1
    assert parsed["b"] == 2

    print("✅ PASS")


def test_openai_format():
    """Test that output matches OpenAI spec exactly"""
    print("\n=== Test 5: OpenAI Format Compliance ===")

    content = """<tool_call>
{"name": "test_func", "arguments": {"param": "value"}}
</tool_call>"""

    content_before, tool_calls, finish_reason = parse_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1

    tc = tool_calls[0]

    # Check structure
    assert "id" in tc, "Missing 'id' field"
    assert "type" in tc, "Missing 'type' field"
    assert "function" in tc, "Missing 'function' field"

    assert tc["type"] == "function", "Type should be 'function'"

    func = tc["function"]
    assert "name" in func, "Missing function.name"
    assert "arguments" in func, "Missing function.arguments"

    assert isinstance(func["name"], str), "function.name should be string"
    assert isinstance(func["arguments"], str), "function.arguments should be JSON string"

    # Check finish_reason
    assert finish_reason == "tool_calls"

    print("✓ Structure matches OpenAI spec:")
    print(json.dumps(tc, indent=2))

    print("✅ PASS")


def main():
    print("="*60)
    print("  Tool Call Parsing Tests")
    print("="*60)

    tests = [
        test_single_tool_call,
        test_parallel_tool_calls,
        test_no_tool_calls,
        test_dict_arguments_conversion,
        test_openai_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✅ All tests passed! Implementation is correct.")
    else:
        print(f"\n❌ {failed} test(s) failed.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
