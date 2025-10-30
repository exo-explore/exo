#!/usr/bin/env python3
"""
Simple AI chat REPL with improved user experience.
Allows interactive conversation with an AI model through the command line.
Supports streaming responses from the AI model and tool calling.
Includes readline support for command history within the session.
Supports loading system messages from files.
Allows customizing the tool call format (watt or llama_json).
"""

import json
import readline
import requests
import sys
import time
import datetime
import os
import argparse
from typing import List, Dict, Iterator, Any, Callable, Optional

# Define available tools
AVAILABLE_TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "Get the current date and time",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          }
        },
        "required": ["location"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "calculate",
      "description": "Perform a calculation",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate"
          }
        },
        "required": ["expression"],
        "additionalProperties": False
      }
    }
  }
]

EXO_PORT = os.getenv("EXO_PORT", "52415")

# Implement tool functions
def get_current_time(args: Dict[str, Any] = {}) -> Dict[str, str]:
  """Get the current date and time."""
  current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  return {"time": current_time}


def get_weather(args: Dict[str, str]) -> Dict[str, str]:
  """Get the current weather for a location (simulated)."""
  location = args.get("location", "Unknown")
  # This is a mock implementation
  weathers = ["sunny", "cloudy", "rainy", "snowy", "windy"]
  temps = [f"{temp}°C" for temp in range(0, 35, 5)]

  import random
  weather = random.choice(weathers)
  temp = random.choice(temps)

  return {
    "location": location,
    "condition": weather,
    "temperature": temp,
    "note": "(This is simulated weather data for demonstration purposes)"
  }


def calculate(args: Dict[str, str]) -> Dict[str, Any]:
  """Evaluate a mathematical expression."""
  expression = args.get("expression", "")
  try:
    # Using eval is generally not safe, but this is a demo
    # In a real application, use a safer method to evaluate expressions
    result = eval(expression)
    return {"expression": expression, "result": result}
  except Exception as e:
    return {"expression": expression, "error": str(e)}


# Map function names to their implementations
TOOL_FUNCTIONS = {
  "get_current_time": get_current_time,
  "get_weather": get_weather,
  "calculate": calculate
}


def chat_completion(messages: List[Dict[str, str]], model: str = "llama-3.2-1b", stream: bool = False,
                    tools: Optional[List[Dict]] = None, tool_call_format: str = "llama_json") -> Any:
  """Send chat completion request to local exo server."""
  request_data = {
    "model": model,
    "messages": messages,
    "temperature": 0.7,
    "stream": stream
  }

  if tools:
    request_data["tools"] = tools
    request_data["tool_choice"] = "auto"
    request_data["tool_behaviour"] = { "format": tool_call_format }

  response = requests.post(
    f"http://localhost:{EXO_PORT}/v1/chat/completions",
    json=request_data,
    stream=stream
  )

  if not stream:
    return response.json()
  else:
    return response


def process_stream(response) -> Iterator[Dict[str, Any]]:
  """Process streaming response and yield content chunks or tool calls."""
  for line in response.iter_lines():
    if line:
      line = line.decode('utf-8')
      # Skip the "data: " prefix
      if line.startswith('data: '):
        line = line[6:]
        # Skip [DONE] message
        if line == '[DONE]':
          break
        try:
          data = json.loads(line)
          delta = data.get('choices', [{}])[0].get('delta', {})

          # Check if this is a tool call
          if 'tool_calls' in delta:
            yield {"type": "tool_call_part", "data": delta}
          # Otherwise it's regular content
          elif 'content' in delta and delta['content']:
            yield {"type": "content", "data": delta['content']}
        except json.JSONDecodeError:
          continue


def print_help():
  """Display available commands."""
  print("\n\033[1mAvailable Commands:\033[0m")
  print("  \033[94m/help\033[0m    - Show this help message")
  print("  \033[94m/clear\033[0m   - Clear the conversation history")
  print("  \033[94m/model\033[0m   - Show or change the current model")
  print("  \033[94m/history\033[0m - Show conversation history")
  print("  \033[94m/tools\033[0m   - Toggle tool calling (on/off)")
  print("  \033[94m/system\033[0m  - Show current system message")
  print("  \033[94m/format\033[0m  - Set tool call format (watt/llama_json)")
  print("  \033[94m/exit\033[0m    - Exit the chat")


def handle_command(command: str, messages: List[Dict[str, Any]], model: str, tools_enabled: bool,
                  tool_call_format: str) -> tuple[bool, str, bool, str]:
  """Handle special commands starting with /."""
  parts = command.split()
  cmd = parts[0].lower()

  if cmd == "/help":
    print_help()
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/clear":
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/model":
    if len(parts) > 1:
      new_model = parts[1]
      print(f"\033[92mSwitched model to: {new_model}\033[0m")
      return True, new_model, tools_enabled, tool_call_format
    else:
      print(f"\033[92mCurrent model: {model}\033[0m")
      return True, model, tools_enabled, tool_call_format

  elif cmd == "/history":
    if not messages:
      print("\033[93mNo conversation history yet.\033[0m")
    else:
      print("\n\033[1mConversation History:\033[0m")
      for i, msg in enumerate(messages):
        role = msg["role"].capitalize()
        if "content" in msg and msg["content"]:
          content = msg["content"]
          if role == "User":
            print(f"\033[94mYou:\033[0m {content}")
          elif role == "Assistant":
            print(f"\033[92mAI:\033[0m {content}")
          elif role == "System":
            print(f"\033[93mSystem:\033[0m {content}")
          elif role == "Tool":
            try:
              result = json.loads(content)
              tool_id = msg.get("tool_call_id", "unknown")
              # Find the corresponding tool call to get the function name
              tool_call = None
              for m in messages:
                if "tool_calls" in m:
                  for tc in m["tool_calls"]:
                    if tc["id"] == tool_id:
                      tool_call = tc
                      break
                if tool_call:
                  break

              function_name = tool_call["function"]["name"] if tool_call else "unknown"

              # Format tool results in a more conversational way
              if function_name == "get_current_time":
                print(f"\033[96mSystem: The current time is {result['time']}\033[0m")
              elif function_name == "get_weather":
                print(f"\033[96mSystem: Weather for {result['location']}: {result['condition']}, {result['temperature']}\033[0m")
              elif function_name == "calculate":
                if "error" in result:
                  print(f"\033[96mSystem: Calculation error: {result['error']}\033[0m")
                else:
                  print(f"\033[96mSystem: {result['expression']} = {result['result']}\033[0m")
              else:
                print(f"\033[96mSystem: Result from {function_name}: {json.dumps(result, indent=2)}\033[0m")
            except:
              print(f"\033[96mSystem: {content}\033[0m")
        elif "tool_calls" in msg:
          for tool_call in msg["tool_calls"]:
            func_name = tool_call.get("function", {}).get("name", "")
            print(f"\033[95mAI is using {func_name}...\033[0m")
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/tools":
    if len(parts) > 1:
      if parts[1].lower() in ("on", "true", "1", "yes"):
        tools_enabled = True
        print("\033[92mTool calling enabled\033[0m")
      elif parts[1].lower() in ("off", "false", "0", "no"):
        tools_enabled = False
        print("\033[93mTool calling disabled\033[0m")
    else:
      status = "enabled" if tools_enabled else "disabled"
      print(f"\033[92mTool calling is currently {status}\033[0m")
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/format":
    if len(parts) > 1:
      format_option = parts[1].lower()
      if format_option in ("watt", "llama_json"):
        tool_call_format = format_option
        print(f"\033[92mTool call format set to: {tool_call_format}\033[0m")
      else:
        print("\033[91mInvalid format. Use 'watt' or 'llama_json'.\033[0m")
    else:
      print(f"\033[92mCurrent tool call format: {tool_call_format}\033[0m")
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/system":
    system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
    if system_message:
      print("\n\033[1mCurrent System Message:\033[0m")
      print(f"\033[93m{system_message}\033[0m")
    else:
      print("\033[93mNo system message set.\033[0m")
    return True, model, tools_enabled, tool_call_format

  elif cmd == "/exit":
    print("\033[92mGoodbye!\033[0m")
    sys.exit(0)

  return False, model, tools_enabled, tool_call_format


def format_user_prompt():
  """Format the user prompt with color."""
  return "\033[94mYou:\033[0m "


def format_ai_prompt():
  """Format the AI prompt with color."""
  return "\033[92mAI:\033[0m "


def handle_tool_calls(tool_calls: List[Dict], messages: List[Dict]) -> None:
  """Process tool calls and add results to messages."""
  for tool_call in tool_calls:
    function_name = tool_call["function"]["name"]
    function_args = json.loads(tool_call["function"]["arguments"])
    tool_call_id = tool_call["id"]

    # More conversational format for tool calls
    print(f"\n\033[95mAI is using {function_name}...\033[0m")

    # Execute the function
    if function_name in TOOL_FUNCTIONS:
      result = TOOL_FUNCTIONS[function_name](function_args)

      # Format the result in a more conversational way
      if function_name == "get_current_time":
        print(f"\033[96mSystem: The current time is {result['time']}\033[0m")
      elif function_name == "get_weather":
        print(f"\033[96mSystem: Weather for {result['location']}: {result['condition']}, {result['temperature']}\033[0m")
        if "note" in result:
          print(f"\033[90m{result['note']}\033[0m")
      elif function_name == "calculate":
        if "error" in result:
          print(f"\033[96mSystem: Calculation error: {result['error']}\033[0m")
        else:
          print(f"\033[96mSystem: {result['expression']} = {result['result']}\033[0m")
      else:
        # Generic formatting for other tools
        print(f"\033[96mSystem: Result from {function_name}: {json.dumps(result, indent=2)}\033[0m")

      # Add the function result to messages
      messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result)
      })
    else:
      print(f"\033[91mSystem: Function {function_name} not found\033[0m")

      # Add the error result to messages
      messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({"error": f"Function {function_name} not found"})
      })


def load_system_message(file_path: str) -> str:
  """Load system message from a file, replacing {functions} with tool definitions."""
  try:
    with open(file_path, 'r') as f:
      system_message = f.read()

    # Replace {functions} placeholder with tool definitions if present
    if "{functions}" in system_message:
      functions_str = json.dumps([tool["function"] for tool in AVAILABLE_TOOLS
                                 if tool["type"] == "function"], indent=2)
      system_message = system_message.replace("{functions}", functions_str)

    return system_message
  except Exception as e:
    print(f"\033[91mError loading system message: {str(e)}\033[0m")
    return ""


def parse_arguments():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(description="AI Chat REPL")
  parser.add_argument("--system", "-s", type=str, help="Path to a file containing the system message")
  parser.add_argument("--model", "-m", type=str, default="llama-3.2-1b",
                     help="Default model to use (default: llama-3.2-1b)")
  parser.add_argument("--tool-format", "-f", type=str, default="llama_json", choices=["watt", "llama_json"],
                     help="Format for tool calls: watt or llama_json (default: llama_json)")
  return parser.parse_args()


def main():
  # Parse command line arguments
  args = parse_arguments()

  # Initialize conversation history and model
  messages = []
  model = args.model
  tools_enabled = True
  tool_call_format = args.tool_format

  # Load system message if provided
  if args.system:
    system_message = load_system_message(args.system)
    if system_message:
      messages.append({
        "role": "system",
        "content": system_message
      })
      print(f"\033[92mLoaded system message from {args.system}\033[0m")

  # Configure readline
  readline.parse_and_bind('tab: complete')

  # Print welcome message
  print("\033[1mAI Chat REPL\033[0m")
  print("Type \033[94m/help\033[0m for available commands")
  print("Press Ctrl+C to clear current input, Ctrl+D or type \033[94m/exit\033[0m to quit")
  print("Tool calling is \033[92menabled\033[0m by default (use \033[94m/tools off\033[0m to disable)")
  print(f"Tool call format is set to \033[92m{tool_call_format}\033[0m (use \033[94m/format\033[0m to change)")

  try:
    while True:
      # Get user input with handling for Ctrl+C and Ctrl+D
      try:
        user_input = input(format_user_prompt()).strip()
      except KeyboardInterrupt:
        # Ctrl+C: clear current input line
        print("\n\033[93mInput cleared\033[0m")
        continue
      except EOFError:
        # Ctrl+D: exit gracefully
        print("\033[92mGoodbye!\033[0m")
        break

      # Skip empty inputs
      if not user_input:
        continue

      # Check for commands
      if user_input.startswith('/'):
        is_command, model, tools_enabled, tool_call_format = handle_command(
          user_input, messages, model, tools_enabled, tool_call_format)
        if is_command:
          if user_input.lower() == "/clear":
            messages = []
            print("\033[93mConversation history cleared.\033[0m")
          continue

      # Add user message to history
      messages.append({
        "role": "user",
        "content": user_input
      })

      # Get AI response
      try:
        print(format_ai_prompt(), end="", flush=True)

        # Use streaming response
        start_time = time.time()
        tools_param = AVAILABLE_TOOLS if tools_enabled else None
        response = chat_completion(messages, model=model, stream=True,
                                  tools=tools_param, tool_call_format=tool_call_format)

        # Process the streaming response
        full_content = ""
        current_tool_calls = []
        tool_call_buffer = {}

        for chunk in process_stream(response):
          if chunk["type"] == "content":
            content = chunk["data"]
            full_content += content
            print(content, end="", flush=True)
          elif chunk["type"] == "tool_call_part":
            # Handle tool call parts
            delta = chunk["data"]
            if "tool_calls" in delta:
              for tool_call_delta in delta["tool_calls"]:
                index = tool_call_delta.get("index", 0)

                # Initialize the tool call if it doesn't exist
                if index >= len(current_tool_calls):
                  current_tool_calls.append({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                  })

                # Update the tool call with the delta
                if "id" in tool_call_delta:
                  current_tool_calls[index]["id"] = tool_call_delta["id"]

                if "function" in tool_call_delta:
                  function_delta = tool_call_delta["function"]
                  if "name" in function_delta:
                    current_tool_calls[index]["function"]["name"] += function_delta["name"]
                  if "arguments" in function_delta:
                    current_tool_calls[index]["function"]["arguments"] += function_delta["arguments"]

        # Add complete message to history
        if current_tool_calls:
          # If we have tool calls, add them to the message
          messages.append({
            "role": "assistant",
            "content": full_content if full_content else None,
            "tool_calls": current_tool_calls
          })

          # Process the tool calls
          print()  # Add a newline after the AI's message
          handle_tool_calls(current_tool_calls, messages)

          # Get the follow-up response after tool calls
          print(format_ai_prompt(), end="", flush=True)
          follow_up_response = chat_completion(messages, model=model, stream=True)
          follow_up_content = ""

          for chunk in process_stream(follow_up_response):
            if chunk["type"] == "content":
              content = chunk["data"]
              follow_up_content += content
              print(content, end="", flush=True)

          # Add the follow-up response to history
          if follow_up_content:
            messages.append({
              "role": "assistant",
              "content": follow_up_content
            })
        else:
          # If no tool calls, just add the content
          messages.append({
            "role": "assistant",
            "content": full_content
          })

        # Show response time for longer responses
        elapsed = time.time() - start_time
        if elapsed > 1.0:
          print(f"\n\033[90m(Response time: {elapsed:.2f}s)\033[0m")
        else:
          print()  # Add a newline after response

      except Exception as e:
        messages.pop()
        raise e

  except KeyboardInterrupt:
    # This should not be reached with the inner try/except, but just in case
    print("\n\033[92mGoodbye!\033[0m")


if __name__ == "__main__":
  main()
