import json
import re
import requests

def get_current_weather(location: str, unit: str = "celsius"):
  """Mock weather data function"""
  # Hardcoded response for demo purposes
  return {
    "location": location,
    "temperature": 22 if unit == "celsius" else 72,
    "unit": unit,
    "forecast": "Sunny with light clouds"
  }

def try_parse_tool_calls(content: str):
  """Try parse the tool calls."""
  tool_calls = []
  offset = 0
  for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
    if i == 0:
      offset = m.start()
    try:
      func = json.loads(m.group(1))
      tool_calls.append({"type": "function", "function": func})
      if isinstance(func["arguments"], str):
        func["arguments"] = json.loads(func["arguments"])
    except json.JSONDecodeError as e:
      print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
      pass
  if tool_calls:
    if offset > 0 and content[:offset].strip():
      c = content[:offset]
    else:
      c = ""
    return {"role": "assistant", "content": c, "tool_calls": tool_calls}
  return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

def chat_completion(messages):
  """Send chat completion request to local server"""
  response = requests.post(
    "http://localhost:52415/v1/chat/completions",
    json={
      "model": "qwen-2.5-1.5b",
      "messages": messages,
      "tools": [{
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }],
      "tool_choice": "auto"
    }
  )
  return response.json()

def main():
  # Initial conversation
  messages = [{
    "role": "user",
    "content": "Hi there, what's the weather in Boston?"
  }]
  
  # Get initial response
  response = chat_completion(messages)
  print(f"First response: {response}")
  assistant_message = try_parse_tool_calls(response["choices"][0]["message"]["content"])
  messages.append(assistant_message)
  
  # If there are tool calls, execute them and continue conversation
  if "tool_calls" in assistant_message:
    for tool_call in assistant_message["tool_calls"]:
      if tool_call["function"]["name"] == "get_current_weather":
        args = tool_call["function"]["arguments"]
        weather_data = get_current_weather(**args)
        
        # Add tool response to messages
        messages.append({
          "role": "tool",
          "content": json.dumps(weather_data),
          "name": tool_call["function"]["name"]
        })
    
    # Get final response with weather data
    response = chat_completion(messages)
    print(f"Final response: {response}")
    messages.append({
      "role": "assistant",
      "content": response["choices"][0]["message"]["content"]
    })
  
  # Print full conversation
  for msg in messages:
    print(f"\n{msg['role'].upper()}: {msg['content']}")

if __name__ == "__main__":
  main()
