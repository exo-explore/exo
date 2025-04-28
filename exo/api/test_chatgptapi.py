import pytest
from openai import OpenAI, AsyncOpenAI
import aiohttp
import json
import re
from exo.api.response_formats import JsonSchemaResponseFormat

# Test configuration
API_BASE_URL = "http://localhost:52415/v1/"
TEST_MODEL = "llama-3.2-1b"


@pytest.fixture
def client():
  return OpenAI(
    base_url=API_BASE_URL,
    api_key="sk-1111"
  )


@pytest.fixture
def async_client():
  return AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key="sk-1111"
  )


@pytest.mark.asyncio
async def test_basic_chat_completion(client):
  """Test basic non-streaming chat completion"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Say 'Hello world'"}],
    temperature=0.0
  )

  assert response.id.startswith("chatcmpl-")
  assert response.object == "chat.completion"
  assert response.model == TEST_MODEL
  assert len(response.choices) == 1
  assert response.choices[0].finish_reason == "stop"
  assert "Hello" in response.choices[0].message.content


@pytest.mark.asyncio
async def test_streaming_chat_completion(async_client):
  """Test streaming chat completion"""
  stream = await async_client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Count to 5 separated by commas"}],
    temperature=0.0,
    stream=True
  )

  responses = []
  async for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
      responses.append(delta.content)

  full_response = "".join(responses)
  assert full_response.count(",") == 4  # "1, 2, 3, 4, 5"
  assert chunk.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_max_completion_tokens(client):
  """Test max_completion_tokens and max_tokens fallback"""
  # Test max_completion_tokens
  response1 = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Repeat 'foo bar' 10 times"}],
    temperature=0.0,
    max_completion_tokens=5
  )

  # Test max_tokens fallback
  response2 = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Repeat 'foo bar' 10 times"}],
    temperature=0.0,
    max_tokens=5  # Deprecated parameter fallback
  )

  for response in [response1, response2]:
    assert response.choices[0].finish_reason == "length"
    assert response.usage.completion_tokens <= 5


@pytest.mark.asyncio
async def test_stop_sequences(client):
  """Test stop sequence handling"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Complete this sequence directly: A B C"}],
    temperature=0.0,
    stop=["D"],
    max_completion_tokens=20
  )

  content = response.choices[0].message.content
  assert "D" not in content
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_raw_http_request():
  """Test API using raw HTTP request for basic completion"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "2+2="}],
        "temperature": 0.0
      }
    ) as resp:
      data = await resp.json()
      assert "4" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_raw_http_request_streaming():
  """Test API using raw HTTP request for streaming"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user",
                      "content": "Count to 3 from 1, separate the numbers with commas and a space only."}],
        "temperature": 0.0,
        "stream": True
      }
    ) as resp:
      data_lines = []
      async for line in resp.content:
        if line.startswith(b"data: "):
          data_lines.append(line[6:])

      # Verify last line is DONE
      assert data_lines[-1].strip() == b"[DONE]"

      # Process all but last line which is DONE
      chunks = []
      for line in data_lines[:-1]:
        chunk = json.loads(line)
        if chunk["choices"][0]["delta"].get("content"):
          chunks.append(chunk["choices"][0]["delta"]["content"])

      result = "".join(chunks)
      assert "1, 2, 3" in result


@pytest.mark.asyncio
async def test_raw_http_no_eot_id():
  """Test that responses don't include the <|eot_id|> special token. Note this token is model dependent."""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Say exactly this: <|eot_id|>"}],
        "temperature": 0.0
      }
    ) as resp:
      data = await resp.json()
      content = data["choices"][0]["message"]["content"]
      # Should return the literal text without the special token
      assert "<|eot_id|>" not in content
      # Verify the response is properly sanitized
      assert "eot_id" not in content.lower()


@pytest.mark.asyncio
async def test_stop_sequence_first_token(client):
  """Test stop sequence when it's the first generated token"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Please repeat the word supercalifragilisticexpialidocious in all lower case"}],
    temperature=0.0,
    stop=["T", "sup", "Sup"],
    max_completion_tokens=20
  )

  content = response.choices[0].message.content
  assert content == ""
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stop_sequence_first_token_streaming(async_client):
  """Test stop sequence handling when first generated token matches stop sequence (streaming)"""
  stream = await async_client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Please repeat the word supercalifragilisticexpialidocious"}],
    temperature=0.0,
    stop=["T", "sup", "Sup"],
    max_completion_tokens=20,
    stream=True
  )

  content = []
  finish_reason = None
  async for chunk in stream:
    if chunk.choices[0].delta.content:
      content.append(chunk.choices[0].delta.content)
    if chunk.choices[0].finish_reason:
      finish_reason = chunk.choices[0].finish_reason

  # Should either get empty content with stop reason,
  # or content that doesn't contain the stop sequence
  assert "".join(content) == ""
  assert finish_reason == "stop"


@pytest.mark.asyncio
async def test_json_object_response_format(client):
  """Test JSON object response format"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Return a JSON object with a 'message' key saying 'hello'"}],
    temperature=0.0,
    response_format={"type": "json_object"}
  )

  content = response.choices[0].message.content
  parsed = json.loads(content)
  assert isinstance(parsed, dict)
  assert parsed.get("message") == "hello"
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_json_schema_response_format(client):
  """Test JSON schema response format"""
  schema = {
    "type": "object",
    "properties": {
      "number": {"type": "integer"},
      "word": {"type": "string"}
    },
    "required": ["number"]
  }

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Return a JSON object with a random number between 1-10"}],
    temperature=0.0,
    response_format=JsonSchemaResponseFormat(
      type="json_schema",
      json_schema=schema
    ).model_dump()
  )

  content = response.choices[0].message.content
  parsed = json.loads(content)
  assert isinstance(parsed.get("number"), int)
  assert 1 <= parsed["number"] <= 10
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_lark_grammar_response_format(client):
  """Test Lark grammar response format"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Answer only 'Yes' or 'No'"}],
    temperature=0.0,
    response_format={
      "type": "lark_grammar",
      "lark_grammar": """
                start: "Yes" | "No"
                %import common.WS
                %ignore WS
            """
    }
  )

  content = response.choices[0].message.content.strip()
  assert content in ["Yes", "No"]
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_regex_response_format(client):
  """Test regex response format"""
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Generate a hexadecimal color code"}],
    temperature=0.0,
    response_format={
      "type": "regex",
      "regex": r"^#[0-9a-fA-F]{6}$"
    }
  )

  content = response.choices[0].message.content.strip()
  assert re.fullmatch(r"^#[0-9a-fA-F]{6}$", content)
  assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_raw_http_json_format():
  """Test JSON response format using raw HTTP request"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user",
                      "content": "Return a JSON object with key 'status' and value 'ok'"}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
      }
    ) as resp:
      data = await resp.json()
      content = data["choices"][0]["message"]["content"]
      parsed = json.loads(content)
      assert parsed.get("status") == "ok"


# Function Calling Tests
# OpenAI recommends using the 'tools' parameter instead of the deprecated 'functions' parameter
# The 'tools' interface provides more flexibility and better supports the latest OpenAI features

@pytest.mark.asyncio
async def test_basic_tool_calling(client):
  """Test basic tool calling with a single function"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
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
              "enum": ["celsius", "fahrenheit"],
              "description": "The temperature unit to use"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    temperature=0.0,
    max_completion_tokens=200
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert response.choices[0].message.tool_calls is not None
  assert len(response.choices[0].message.tool_calls) == 1
  assert response.choices[0].message.tool_calls[0].function.name == "get_weather"

  # Parse the function arguments
  args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
  assert 'San Francisco' in args.get("location")


@pytest.mark.asyncio
async def test_function_calling_with_tool_choice(client):
  """Test function calling with tool_choice parameter"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
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
              "enum": ["celsius", "fahrenheit"],
              "description": "The temperature unit to use"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]

  # Test forcing the model to call the function
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
    temperature=0.0
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) == 1
  assert response.choices[0].message.tool_calls[0].function.name == "get_weather"

  # Parse the function arguments
  args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
  assert "Tokyo" in args.get("location")

  # Test allowing the model to choose whether to call the function
  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[
      {"role": "system", "content": """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you may need to make one or more function/tool calls to achieve the purpose.
      You should only invoke the function(s) which will assist you in fulfilling the user's request, if their request does not require any function call, you should reply to them directly as a helpful assistant.
      DO NOT USE FUNCTIONS WHEN THEY WILL NOT HELP YOU ANSWER THE USER'S QUESTION.
      You MUST NOT make any assumptions about what tools you have access to or set of functions you can generate.
      You SHOULD NOT make any function calls that are not provided in the list of functions.
      You SHOULD NOT make any function calls that are not needed to answer the question.
      You should only return the function call in tools call sections."""},
      {"role": "user", "content": "Hello, how are you?"}
    ],  # Unrelated to weather
    tools=tools,
    tool_choice="auto",
    temperature=0.0
  )

  # Model should choose not to call a function for this prompt
  assert response.choices[0].finish_reason == "stop"
  assert not hasattr(response.choices[0].message, "tool_calls") or not response.choices[0].message.tool_calls


@pytest.mark.asyncio
async def test_multiple_function_calls(client):
  """Test with multiple function definitions"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_restaurant",
        "description": "Find a restaurant in a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            },
            "cuisine": {
              "type": "string",
              "description": "Type of food, e.g. Italian, Chinese"
            }
          },
          "required": ["location", "cuisine"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "Find me an Italian restaurant in New York"}],
    tools=tools,
    tool_choice="required",
    temperature=0.0
  )

  print(response)
  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) == 1
  assert response.choices[0].message.tool_calls[0].function.name == "get_restaurant"

  args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
  assert args.get("location") == "New York"
  assert args.get("cuisine") == "Italian"


@pytest.mark.asyncio
async def test_streaming_function_calls(async_client):
  """Test streaming with function calls"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]

  stream = await async_client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "What's the weather like in Chicago?"}],
    tools=tools,
    stream=True,
    temperature=0.0
  )

  function_name = None
  function_args = {}
  finish_reason = None

  async for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.tool_calls:
      tool_call = chunk.choices[0].delta.tool_calls[0]

      # Collect function name
      if hasattr(tool_call.function, 'name') and tool_call.function.name:
        function_name = tool_call.function.name

      # Collect function arguments chunks
      if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
        function_args_str = function_args.get('partial', '') + tool_call.function.arguments
        function_args['partial'] = function_args_str

    # Get the finish reason when available
    if chunk.choices and chunk.choices[0].finish_reason:
      finish_reason = chunk.choices[0].finish_reason

  assert finish_reason == "tool_calls"
  assert function_name == "get_weather"

  # Parse the accumulated arguments
  if 'partial' in function_args:
    try:
      parsed_args = json.loads(function_args['partial'])
      assert 'Chicago' in parsed_args.get("location")
    except json.JSONDecodeError:
      pytest.fail("Failed to parse function arguments JSON")


@pytest.mark.asyncio
async def test_raw_http_function_calling():
  """Test function calling using raw HTTP request"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "What's the population of Paris?"}],
        "temperature": 0.0,
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_city_data",
              "description": "Get population data for a city",
              "parameters": {
                "type": "object",
                "properties": {
                  "city": {
                    "type": "string",
                    "description": "The name of the city"
                  },
                  "country": {
                    "type": "string",
                    "description": "The country the city is in"
                  }
                },
                "required": ["city"]
              }
            }
          }
        ]
      }
    ) as resp:
      data = await resp.json()
      assert data["choices"][0]["finish_reason"] == "tool_calls"

      tool_calls = data["choices"][0]["message"]["tool_calls"]
      assert len(tool_calls) == 1
      assert tool_calls[0]["function"]["name"] == "get_city_data"

      args = json.loads(tool_calls[0]["function"]["arguments"])
      assert args.get("city") == "Paris"


@pytest.mark.asyncio
async def test_raw_http_tools_with_tool_choice():
  """Test tools with tool_choice parameter using raw HTTP request"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Hello!"}],  # Unrelated to the tool
        "temperature": 0.0,
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_current_time",
              "description": "Get the current time in a specific timezone",
              "parameters": {
                "type": "object",
                "properties": {
                  "timezone": {
                    "type": "string",
                    "description": "The timezone to get the current time for"
                  }
                },
                "required": ["timezone"]
              }
            }
          }
        ],
        "tool_choice": {
          "type": "function",
          "function": {"name": "get_current_time"}
        }
      }
    ) as resp:
      data = await resp.json()
      assert data["choices"][0]["finish_reason"] == "tool_calls"

      tool_calls = data["choices"][0]["message"]["tool_calls"]
      assert len(tool_calls) == 1
      assert tool_calls[0]["function"]["name"] == "get_current_time"

      args = json.loads(tool_calls[0]["function"]["arguments"])
      assert "timezone" in args


@pytest.mark.asyncio
async def test_multiple_tools_one_call(client):
  """Test providing multiple different tool types in a single request"""
  tools = [
    # Traditional function tool
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city name"
            }
          },
          "required": ["location"]
        }
      }
    },
    # This demonstrates the extensibility of the tools interface
    # Future OpenAI APIs might support more tool types beyond functions
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query"
            }
          },
          "required": ["query"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "How's the weather in Berlin and what are some tourist attractions there?"}],
    tools=tools,
    temperature=0.0
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) >= 1

  # Check that the appropriate tool was selected
  tool_names = [call.function.name for call in response.choices[0].message.tool_calls]
  assert "get_weather" in tool_names or "search_web" in tool_names

  # Verify arguments for the first tool call
  first_call = response.choices[0].message.tool_calls[0]
  args = json.loads(first_call.function.arguments)

  if first_call.function.name == "get_weather":
    assert args.get("location") == "Berlin"
  elif first_call.function.name == "search_web":
    assert "Berlin" in args.get("query")


@pytest.mark.asyncio
async def test_complex_tool_schema(client):
  """Test tool calling with a complex nested parameter schema"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "book_flight",
        "description": "Book a flight ticket",
        "strict": True,
        "parameters": {
          "type": "object",
          "properties": {
            "trip_type": {
              "type": "string",
              "enum": ["one_way", "round_trip"],
              "description": "Type of trip"
            },
            "departure": {
              "type": "object",
              "properties": {
                "airport": {
                  "type": "string",
                  "description": "Departure airport code"
                },
                "date": {
                  "type": "string",
                  "format": "date",
                  "description": "Departure date in YYYY-MM-DD format"
                }
              },
              "required": ["airport", "date"]
            },
            "arrival": {
              "type": "object",
              "properties": {
                "airport": {
                  "type": "string",
                  "description": "Arrival airport code"
                }
              },
              "required": ["airport"]
            },
            "return_date": {
              "type": "string",
              "format": "date",
              "description": "Return date in YYYY-MM-DD format (for round trips)"
            },
            "passengers": {
              "type": "integer",
              "minimum": 1,
              "description": "Number of passengers"
            },
            "seat_class": {
              "type": "string",
              "enum": ["economy", "premium_economy", "business", "first"],
              "description": "Seat class preference"
            }
          },
          "required": ["trip_type", "departure", "arrival", "passengers"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user",
               "content": "I need a flight from JFK to LAX on December 15, 2023 for 2 people in business class"}],
    tools=tools,
    temperature=0.0
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) == 1
  assert response.choices[0].message.tool_calls[0].function.name == "book_flight"

  # Parse the complex structured arguments
  args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
  assert args.get("trip_type") in ["one_way", "round_trip"]
  assert args.get("departure").get("airport") == "JFK"
  assert args.get("arrival").get("airport") == "LAX"
  assert args.get("passengers") == 2
  assert args.get("seat_class") == "business"


@pytest.mark.asyncio
async def test_raw_http_streaming_function_calling():
  """Test streaming function calling using raw HTTP request"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "What's the weather like in Seattle?"}],
        "temperature": 0.0,
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get the current weather in a given location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {
                    "type": "string",
                    "description": "The city name"
                  }
                },
                "required": ["location"]
              }
            }
          }
        ],
        "stream": True
      }
    ) as resp:
      data_lines = []
      async for line in resp.content:
        if line.startswith(b"data: "):
          data_lines.append(line[6:])

      # Verify last line is DONE
      assert data_lines[-1].strip() == b"[DONE]"

      # Process all but last line which is DONE
      function_name = None
      function_args_chunks = []

      for line in data_lines[:-1]:
        chunk = json.loads(line)
        if "tool_calls" in chunk["choices"][0]["delta"]:
          tool_call = chunk["choices"][0]["delta"]["tool_calls"][0]

          # Get function name
          if "function" in tool_call and "name" in tool_call["function"]:
            function_name = tool_call["function"]["name"]

          # Collect function arguments chunks
          if "function" in tool_call and "arguments" in tool_call["function"]:
            function_args_chunks.append(tool_call["function"]["arguments"])

      assert function_name == "get_weather"

      # Combine argument chunks and parse
      if function_args_chunks:
        combined_args = "".join(function_args_chunks)
        try:
          parsed_args = json.loads(combined_args)
          assert parsed_args.get("location") == "Seattle"
        except json.JSONDecodeError:
          pytest.fail("Failed to parse function arguments JSON")


@pytest.mark.asyncio
async def test_raw_http_tools_with_explicit_choice():
  """Test tools with explicit tool_choice using raw HTTP request"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Hello!"}],  # Unrelated to the tool
        "temperature": 0.0,
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_current_time",
              "description": "Get the current time in a specific timezone",
              "parameters": {
                "type": "object",
                "properties": {
                  "timezone": {
                    "type": "string",
                    "description": "The timezone to get the current time for"
                  }
                },
                "required": ["timezone"]
              }
            }
          }
        ],
        "tool_choice": {
          "type": "function",
          "function": {"name": "get_current_time"}
        }
      }
    ) as resp:
      data = await resp.json()
      assert data["choices"][0]["finish_reason"] == "tool_calls"

      tool_calls = data["choices"][0]["message"]["tool_calls"]
      assert len(tool_calls) == 1
      assert tool_calls[0]["function"]["name"] == "get_current_time"

      args = json.loads(tool_calls[0]["function"]["arguments"])
      assert "timezone" in args


@pytest.mark.asyncio
async def test_diverse_tool_types(client):
  """Test providing multiple different tool types in a single request"""
  tools = [
    # Traditional function tool
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city name"
            }
          },
          "required": ["location"]
        }
      }
    },
    # This demonstrates the extensibility of the tools interface
    # Future OpenAI APIs might support more tool types beyond functions
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query"
            }
          },
          "required": ["query"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    model=TEST_MODEL,
    messages=[{"role": "user", "content": "How's the weather in Berlin and what are some tourist attractions there?"}],
    tools=tools,
    temperature=0.0
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) >= 1

  # Check that the appropriate tool was selected
  tool_names = [call.function.name for call in response.choices[0].message.tool_calls]
  assert "get_weather" in tool_names or "search_web" in tool_names

  # Verify arguments for the first tool call
  first_call = response.choices[0].message.tool_calls[0]
  args = json.loads(first_call.function.arguments)

  if first_call.function.name == "get_weather":
    assert args.get("location") == "Berlin"
  elif first_call.function.name == "search_web":
    assert "Berlin" in args.get("query")


@pytest.mark.asyncio
async def test_parallel_tool_calls_sdk(client):
  """Test parallel tool calling using OpenAI SDK with watt format"""
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_stock_price",
        "description": "Get current stock price for a company",
        "parameters": {
          "type": "object",
          "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"}
          },
          "required": ["ticker"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_news_headlines",
        "description": "Get recent news headlines for a company",
        "parameters": {
          "type": "object",
          "properties": {
            "company": {"type": "string", "description": "Company name"},
            "limit": {"type": "integer", "description": "Number of headlines to return"}
          },
          "required": ["company"]
        }
      }
    }
  ]

  response = client.chat.completions.create(
    # model="watt-tool-ct",
    model=TEST_MODEL,
    messages=[
      {"role": "system", "content": f"""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you may need to make one or more function/tool calls to achieve the purpose.
You should only invoke the function(s) which will assist you in fulfilling the user's request, if their request does not require any function call, you should return the answer directly.
You MUST NOT make any assumptions about what tools you have access to or set of functions you can generate.
You SHOULD NOT make any function calls that are not provided in the list of functions.
You SHOULD NOT make any function calls that are not needed to answer the question.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.
{json.dumps(tools)}
"""},
      {"role": "user", "content": "What's the current stock price and recent news for Microsoft?"}
    ],
    parallel_tool_calls=True,
    tool_choice="required",
    tools=tools,
    # Only watt support parallel tool calls
    extra_body={"tool_behaviour": {"format": "watt"}},
    temperature=0.0
  )

  assert response.choices[0].finish_reason == "tool_calls"
  assert len(response.choices[0].message.tool_calls) >= 2

  # Verify at least one call for each tool
  tool_names = {call.function.name for call in response.choices[0].message.tool_calls}
  assert "get_stock_price" in tool_names
  assert "get_news_headlines" in tool_names

  # Validate arguments for each tool
  for call in response.choices[0].message.tool_calls:
    args = json.loads(call.function.arguments)
    if call.function.name == "get_stock_price":
      assert "MSFT" in args.get("ticker", "").upper()
    elif call.function.name == "get_news_headlines":
      assert "microsoft" in args.get("company", "").lower()


@pytest.mark.asyncio
async def test_parallel_tool_calls_raw_http():
  """Test parallel tool calling using raw HTTP request with watt format"""
  async with aiohttp.ClientSession() as session:
    async with session.post(
      f"{API_BASE_URL}chat/completions",
      json={
        "model": TEST_MODEL,
        "messages": [{
          "role": "user",
          "content": "Get weather in Paris and find flights from CDG to JFK"
        }],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get current weather for a location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {"type": "string"}
                },
                "required": ["location"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "search_flights",
              "description": "Search for available flights",
              "parameters": {
                "type": "object",
                "properties": {
                  "origin": {"type": "string"},
                  "destination": {"type": "string"}
                },
                "required": ["origin", "destination"]
              }
            }
          }
        ],
        "tool_choice": "required",
        "tool_behaviour": {"format": "watt"},  # Required parameter
        "temperature": 0.0,
        "parallel_tool_calls": True
      }
    ) as resp:
      data = await resp.json()
      assert data["choices"][0]["finish_reason"] == "tool_calls"

      tool_calls = data["choices"][0]["message"]["tool_calls"]
      assert len(tool_calls) >= 2

      # Verify both tools were called
      called_tools = {call["function"]["name"] for call in tool_calls}
      assert "get_weather" in called_tools
      assert "search_flights" in called_tools

      # Validate arguments
      for call in tool_calls:
        args = json.loads(call["function"]["arguments"])
        if call["function"]["name"] == "get_weather":
          assert "Paris" in args.get("location", "")
        elif call["function"]["name"] == "search_flights":
          assert args.get("origin", "").upper() == "CDG"
          assert args.get("destination", "").upper() == "JFK"
