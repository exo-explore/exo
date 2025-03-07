import pytest
from openai import OpenAI, AsyncOpenAI
import aiohttp
import json

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
        "messages": [{"role": "user", "content": "Count to 3 from 1, separate the numbers with commas and a space only."}],
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
