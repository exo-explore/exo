import aiohttp
import asyncio
import time
import json
import os

async def measure_performance(api_endpoint: str, prompt: str = "Who are you?"):
  async with aiohttp.ClientSession() as session:
    request = {
      "model": "llama-3.2-3b",
      "messages": [{"role": "user", "content": prompt}],
      "stream": True
    }

    start_time = time.time()
    first_token_time = None
    total_tokens = 0

    print(f"Sending request to {api_endpoint}...")

    async with session.post(api_endpoint, json=request) as response:
      async for line in response.content:
        if not line.strip():
          continue

        line = line.decode('utf-8')
        if line.startswith('data: '):
          line = line[6:]  # Remove 'data: ' prefix
        if line == '[DONE]':
          break

        try:
          chunk = json.loads(line)
          if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
            if first_token_time is None:
              first_token_time = time.time()
              ttft = first_token_time - start_time
              print(f"Time to first token: {ttft:.3f}s")

            total_tokens += 1

        except json.JSONDecodeError:
          continue

    end_time = time.time()
    total_time = end_time - start_time

    if total_tokens > 0:
      tps = total_tokens / total_time
      print(f"Tokens per second: {tps:.1f}")
      print(f"Total tokens generated: {total_tokens}")
      print(f"Total time: {total_time:.3f}s")
    else:
      print("No tokens were generated")

if __name__ == "__main__":
  API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:52415/v1/chat/completions")
  asyncio.run(measure_performance(API_ENDPOINT, prompt="Write an essay about life, the universe, and everything."))
