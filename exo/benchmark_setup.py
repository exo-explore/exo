"""
Prerequisites: Set up your local cluster
"""

from exo.main import main, shutdown
import asyncio
import signal
import aiohttp
import time
import os

# Looks like parallel conks out at 3-4 on an M2 MBP 16GB
serial = os.getenv("BENCH_SERIAL", "true").lower() == "true"
prompt_count = int(os.getenv("BENCH_PROMPT_COUNT", "5"))

prompts = [
    "Explain the theory of gravity in one sentence.",
    "What is the capital of France?",
    "Translate 'hello' to Spanish.",
    "Solve for x: 3x + 7 = 22.",
    "What is the atomic number of oxygen?",
    "Who wrote 'To Kill a Mockingbird'?",
    "Summarize the process of photosynthesis.",
    "What is 15% of 200?",
    "Name the largest planet in our solar system.",
    "Define the word 'algorithm'.",
    "What is 9 squared?",
    "Who was the first president of the United States?",
    "What is the chemical symbol for gold?",
    "Describe the function of a CPU in a computer.",
    "What year did the Titanic sink?",
    "Name the three primary colors.",
    "What is the Pythagorean theorem?",
    "Who painted the Mona Lisa?",
    "Calculate the area of a triangle with a base of 5 and height of 10.",
    "What is the boiling point of water in Celsius?",
    "Write a simple Python function to return the sum of two numbers.",
    "What is the freezing point of water in Fahrenheit?",
    "Convert 1 kilometer to meters.",
    "What is the square root of 64?",
    "Who developed the theory of relativity?",
    "How many continents are there?",
    "What is the currency of Japan?",
    "What is 8 multiplied by 7?",
    "Define the term 'machine learning'.",
    "What is the speed of light in meters per second?"
][:prompt_count]

def get_url():
  return 'http://localhost:8000/v1/chat/completions'

async def prompt_request(session: aiohttp.ClientSession, prompt: str):
  print('issuing request')
  async with session.post(get_url(), json={"messages": [{"role": "user", "content": prompt}]}, timeout=30) as response:
    print('got response')
    result = await response.json()
    print(result)
    return result

async def benchmark(serial: bool = True):
  # wait for server to start
  # TODO: make this more robust (healthcheck?)
  await asyncio.sleep(10)
  start_time = time.time()

  print("Server is running")
  prompt_token_count = 0
  completion_token_count = 0
  async with aiohttp.ClientSession() as session:
      if serial:
        print(f"Running benchmark in serial mode")
        for prompt in prompts:
          response_json = await prompt_request(session, prompt)
          prompt_token_count += response_json["usage"]["prompt_tokens"]
          completion_token_count += response_json["usage"]["completion_tokens"]
      else:
        print(f"Running benchmark in parallel mode")
        results = await asyncio.gather(*[asyncio.create_task(prompt_request(session, prompt)) for prompt in prompts], return_exceptions= True)
        for result in results:
          if isinstance(result, Exception):
            raise result
          prompt_token_count += result["usage"]["prompt_tokens"]
          completion_token_count += result["usage"]["completion_tokens"]

  print("--"*10)
  print(f"Prompt token count: {prompt_token_count}")
  print(f"Completion token count: {completion_token_count}")
  print(f"Time taken: {time.time() - start_time} seconds")
  print("--"*10)
  exit()

async def run_and_benchmark():
  print("Starting server...")
  main_task = main()
  benchmark_task = benchmark(serial=serial)
  await asyncio.gather(main_task, benchmark_task, return_exceptions=True)
  print("Server is shutting down")

if __name__ == "__main__":
  
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    loop.run_until_complete(run_and_benchmark())
  except KeyboardInterrupt:
    print("Received keyboard interrupt. Shutting down...")
  finally:
    loop.run_until_complete(shutdown(signal.SIGTERM, loop))
    loop.close()
