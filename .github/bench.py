import aiohttp
import asyncio
import time
import json
import os
from typing import Dict, Any


async def measure_performance(api_endpoint: str, prompt: str) -> Dict[str, Any]:
    """
    Measures the performance of an API endpoint by sending a prompt and recording metrics.

    Args:
        api_endpoint (str): The API endpoint URL.
        prompt (str): The prompt to send to the API.

    Returns:
        Dict[str, Any]: A dictionary containing performance metrics or error information.
    """
    results: Dict[str, Any] = {}
    request_payload = {
        "model": "llama-3.2-3b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            first_token_time = None
            total_tokens = 0

            async with session.post(api_endpoint, json=request_payload) as response:
                if response.status != 200:
                    results["error"] = f"HTTP {response.status}: {response.reason}"
                    return results

                async for raw_line in response.content:
                    line = raw_line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue

                    line_content = line[6:]  # Remove 'data: ' prefix
                    if line_content == '[DONE]':
                        break

                    try:
                        chunk = json.loads(line_content)
                        choice = chunk.get('choices', [{}])[0]
                        content = choice.get('delta', {}).get('content')

                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                results["time_to_first_token"] = first_token_time - start_time

                            total_tokens += 1
                    except json.JSONDecodeError:
                        # Log or handle malformed JSON if necessary
                        continue

            end_time = time.time()
            total_time = end_time - start_time

            if total_tokens > 0:
                results.update({
                    "tokens_per_second": total_tokens / total_time,
                    "total_tokens": total_tokens,
                    "total_time": total_time
                })
            else:
                results["error"] = "No tokens were generated"

        except aiohttp.ClientError as e:
            results["error"] = f"Client error: {e}"
        except Exception as e:
            results["error"] = f"Unexpected error: {e}"

    return results


async def main() -> None:
    api_endpoint = "http://localhost:52415/v1/chat/completions"

    # Define prompts
    prompt_basic = "this is a ping"
    prompt_essay = "write an essay about cats"

    # Measure performance for the basic prompt
    print("Measuring performance for the basic prompt...")
    results_basic = await measure_performance(api_endpoint, prompt_basic)
    print("Basic prompt performance metrics:")
    print(json.dumps(results_basic, indent=4))

    # Measure performance for the essay prompt, which depends on the first measurement
    print("\nMeasuring performance for the essay prompt...")
    results = await measure_performance(api_endpoint, prompt_essay)

    # Save metrics from the "universe and everything" prompt
    metrics_file = os.path.join("artifacts", "benchmark.json")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    try:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Performance metrics saved to {metrics_file}")
    except IOError as e:
        print(f"Failed to save metrics: {e}")

    # Optionally print the metrics for visibility
    print("Performance metrics:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    asyncio.run(main()) 