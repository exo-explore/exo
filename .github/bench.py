import aiohttp
import asyncio
import time
import json
import os
import boto3
from typing import Dict, Any
from datetime import datetime


async def measure_performance(api_endpoint: str, prompt: str) -> Dict[str, Any]:
    """
    Measures the performance of an API endpoint by sending a prompt and recording metrics.

    Args:
        api_endpoint (str): The API endpoint URL.
        prompt (str): The prompt to send to the API.

    Returns:
        Dict[str, Any]: A dictionary containing performance metrics or error information.
    """
    model = os.environ.get('model')
    results: Dict[str, Any] = {'model': model, 'run_id': os.environ.get('GITHUB_RUN_ID')}
    results['configuration'] = json.loads(os.environ.get('HARDWARE_CONFIG'))

    # Get prompt length in tokens
    async with aiohttp.ClientSession() as session:
        try:
            request_payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            async with session.post(
                "http://localhost:52415/v1/chat/token/encode",
                json=request_payload
            ) as response:
                token_data = await response.json()
                prompt_tokens = token_data.get('num_tokens', 0)
                print(f"Prompt length: {prompt_tokens} tokens", flush=True)
        except Exception as e:
            print(f"Failed to get prompt length: {e}", flush=True)
            prompt_tokens = 0
    results['prompt_len'] = prompt_tokens

    request_payload = {
        "model": model,
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
                                results['ttft'] = first_token_time - start_time
                                results['prompt_tps'] = prompt_tokens/results['ttft']

                            total_tokens += 1
                    except json.JSONDecodeError:
                        # Log or handle malformed JSON if necessary
                        continue

            end_time = time.time()
            total_time = end_time - start_time

            if total_tokens > 0:
                results.update({
                    "generation_tps": total_tokens / total_time,
                    "response_len": total_tokens,
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
    prompt_essay = "write an essay about cats"

    # Measure performance for the essay prompt, which depends on the first measurement
    print("\nMeasuring performance for the essay prompt...", flush=True)
    results = await measure_performance(api_endpoint, prompt_essay)

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('aws_access_key_id'),
            aws_secret_access_key=os.environ.get('aws_secret_key')
        )
        job_name = os.environ.get('GITHUB_JOB')
        
        # Create S3 key with timestamp and commit info
        now = datetime.utcnow()
        timestamp = now.strftime('%H-%M-%S')
        commit_sha = os.environ.get('GITHUB_SHA', 'unknown')[:7]
        s3_key = f"{job_name}/{now.year}/{now.month}/{now.day}/{timestamp}_{commit_sha}.json"
        
        # Upload to S3
        s3_client.put_object(
            Bucket='exo-benchmarks',
            Key=s3_key,
            Body=json.dumps(results),
            ContentType='application/json'
        )
        print(f"Performance metrics uploaded to S3: s3://exo-benchmarks/{s3_key}", flush=True)
    except Exception as e:
        print(f"Failed to upload metrics to S3: {e}", flush=True)

    # Optionally print the metrics for visibility
    print("Performance metrics:", flush=True)
    print(json.dumps(results, indent=4), flush=True)


if __name__ == "__main__":
    asyncio.run(main()) 