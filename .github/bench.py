import aiohttp
import asyncio
import time
import json
import os
import boto3
from typing import Dict, Any
from datetime import datetime


async def measure_performance(api_endpoint: str, prompt: str, model: str) -> Dict[str, Any]:
    """
    Measures the performance of an API endpoint by sending a prompt and recording metrics.

    Args:
        api_endpoint (str): The API endpoint URL.
        prompt (str): The prompt to send to the API.

    Returns:
        Dict[str, Any]: A dictionary containing performance metrics or error information.
    """

    results = {
        'model': model,
        'run_id': os.environ.get('GITHUB_RUN_ID', 'unknown'),
        'branch': os.environ.get('GITHUB_REF_NAME', 'unknown'),
        'configuration': json.loads(os.environ.get('HARDWARE_CONFIG', '{}'))
    }

    # Get token count
    session = aiohttp.ClientSession()
    try:
        response = await session.post(
            "http://localhost:52415/v1/chat/token/encode",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        token_data = await response.json()
        results['prompt_len'] = token_data['num_tokens']
    except Exception as e:
        await session.close()
        raise RuntimeError(f"Failed to get token count: {str(e)}")

    # Measure completion performance
    try:
        start_time = time.time()
        response = await session.post(
            api_endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": True
            }
        )
        response.raise_for_status()

        first_token_time = None
        total_tokens = 0

        async for line in response.content.iter_chunks():
            line = line[0].decode('utf-8').strip()
            if not line.startswith('data: '):
                continue

            data = json.loads(line[6:])  # Skip 'data: ' prefix
            if content := data.get('choices', [{}])[0].get('delta', {}).get('content'):
                print(f"Received content: {content}", flush=True)
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    results.update({
                        'ttft': ttft,
                        'prompt_tps': results['prompt_len'] / ttft
                    })
                total_tokens += 1

        total_time = time.time() - start_time
        results.update({
            'generation_tps': total_tokens / total_time,
            'response_len': total_tokens,
            'total_time': total_time
        })

    except Exception as e:
        raise RuntimeError(f"Performance measurement failed: {str(e)}")
    finally:
        await session.close()

    return results


async def main() -> None:
    api_endpoint = "http://localhost:52415/v1/chat/completions"

    # Define prompts
    prompt_warmup = "what is the capital of France?"
    prompt_essay = "write an essay about cats"

    model = os.environ.get('model', 'llama-3.2-1b')
    # Warmup request
    print("\nPerforming warmup request...", flush=True)
    try:
        warmup_results = await measure_performance(api_endpoint, prompt_warmup, model)
        print("Warmup completed successfully", flush=True)
    except Exception as e:
        print(f"Warmup request failed: {e}", flush=True)

    # Measure performance for the essay prompt
    print("\nMeasuring performance for the essay prompt...", flush=True)
    results = await measure_performance(api_endpoint, prompt_essay, model)

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
        s3_key = f"{job_name}/{model}/{now.year}/{now.month}/{now.day}/{timestamp}_{commit_sha}.json"

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