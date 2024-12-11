import aiohttp
import asyncio
import time
import json
import os
import boto3
from typing import Dict, Any
from datetime import datetime
import subprocess
import psutil
import platform
from pathlib import Path


def check_system_state():
    print("\n=== System State Check ===", flush=True)
    
    # CPU Info
    print("\nCPU Information:", flush=True)
    try:
        cpu_freq = psutil.cpu_freq()
        print(f"CPU Frequency - Current: {cpu_freq.current:.2f}MHz, Min: {cpu_freq.min:.2f}MHz, Max: {cpu_freq.max:.2f}MHz", flush=True)
        print(f"CPU Usage per Core: {psutil.cpu_percent(percpu=True)}%", flush=True)
        
        # Check if running in low power mode
        power_mode = subprocess.run(['pmset', '-g'], capture_output=True, text=True)
        print("Power Settings:", power_mode.stdout, flush=True)
    except Exception as e:
        print(f"Error getting CPU info: {e}", flush=True)

    # Memory Info
    print("\nMemory Information:", flush=True)
    try:
        mem = psutil.virtual_memory()
        print(f"Total: {mem.total/1024/1024/1024:.2f}GB", flush=True)
        print(f"Available: {mem.available/1024/1024/1024:.2f}GB", flush=True)
        print(f"Used: {mem.used/1024/1024/1024:.2f}GB ({mem.percent}%)", flush=True)
        
        # Check swap
        swap = psutil.swap_memory()
        print(f"Swap Used: {swap.used/1024/1024/1024:.2f}GB of {swap.total/1024/1024/1024:.2f}GB", flush=True)
    except Exception as e:
        print(f"Error getting memory info: {e}", flush=True)

    # GPU Info
    print("\nGPU Information:", flush=True)
    try:
        # Check MLX GPU settings
        print("MLX Environment Variables:", flush=True)
        mlx_vars = {k: v for k, v in os.environ.items() if k.startswith('MLX')}
        print(json.dumps(mlx_vars, indent=2), flush=True)
        
        # Check Metal GPU memory allocation
        gpu_mem = subprocess.run(['sysctl', 'iogpu'], capture_output=True, text=True)
        print("GPU Memory Settings:", gpu_mem.stdout, flush=True)
    except Exception as e:
        print(f"Error getting GPU info: {e}", flush=True)

    # Process Priority
    print("\nProcess Priority Information:", flush=True)
    try:
        current_process = psutil.Process()
        print(f"Process Nice Value: {current_process.nice()}", flush=True)
        print(f"Process IO Nice Value: {current_process.ionice()}", flush=True)
        print(f"Process CPU Affinity: {current_process.cpu_affinity()}", flush=True)
    except Exception as e:
        print(f"Error getting process priority info: {e}", flush=True)

    # System Load
    print("\nSystem Load:", flush=True)
    try:
        print(f"Load Average: {psutil.getloadavg()}", flush=True)
        
        # Get top processes by CPU and Memory
        print("\nTop Processes:", flush=True)
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        sorted_by_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5]
        print("Top 5 CPU-consuming processes:", json.dumps(sorted_by_cpu, indent=2), flush=True)
    except Exception as e:
        print(f"Error getting system load info: {e}", flush=True)

    print("\n=== End System State Check ===\n", flush=True)


def check_gpu_access():
    try:
        # Check if MLX can see the GPU
        import mlx.core as mx
        print("MLX device info:", mx.default_device())
        
        # Check Metal device availability
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
        print("GPU Info:", result.stdout)
    except Exception as e:
        print(f"Failed to check GPU access: {e}")


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
    check_system_state()
    check_gpu_access()
    asyncio.run(main())