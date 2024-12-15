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
    
    # Add macOS-specific checks
    try:
        # Check powermetrics with sudo
        try:
            power_metrics = subprocess.run(
                ['sudo', 'powermetrics', '-n', '1', '-i', '1000', '--samplers', 'cpu_power'],
                capture_output=True, text=True
            )
            print("\nPower Metrics:", power_metrics.stdout, flush=True)
        except Exception as e:
            print(f"Error getting power metrics: {e}", flush=True)
        
        # Check thermal state
        thermal_state = subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True)
        print("\nThermal State:", thermal_state.stdout, flush=True)
        
        # Check if running under Rosetta
        arch = subprocess.run(['arch'], capture_output=True, text=True)
        print("\nArchitecture:", arch.stdout, flush=True)
        
        # Check MLX compilation mode - only if mlx is available
        try:
            import mlx.core as mx
            if hasattr(mx, 'build_info'):
                print("\nMLX Build Info:", mx.build_info(), flush=True)
            else:
                print("\nMLX Build Info: Not available in this version", flush=True)
        except ImportError:
            print("\nMLX: Not installed", flush=True)
        except Exception as e:
            print(f"\nError checking MLX: {e}", flush=True)
        
    except Exception as e:
        print(f"Error in macOS checks: {e}", flush=True)

    # CPU Info
    print("\nCPU Information:", flush=True)
    try:
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            # Use sysctl for Apple Silicon Macs
            cpu_info = subprocess.run(['sysctl', 'machdep.cpu'], capture_output=True, text=True)
            if cpu_info.returncode == 0:
                print(f"CPU Info (Apple Silicon):", cpu_info.stdout, flush=True)
            
            # Parse powermetrics output for clearer CPU frequency display
            try:
                power_metrics = subprocess.run(
                    ['sudo', 'powermetrics', '-n', '1', '-i', '100', '--samplers', 'cpu_power'],
                    capture_output=True, text=True
                )
                if power_metrics.returncode == 0:
                    output = power_metrics.stdout
                    print("\nDetailed CPU Frequency Information:")
                    
                    # Extract cluster frequencies and max frequencies
                    current_cluster = None
                    max_freqs = {'E': 0, 'P0': 0, 'P1': 0}
                    
                    for line in output.split('\n'):
                        # Track which cluster we're processing
                        if "E-Cluster" in line:
                            current_cluster = 'E'
                        elif "P0-Cluster" in line:
                            current_cluster = 'P0'
                        elif "P1-Cluster" in line:
                            current_cluster = 'P1'
                            
                        # Get current frequencies
                        if "HW active frequency:" in line:
                            freq = line.split(':')[1].strip()
                            if freq != "0 MHz":
                                print(f"Current {current_cluster}-Cluster Frequency: {freq}")
                        
                        # Get max frequencies from residency lines
                        if current_cluster and "active residency:" in line and "MHz:" in line:
                            try:
                                # Extract all frequency values
                                freqs = []
                                parts = line.split('MHz:')[:-1]  # Skip last part as it's not a frequency
                                for part in parts:
                                    freq_str = part.split()[-1]
                                    try:
                                        freq = float(freq_str)
                                        freqs.append(freq)
                                    except ValueError:
                                        continue
                                if freqs:
                                    max_freqs[current_cluster] = max(max_freqs[current_cluster], max(freqs))
                            except Exception:
                                continue
                    
                    # Print max frequencies
                    print("\nMaximum Available Frequencies:")
                    for cluster, max_freq in max_freqs.items():
                        if max_freq > 0:
                            print(f"{cluster}-Cluster Max: {max_freq:.0f} MHz")
                            
            except Exception as e:
                print(f"Error parsing powermetrics: {e}", flush=True)
        else:
            # Use psutil for other systems
            cpu_freq = psutil.cpu_freq()
            print(f"CPU Frequency - Current: {cpu_freq.current:.2f}MHz, Min: {cpu_freq.min:.2f}MHz, Max: {cpu_freq.max:.2f}MHz", flush=True)
        
        print(f"\nCPU Usage per Core: {psutil.cpu_percent(percpu=True)}%", flush=True)
        
        # Check if running in low power mode
        power_mode = subprocess.run(['pmset', '-g'], capture_output=True, text=True)
        print("\nPower Settings:", power_mode.stdout, flush=True)
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
        # Only try to get ionice if the platform supports it
        if hasattr(current_process, 'ionice'):
            print(f"Process IO Nice Value: {current_process.ionice()}", flush=True)
    except Exception as e:
        print(f"Error getting process priority info: {e}", flush=True)

    # System Load
    print("\nSystem Load:", flush=True)
    try:
        load_avg = psutil.getloadavg()
        print(f"Load Average: {load_avg}", flush=True)
        
        # Get top processes by CPU and Memory
        print("\nTop Processes by CPU Usage:", flush=True)
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] is not None and pinfo['memory_percent'] is not None:
                    processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort and display top 5 CPU-consuming processes
        sorted_by_cpu = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5]
        for proc in sorted_by_cpu:
            print(f"PID: {proc['pid']}, Name: {proc['name']}, CPU: {proc['cpu_percent']}%, Memory: {proc['memory_percent']:.1f}%")
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
        'commit': os.environ.get('GITHUB_SHA', 'unknown'),
        'configuration': json.loads(os.environ.get('HARDWARE_CONFIG', '{}'))
    }

    # Get token count
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600, connect=10, sock_read=600, sock_connect=10))
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


def optimize_system_performance():
    """Set optimal system performance settings before running benchmark."""
    try:
        # Try to set high performance power mode
        subprocess.run(['sudo', 'pmset', '-a', 'powermode', '2'], check=False)
        
        # Ensure MLX uses performance cores and GPU
        os.environ['MLX_FORCE_P_CORES'] = '1'
        os.environ['MLX_METAL_PREWARM'] = '1'
        os.environ['MLX_USE_GPU'] = '1'
        
        # Set process priority
        current_process = psutil.Process()
        try:
            # Set highest priority
            subprocess.run(['sudo', 'renice', '-n', '-20', '-p', str(current_process.pid)], check=False)
            
            # Print current process state
            print("\nProcess State Before Benchmark:", flush=True)
            proc_info = subprocess.run(
                ['ps', '-o', 'pid,ppid,user,%cpu,%mem,nice,stat,pri,command', '-p', str(current_process.pid)],
                capture_output=True, text=True
            )
            print(proc_info.stdout, flush=True)
            
            # Verify power mode
            power_info = subprocess.run(['pmset', '-g'], capture_output=True, text=True)
            if 'powermode            0' in power_info.stdout:
                print("\nWarning: System still in normal power mode. Trying to set high performance mode again...", flush=True)
                subprocess.run(['sudo', 'pmset', '-a', 'powermode', '2'], check=False)
            
        except Exception as e:
            print(f"Warning: Could not set process priority: {e}", flush=True)
            
    except Exception as e:
        print(f"Warning: Could not optimize system performance: {e}", flush=True)
    
    # Print optimization status
    print("\nOptimization Settings:", flush=True)
    print("MLX Environment Variables:", flush=True)
    for var in ['MLX_FORCE_P_CORES', 'MLX_METAL_PREWARM', 'MLX_USE_GPU']:
        print(f"{var}: {os.environ.get(var, 'Not set')}", flush=True)
    
    try:
        nice_value = psutil.Process().nice()
        print(f"Process Nice Value: {nice_value}", flush=True)
        if nice_value != -20:
            print("Warning: Process not running at highest priority", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    check_system_state()
    check_gpu_access()
    optimize_system_performance()
    asyncio.run(main())
