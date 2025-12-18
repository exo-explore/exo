# EXO Benchmark Configurations

This directory contains configuration files for the EXO staged benchmark system.

## Overview

The staged benchmark system allows you to run complex, multi-stage load tests against EXO clusters. Each stage can have different characteristics:

- **Prompt Length**: Number of tokens in the input prompt
- **Generation Length**: Maximum tokens to generate in the response
- **Time Between Requests**: Delay (in seconds) between firing consecutive requests
- **Iterations**: Number of requests to send in this stage

Requests are **fire-and-forget** - they don't wait for the previous request to complete. This allows you to test overlapping request handling and measure success rates under load.

## Configuration Files

### `bench_simple.yaml`
A minimal configuration that replicates the behavior of the original `bench.py` script:
- Single stage with 1 iteration
- Short prompt (~20 tokens)
- Generates up to 100 tokens

This is useful for quick smoke tests.

### `bench_config.yaml`
A comprehensive multi-stage benchmark with:
1. **Warmup** (10 requests): Light load with short prompts
2. **Medium Load** (20 requests): Moderate load with medium prompts
3. **Stress Test** (30 requests): Heavy overlapping requests with long prompts
4. **Cooldown** (5 requests): Light load to wind down

This tests the cluster's behavior under varying load patterns.

## Configuration Schema

```yaml
# Hardware configuration - maps runner labels to instance counts
hardware_plan:
  M3ULTRA_GPU80_512GB: 4

# Environment variables to set on each node (optional)
environment:
  OVERRIDE_MEMORY_MB: 512

# Timeout for instance and runner readiness (seconds)
timeout_seconds: 600

# Model instances to run concurrently
model_ids:
  - "mlx-community/Llama-3.2-1B-Instruct-4bit"

# Benchmark stages
stages:
  - name: "stage_name"              # Human-readable name for this stage
    prompt_length: 100               # Target prompt length in tokens
    generation_length: 200           # Max tokens to generate
    time_between_requests: 2.0       # Seconds between firing requests
    iterations: 10                   # Number of requests in this stage
```

## Running Benchmarks

### Via GitHub Actions

**Automatic (every commit):**
- The **`bench`** workflow runs automatically on every push
- Uses `bench_simple.yaml` as the default configuration
- All settings (hardware plan, timeout, environment variables, models, stages) are defined in the config file

**Manual (on-demand):**
1. Go to **Actions** → **bench** workflow
2. Click **Run workflow**
3. Configure:
   - **Config File**: Path to your YAML config (default: `.github/configs/bench_simple.yaml`)
     - `.github/configs/bench_simple.yaml` for quick tests
     - `.github/configs/bench_config.yaml` for complex multi-stage tests
   
All other settings (hardware plan, timeout, environment variables, models, stages) are read from the specified config file.

### Via Command Line

```bash
# Start EXO on localhost:8000
uv run exo --api-port 8000

# Run simple benchmark (1 stage, 1 iteration)
python3 .github/scripts/bench.py \
  --api-port 8000 \
  --config .github/configs/bench_simple.yaml \
  --expected-nodes 1 \
  --is-primary true \
  --timeout-seconds 600

# Run complex staged benchmark (4 stages, multiple iterations)
python3 .github/scripts/bench.py \
  --api-port 8000 \
  --config .github/configs/bench_config.yaml \
  --expected-nodes 1 \
  --is-primary true \
  --timeout-seconds 600
```

## Output Metrics

For each stage, the benchmark reports:

- **Total Requests**: Number of requests fired
- **Successful Requests**: Requests that completed successfully
- **Failed Requests**: Requests that encountered errors
- **Success Rate**: Percentage of successful requests
- **Total Tokens**: Sum of all tokens generated across successful requests
- **Avg Tokens/Request**: Average tokens per successful request
- **Avg Time/Request**: Average completion time per successful request

A JSON summary is also printed for easy parsing and storage.

## Creating Custom Benchmarks

To create a custom benchmark:

1. Copy an existing config file (e.g., `bench_config.yaml`)
2. Modify the stages to match your test scenario
3. Save it in this directory with a descriptive name
4. Run it using the workflow or command line

### Example: Sustained Load Test

```yaml
hardware_plan:
  M3ULTRA_GPU80_512GB: 2

environment:
  OVERRIDE_MEMORY_MB: 1024

timeout_seconds: 600

model_ids:
  - "mlx-community/Llama-3.2-1B-Instruct-4bit"

stages:
  - name: "sustained_load"
    prompt_length: 200
    generation_length: 150
    time_between_requests: 0.5     # Very fast - 2 requests/second
    iterations: 100                 # Run for ~50 seconds
```

### Example: Varying Prompt Sizes

```yaml
hardware_plan:
  M4PRO_GPU16_24GB: 3

timeout_seconds: 900

model_ids:
  - "mlx-community/Llama-3.2-1B-Instruct-4bit"

stages:
  - name: "tiny_prompts"
    prompt_length: 10
    generation_length: 100
    time_between_requests: 1.0
    iterations: 10
    
  - name: "medium_prompts"
    prompt_length: 200
    generation_length: 100
    time_between_requests: 1.0
    iterations: 10
    
  - name: "large_prompts"
    prompt_length: 1000
    generation_length: 100
    time_between_requests: 1.0
    iterations: 10
```

## Tips

- **Overlapping Requests**: Set `time_between_requests` < expected completion time to test concurrent request handling
- **Sequential Requests**: Set `time_between_requests` > expected completion time to ensure requests don't overlap
- **Realistic Load**: Model real usage patterns by varying prompt/generation lengths across stages
- **Success Rate**: A 100% success rate indicates the cluster handled the load well; lower rates suggest capacity limits

