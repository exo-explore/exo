# Benchmark Workflow Usage

## Overview

The `bench_matrix.yml` workflow enables distributed benchmarking of models across multiple self-hosted macOS runners with different hardware configurations.

## Workflow Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `model_id` | Model ID to benchmark | `mlx-community/Llama-3.2-1B-Instruct-4bit` | Yes |
| `hardware_plan` | JSON mapping of runner labels to counts | `{"M4PRO_GPU16_24GB": 1}` | Yes |
| `prompt` | Benchmark prompt text | `What is the capital of France?` | No |
| `timeout_seconds` | Timeout for instance/runner readiness | `600` | No |

## Hardware Plan Format

The `hardware_plan` input is a JSON object mapping runner labels to the number of machines:

```json
{
  "M4PRO_GPU16_24GB": 2,
  "M3ULTRA_GPU80_512GB": 1
}
```

This example would:
- Start 2 runners with the `M4PRO_GPU16_24GB` label
- Start 1 runner with the `M3ULTRA_GPU80_512GB` label
- Total of 3 runners coordinating on a single distributed inference instance

## How It Works

1. **Planning Job** (`plan`)
   - Runs on `ubuntu-latest`
   - Parses the `hardware_plan` JSON
   - Generates a dynamic matrix with one entry per runner
   - Only the first runner (index 0) is marked as `is_primary`

2. **Benchmark Worker Jobs** (`bench_worker`)
   - Each job runs on a self-hosted macOS runner with the specified label
   - All runners start EXO in parallel
   - The primary runner creates the model instance
   - All runners wait for their assigned runner to be ready (Loaded/Running status)
   - The primary runner executes the benchmark and prints results
   - The primary runner deletes the instance

## Example Usage

### Single Machine Benchmark

```yaml
model_id: mlx-community/Llama-3.2-1B-Instruct-4bit
hardware_plan: '{"M4PRO_GPU16_24GB": 1}'
prompt: What is the capital of France?
timeout_seconds: 600
```

### Multi-Machine Distributed Benchmark

```yaml
model_id: mlx-community/Llama-3.2-3B-Instruct-4bit
hardware_plan: '{"M4PRO_GPU16_24GB": 2, "M3ULTRA_GPU80_512GB": 1}'
prompt: Explain quantum computing in simple terms.
timeout_seconds: 900
```

## Benchmark Output

The primary runner outputs a JSON object with benchmark results:

```json
{
  "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "instance_id": "abc-123-def",
  "tokens": 42,
  "elapsed_s": 2.451,
  "tps": 17.136
}
```

Where:
- `tokens`: Number of chunks/tokens generated
- `elapsed_s`: Total elapsed time in seconds
- `tps`: Tokens per second (tokens / elapsed_s)

## Runner Requirements

Each self-hosted runner must:
- Be labeled with appropriate hardware tags (e.g., `M4PRO_GPU16_24GB`)
- Have the `self-hosted` and `macOS` labels
- Have Nix installed with flakes enabled
- Have network connectivity to other runners in the same job

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions Workflow (bench_matrix.yml)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐                                         │
│  │  Plan Job      │                                         │
│  │  (ubuntu)      │──┬─► Matrix: [{label, index, primary}] │
│  └────────────────┘  │                                      │
│                      │                                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  Bench Worker Jobs (Matrix)                         │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │                                                       │  │
│  │  Runner 0 (Primary)     Runner 1         Runner 2    │  │
│  │  ┌─────────────┐       ┌─────────────┐ ┌──────────┐ │  │
│  │  │ Start EXO   │       │ Start EXO   │ │ Start EXO│ │  │
│  │  │ Create Inst │       │ Wait...     │ │ Wait...  │ │  │
│  │  │ Wait Ready  │       │ Wait Ready  │ │ Wait...  │ │  │
│  │  │ Run Bench   │       │ (idle)      │ │ (idle)   │ │  │
│  │  │ Print TPS   │       │             │ │          │ │  │
│  │  │ Delete Inst │       │             │ │          │ │  │
│  │  └─────────────┘       └─────────────┘ └──────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### `scripts/bench.py`

A standalone Python script that:
- Creates instance (primary only)
- Polls `/state` endpoint until instance and all runners are ready
- Executes chat completion with timing (primary only)
- Parses SSE stream and counts tokens
- Computes TPS metrics
- Cleans up instance (primary only)

### Key Functions

- `wait_for_instance()`: Polls until instance with model_id appears
- `wait_for_runners_ready()`: Polls until expected number of runners reach Loaded/Running status
- `run_benchmark()`: Executes chat completion, measures time, counts tokens

## Troubleshooting

### Instance never becomes ready
- Check EXO logs in the workflow output
- Verify model_id is valid and accessible
- Increase `timeout_seconds`

### Runner mismatch
- Ensure hardware_plan counts match available labeled runners
- Check runner labels match exactly (case-sensitive)

### Network issues
- Verify runners can communicate on the network
- Check firewall rules between runner hosts

