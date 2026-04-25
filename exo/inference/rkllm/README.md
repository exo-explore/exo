# RKLLM Inference Engine

RKLLM inference engine for Rockchip RK3588/RK3576 NPU devices. This engine integrates with the [rkllama](https://github.com/jfreed-dev/rkllama) server to provide LLM inference on Rockchip NPUs.

## Performance Benchmarks

### Hardware Configuration
- **Device**: Rockchip RK3588 (6 TOPS INT8 NPU)
- **Runtime**: RKLLM SDK 1.1.4
- **Quantization**: W8A8 (8-bit weights, 8-bit activations)

### Benchmark Results

#### Qwen2.5-1.5B-Instruct (w8a8)

| Test            | Tokens/sec | Total Tokens | Completion Tokens | Response Time |
|-----------------|------------|--------------|-------------------|---------------|
| Math            | 7.23       | 56           | 23                | ~3s           |
| Explanation     | 7.81       | 57           | 29                | ~4s           |
| Code Generation | 8.21       | 189          | 156               | ~19s          |
| Reasoning       | 8.09       | 253          | 222               | ~27s          |
| Creative        | 7.48       | 54           | 26                | ~3s           |
| **AVERAGE**     | **7.76**   | **122**      | **91**            | **~11s**      |

#### DeepSeek-R1-1.5B (chain-of-thought)

| Test            | Tokens/sec | Total Tokens | Completion Tokens | Response Time |
|-----------------|------------|--------------|-------------------|---------------|
| Math            | 7.95       | 534          | 501               | ~63s          |
| Explanation     | 7.75       | 1027         | 999               | ~129s         |
| Code Generation | 7.92       | 674          | 641               | ~81s          |
| Reasoning       | 7.86       | 659          | 628               | ~80s          |
| Creative        | 8.07       | 240          | 212               | ~26s          |
| **AVERAGE**     | **7.91**   | **627**      | **596**           | **~76s**      |

### Model Comparison

| Metric                    | Qwen2.5-1.5B-Instruct | DeepSeek-R1-1.5B     |
|---------------------------|----------------------|----------------------|
| **Average Speed**         | 7.76 tok/s           | 7.91 tok/s           |
| **Avg Completion Tokens** | 91                   | 596                  |
| **Token Efficiency**      | 6.5x fewer tokens    | Verbose (CoT)        |
| **Response Style**        | Concise, direct      | Chain-of-thought     |
| **Avg Response Time**     | ~11 seconds          | ~76 seconds          |
| **Model Size**            | 1.9 GB (w8a8)        | ~1.8 GB              |
| **Best For**              | APIs, chatbots       | Reasoning, education |

### Key Findings

1. **Inference Speed**: Both models achieve ~7.8-8.0 tokens/second on the RK3588 NPU, which is consistent regardless of model architecture.

2. **Token Efficiency**: Qwen generates **6.5x fewer tokens** for equivalent prompts because DeepSeek includes `<think>` reasoning blocks in responses.

3. **Response Latency**: Due to token count differences:
   - Qwen: 3-32 seconds per response
   - DeepSeek: 26-129 seconds per response

4. **Use Case Recommendations**:
   - **Qwen2.5-1.5B-Instruct**: Production APIs, chatbots, quick Q&A, mobile applications
   - **DeepSeek-R1-1.5B**: Educational tools, step-by-step explanations, complex reasoning tasks

### Sample Responses

**Prompt: "What is 145 + 278?"**

*Qwen (23 tokens):*
```
Solution: $$ 145 + 278 = \boxed{423} $$
```

*DeepSeek (501 tokens):*
```
Calculo de la suma de 145 y 278.

Primero, sumaré los dígitos unitarios: 5 + 8 = 13...
[detailed step-by-step explanation]
...
Por lo tanto, la suma es 423.
</think>

**Solución:**
$$ 145 + 278 = \boxed{423} $$
```

### Benchmark Methodology

Tests performed with the following prompts:
1. **Math**: "What is 145 + 278?"
2. **Explanation**: "What is Python in one sentence?"
3. **Code Generation**: "Write a Python function to check if a number is prime."
4. **Reasoning**: "Is 17 a prime number? Explain."
5. **Creative**: "Write a haiku about AI."

Each test was run with `stream: false` to measure complete response generation time.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              EXO Node                                    │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │  ChatGPT    │───▶│  Orchestration   │───▶│  RKLLMInferenceEngine   │ │
│  │  API        │    │  (node.py)       │    │  (rkllm_engine.py)      │ │
│  │  :52415     │    └──────────────────┘    └───────────┬─────────────┘ │
│  └─────────────┘                                        │               │
│                                                         │ HTTP          │
│                                                         ▼               │
│                                          ┌──────────────────────────┐   │
│                                          │  RKLLMHTTPClient         │   │
│                                          │  (rkllm_http_client.py)  │   │
│                                          └───────────┬──────────────┘   │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                                                       ▼ HTTP :8080
┌─────────────────────────────────────────────────────────────────────────┐
│                           RKLLAMA Server                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │  Flask API  │───▶│  RKLLM Class     │───▶│  librkllmrt.so          │ │
│  │  /generate  │    │  (rkllm.py)      │    │  (NPU Runtime)          │ │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘ │
│                                                         │               │
│                                                         ▼               │
│                                              ┌─────────────────────┐    │
│                                              │  RK3588 NPU         │    │
│                                              │  6 TOPS INT8        │    │
│                                              └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  User    │     │  ChatGPT │     │  RKLLM       │     │  RKLLAMA     │
│  Request │     │  API     │     │  Engine      │     │  Server      │
└────┬─────┘     └────┬─────┘     └──────┬───────┘     └──────┬───────┘
     │                │                   │                    │
     │  POST /v1/chat │                   │                    │
     │  /completions  │                   │                    │
     │───────────────▶│                   │                    │
     │                │                   │                    │
     │                │  encode(prompt)   │                    │
     │                │──────────────────▶│                    │
     │                │                   │                    │
     │                │  infer_tensor     │                    │
     │                │  (first call)     │                    │
     │                │──────────────────▶│                    │
     │                │                   │                    │
     │                │                   │  POST /generate    │
     │                │                   │  (user content)    │
     │                │                   │───────────────────▶│
     │                │                   │                    │
     │                │                   │  Full response     │
     │                │                   │◀───────────────────│
     │                │                   │                    │
     │                │                   │  Cache tokens      │
     │                │                   │  Return token[0]   │
     │                │◀──────────────────│                    │
     │                │                   │                    │
     │                │  infer_tensor     │                    │
     │                │  (subsequent)     │                    │
     │                │──────────────────▶│                    │
     │                │                   │                    │
     │                │  Return token[n]  │  (from cache)      │
     │                │  from cache       │                    │
     │                │◀──────────────────│                    │
     │                │                   │                    │
     │                │       ...         │                    │
     │                │  (repeat until    │                    │
     │                │   all tokens      │                    │
     │                │   returned)       │                    │
     │                │                   │                    │
     │                │  Return EOS       │                    │
     │                │◀──────────────────│                    │
     │                │                   │                    │
     │  JSON Response │                   │                    │
     │◀───────────────│                   │                    │
     │                │                   │                    │
```

### Token Caching Mechanism

RKLLM generates complete responses in one shot, but exo expects token-by-token generation. The engine handles this mismatch:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Session Cache                                 │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  request_id: "abc-123"                                      │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │ response_tokens_abc-123: [tok1, tok2, tok3, ..., tokN]│   │   │
│   │  │ token_index_abc-123: 3  (current position)           │    │   │
│   │  └─────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   First Call:                                                        │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐       │
│   │ Full Prompt │────▶│ HTTP Call   │────▶│ Cache Response  │       │
│   │ (tokens)    │     │ to RKLLAMA  │     │ Return tok[0]   │       │
│   └─────────────┘     └─────────────┘     └─────────────────┘       │
│                                                                      │
│   Subsequent Calls:                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐       │
│   │ Single      │────▶│ Check Cache │────▶│ Return tok[n]   │       │
│   │ Token       │     │ index++     │     │ or EOS if done  │       │
│   └─────────────┘     └─────────────┘     └─────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

1. **Rockchip RK3588/RK3576 device** with NPU support
2. **RKLLM Runtime** installed (`librkllmrt.so` v1.1.4)
3. **RKLLAMA server** running on the device
4. **Converted `.rkllm` model** in `~/RKLLAMA/models/`

### Directory Structure

```
~/RKLLAMA/
├── lib/
│   ├── librkllmrt.so          # RKLLM runtime library
│   └── fix_freq_rk3588.sh     # NPU frequency optimization
└── models/
    ├── DeepSeek-R1-1.5B/
    │   ├── DeepSeek-R1-1.5B.rkllm   # Converted model
    │   └── Modelfile                 # Model configuration
    └── Qwen2.5-1.5B-Instruct/
        ├── Qwen2.5-1.5B-Instruct.rkllm
        └── Modelfile
```

### Starting RKLLAMA Server

```bash
# On the RK3588 device
cd /opt/rkllama
source venv/bin/activate
python server.py --target_platform rk3588 --port 8080
```

### Starting Exo with RKLLM

```bash
# Set environment variables (optional - defaults shown)
export RKLLM_SERVER_HOST=localhost
export RKLLM_SERVER_PORT=8080

# Start exo
python -m exo.main --inference-engine rkllm --disable-tui
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RKLLM_SERVER_HOST` | `localhost` | RKLLAMA server hostname |
| `RKLLM_SERVER_PORT` | `8080` | RKLLAMA server port |
| `DEBUG` | `0` | Debug verbosity (0-9) |

### Model Mapping

The engine maps exo model IDs to RKLLAMA model directories:

| Exo Model ID | RKLLAMA Directory | Tokenizer | Status |
|--------------|-------------------|-----------|--------|
| `deepseek-r1-1.5b-rkllm` | `DeepSeek-R1-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | Tested |
| `qwen2.5-1.5b-instruct-rkllm` | `Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Tested |
| `qwen2.5-1.5b-rkllm` | `Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | - |
| `qwen2.5-3b-rkllm` | `Qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` | - |
| `phi-3-mini-rkllm` | `Phi-3-mini` | `microsoft/Phi-3-mini-4k-instruct` | - |

## Supported Models

### Pre-converted Models (Recommended)

These models are pre-converted and tested with RKLLM 1.1.4:

| Model | HuggingFace Source | Size | Notes |
|-------|-------------------|------|-------|
| Qwen2.5-1.5B-Instruct | [c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4](https://huggingface.co/c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4) | 1.9 GB | w8a8 quantized, multiple variants |
| DeepSeek-R1-1.5B | Pre-installed with rkllama | ~1.8 GB | Chain-of-thought reasoning |

### Converting Your Own Models

Use RKLLM-Toolkit on x86_64:

```bash
# Clone toolkit
git clone --branch release-v1.1.4 https://github.com/airockchip/rknn-llm.git

# Install
pip install rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl

# Convert (see examples in rkllm-toolkit/examples/)
```

## Limitations

### No Layer Sharding

**RKLLM loads complete models only** - it cannot run partial layers. This means:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Layer Sharding (NOT SUPPORTED)                   │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  Node 1  │───▶│  Node 2  │───▶│  Node 3  │───▶│  Node 4  │     │
│   │Layer 0-6 │    │Layer 7-13│    │Layer14-20│    │Layer21-27│     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                                      │
│   ❌ This architecture does NOT work with RKLLM                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Single Node (SUPPORTED)                          │
│                                                                      │
│                        ┌──────────────────┐                         │
│                        │     Node 1       │                         │
│                        │  Full Model      │                         │
│                        │  Layers 0-27     │                         │
│                        └──────────────────┘                         │
│                                                                      │
│   ✅ Single node runs complete model                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Node Deployment

For multiple RK3588 nodes, use **request-level parallelism** (load balancer) instead of layer-level parallelism:

```
┌─────────────────────────────────────────────────────────────────────┐
│                Request-Level Parallelism (SUPPORTED)                 │
│                                                                      │
│                      ┌──────────────────┐                           │
│                      │  Load Balancer   │                           │
│                      └────────┬─────────┘                           │
│                               │                                      │
│              ┌────────────────┼────────────────┐                    │
│              │                │                │                    │
│              ▼                ▼                ▼                    │
│       ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│       │  Node 1  │     │  Node 2  │     │  Node 3  │               │
│       │Full Model│     │Full Model│     │Full Model│               │
│       │  + exo   │     │  + exo   │     │  + exo   │               │
│       └──────────┘     └──────────┘     └──────────┘               │
│                                                                      │
│   ✅ Each node handles complete requests independently               │
└─────────────────────────────────────────────────────────────────────┘
```

## API

### RKLLMInferenceEngine

```python
class RKLLMInferenceEngine(InferenceEngine):
    def __init__(
        self,
        shard_downloader: ShardDownloader,
        server_host: str = "localhost",
        server_port: int = 8080
    ):
        ...

    # Required InferenceEngine methods
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str
    async def sample(self, x: np.ndarray, temp: float, top_p: float) -> np.ndarray
    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict]
    ) -> Tuple[np.ndarray, Optional[dict]]
    async def ensure_shard(self, shard: Shard)
    async def cleanup()
```

### RKLLMHTTPClient

```python
class RKLLMHTTPClient:
    async def health_check() -> bool
    async def list_models() -> List[str]
    async def load_model(model_name: str) -> bool
    async def unload_model() -> bool
    async def generate(messages: List[Dict], stream: bool) -> str
    async def generate_from_prompt(prompt: str) -> str
```

## Troubleshooting

### Common Issues

1. **"RKLLM server not available"**
   - Ensure rkllama server is running: `curl http://localhost:8080/`
   - Check port configuration matches

2. **"No .rkllm file found"**
   - Verify model exists in `~/RKLLAMA/models/`
   - Check model directory name matches mapping

3. **"Model loaded but gibberish output"**
   - Usually indicates runtime version mismatch
   - Ensure model was converted with matching RKLLM toolkit version
   - Check: `strings librkllmrt.so | grep "version:"`

4. **"invalid rknn llm model magic"**
   - Model was converted with different RKLLM toolkit version
   - Download pre-converted model for your runtime version

5. **Segmentation fault with direct ctypes**
   - Use HTTP mode (default) instead of direct ctypes
   - The rkllama server handles NPU frequency optimization

### Debug Mode

```bash
DEBUG=2 python -m exo.main --inference-engine rkllm --disable-tui
```

This shows:
- Prompt content sent to RKLLM
- Response received
- Token caching operations

## Files

```
exo/inference/rkllm/
├── __init__.py              # Package exports
├── rkllm_engine.py          # Main RKLLMInferenceEngine class
├── rkllm_http_client.py     # Async HTTP client for rkllama
├── rkllm_ctypes_wrapper.py  # Direct ctypes bindings (fallback)
└── README.md                # This documentation
```
