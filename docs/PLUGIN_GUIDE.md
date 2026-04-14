# Creating Exo Inference Engine Plugins

This guide explains how to create third-party inference engine plugins for exo.

## Overview

Exo supports plugin inference engines via Python entry points. This allows you to:
- Create separate pip-installable packages for custom inference backends
- Add support for new hardware (NPUs, TPUs, custom accelerators)
- Integrate alternative inference frameworks (ONNX, TensorRT, etc.)

## Quick Start

### 1. Create Your Package Structure

```
exo-myengine/
├── pyproject.toml
├── exo_myengine/
│   ├── __init__.py
│   ├── engine.py          # Your InferenceEngine implementation
│   ├── models.py          # Model definitions (optional)
│   └── detection.py       # Device detection (optional)
```

### 2. Implement the InferenceEngine Interface

```python
# exo_myengine/engine.py
import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader

class MyInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.tokenizer = None
        self.model = None

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Tokenize a prompt string to token IDs."""
        # Load tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = await self._load_tokenizer(shard)
        tokens = self.tokenizer.encode(prompt)
        return np.array(tokens, dtype=np.int32)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode token IDs to string."""
        if self.tokenizer is None:
            self.tokenizer = await self._load_tokenizer(shard)
        return self.tokenizer.decode(tokens.tolist())

    async def sample(self, x: np.ndarray) -> np.ndarray:
        """Sample next token from logits."""
        # x is either logits or token IDs depending on your engine
        if x.ndim == 1:
            # Already token IDs
            return x
        # Sample from logits
        return np.argmax(x, axis=-1)

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: dict = None
    ) -> tuple[np.ndarray, dict]:
        """Run inference on input tokens."""
        # Load model if needed
        if self.model is None:
            await self.load_checkpoint(shard, None)

        # Your inference logic here
        output = self._run_inference(input_data, inference_state)

        return output, inference_state

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load model weights."""
        # Download model if needed
        model_path = await self.shard_downloader.ensure_shard(shard)
        self.model = self._load_model(model_path)
```

### 3. Define Your Models (Optional)

```python
# exo_myengine/models.py

MY_MODELS = {
    "my-model-7b": {
        "layers": 32,
        "repo": {
            "MyInferenceEngine": "organization/my-model-7b",
        },
    },
    "my-model-13b": {
        "layers": 40,
        "repo": {
            "MyInferenceEngine": "organization/my-model-13b",
        },
    },
}

MY_PRETTY_NAMES = {
    "my-model-7b": "My Model 7B",
    "my-model-13b": "My Model 13B",
}
```

### 4. Add Device Detection (Optional)

```python
# exo_myengine/detection.py

def detect_my_device() -> bool:
    """Check if this device supports your engine."""
    # Check for hardware, drivers, libraries, etc.
    try:
        import my_hardware_sdk
        return my_hardware_sdk.is_available()
    except ImportError:
        return False
```

### 5. Register via Entry Points

```toml
# pyproject.toml
[project]
name = "exo-myengine"
version = "0.1.0"
dependencies = ["exo"]

[project.entry-points."exo.inference_engines"]
myengine = "exo_myengine.engine:MyInferenceEngine"

[project.entry-points."exo.models"]
myengine = "exo_myengine.models:MY_MODELS"

[project.entry-points."exo.device_detectors"]
mydevice = "exo_myengine.detection:detect_my_device"
```

### 6. Install and Use

```bash
# Install your plugin
pip install exo-myengine

# Use with exo
exo --inference-engine myengine
```

## Entry Point Groups

| Group | Purpose | Value Format |
|-------|---------|--------------|
| `exo.inference_engines` | Register inference engines | `module.path:ClassName` |
| `exo.models` | Add model definitions | `module.path:MODELS_DICT` |
| `exo.device_detectors` | Auto-detect hardware | `module.path:detector_func` |

## InferenceEngine Interface

### Required Methods

| Method | Description |
|--------|-------------|
| `encode(shard, prompt)` | Tokenize string to numpy array of token IDs |
| `decode(shard, tokens)` | Decode token IDs to string |
| `sample(x)` | Sample next token from logits or return token IDs |
| `infer_tensor(request_id, shard, input_data, inference_state)` | Run inference |
| `load_checkpoint(shard, path)` | Load model weights |

### Optional Methods

| Method | Description |
|--------|-------------|
| `save_checkpoint(shard, path)` | Save model checkpoint |
| `save_session(key, value)` | Store session data |
| `clear_session()` | Clear session data |
| `infer_prompt(request_id, shard, prompt, inference_state)` | High-level prompt inference (default implementation provided) |

## Model Definition Format

```python
{
    "model-name": {
        "layers": 32,              # Number of transformer layers
        "repo": {
            "MyInferenceEngine": "huggingface/repo-id",
            # Can also specify other engines
        },
    },
}
```

## Best Practices

1. **Lazy Loading**: Don't load models at import time; wait until `load_checkpoint` is called
2. **Async Operations**: Use `async/await` for I/O operations
3. **Error Handling**: Provide clear error messages for missing dependencies
4. **Metrics**: Integrate with Prometheus metrics for observability
5. **Documentation**: Include setup instructions and hardware requirements

## Example: RKLLM Plugin Structure

The RKLLM engine in this repository serves as a reference implementation:

```
exo/inference/rkllm/
├── __init__.py           # Module exports
├── rkllm_engine.py       # InferenceEngine implementation
├── rkllm_http_client.py  # Backend communication
├── models.py             # RKLLM model definitions
├── detection.py          # Rockchip NPU detection
├── metrics.py            # Prometheus metrics
└── README.md             # Documentation
```

## Testing Your Plugin

```python
# test_myengine.py
import asyncio
from exo.inference.plugin_discovery import (
    discover_inference_engines,
    load_inference_engine,
)

def test_discovery():
    engines = discover_inference_engines()
    assert "myengine" in engines

async def test_loading():
    # Mock shard_downloader for testing
    class MockDownloader:
        async def ensure_shard(self, shard):
            return "/path/to/model"

    engine = load_inference_engine("myengine", MockDownloader())
    assert engine is not None

if __name__ == "__main__":
    test_discovery()
    asyncio.run(test_loading())
    print("All tests passed!")
```

## Troubleshooting

### Plugin Not Discovered

1. Ensure package is installed: `pip list | grep exo-myengine`
2. Check entry point registration: `python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='exo.inference_engines')))"`
3. Verify import works: `python -c "from exo_myengine.engine import MyInferenceEngine"`

### Import Errors

1. Check dependencies are installed
2. Verify module path in entry point matches actual location
3. Test direct import before using entry points

### Engine Not Loading

1. Enable debug output: `DEBUG=2 exo --inference-engine myengine`
2. Check `load_inference_engine` error messages
3. Verify constructor signature matches expected `(shard_downloader)` pattern
