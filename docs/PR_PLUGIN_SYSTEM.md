# PR Draft: Inference Engine Plugin System

**Target Repository:** [exo-explore/exo](https://github.com/exo-explore/exo)
**PR Type:** Feature Enhancement
**Status:** Draft

---

## Summary

Add a plugin system for inference engines using Python entry points, enabling third-party engines (RKLLM, llama.cpp, PyTorch, etc.) to be installed as separate packages without modifying core exo code.

## Motivation

Currently, adding a new inference engine requires:
1. Forking the repository
2. Modifying `exo/inference/inference_engine.py` (factory function)
3. Modifying `exo/models.py` (model definitions)
4. Modifying `exo/helpers.py` (device detection)
5. Maintaining a permanent fork

This creates friction for:
- **Hardware vendors** wanting to add NPU/accelerator support (Rockchip, Qualcomm, etc.)
- **Framework authors** wanting to integrate (llama.cpp, ONNX, TensorRT)
- **Researchers** experimenting with custom inference backends
- **Users** who want optional engines without bloating the core package

### Real-World Use Case: RKLLM Engine

I've implemented an RKLLM inference engine for Rockchip RK3588 NPUs. Currently, this requires maintaining a fork with ~120 lines of changes across 6 core files. With a plugin system, this could be:

```bash
pip install exo exo-rkllm  # Zero fork needed
exo --inference-engine rkllm
```

## Proposed Changes

### 1. Entry Point Discovery for Inference Engines

**File:** `exo/inference/inference_engine.py`

```python
import importlib.metadata
from typing import Optional

# Built-in engines (unchanged)
BUILTIN_ENGINES = {
    "mlx": ("exo.inference.mlx.sharded_inference_engine", "MLXDynamicShardInferenceEngine"),
    "tinygrad": ("exo.inference.tinygrad.inference", "TinygradDynamicShardInferenceEngine"),
    "dummy": ("exo.inference.dummy_inference_engine", "DummyInferenceEngine"),
}


def discover_engines() -> dict[str, str]:
    """Discover inference engines from entry points."""
    engines = dict(BUILTIN_ENGINES)

    try:
        eps = importlib.metadata.entry_points(group="exo.inference_engines")
        for ep in eps:
            engines[ep.name] = (ep.value.rsplit(":", 1)[0], ep.value.rsplit(":", 1)[1])
    except Exception:
        pass  # No plugins installed

    return engines


def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
    """Get inference engine by name, supporting both built-in and plugin engines."""
    engines = discover_engines()

    if inference_engine_name not in engines:
        available = ", ".join(sorted(engines.keys()))
        raise ValueError(f"Unknown inference engine: {inference_engine_name}. Available: {available}")

    module_path, class_name = engines[inference_engine_name]

    # Lazy import to avoid loading unused engines
    module = importlib.import_module(module_path)
    engine_class = getattr(module, class_name)

    # Handle engines that don't need shard_downloader (like dummy)
    if inference_engine_name == "dummy":
        return engine_class()

    return engine_class(shard_downloader)


def list_available_engines() -> list[str]:
    """List all available inference engines (built-in + plugins)."""
    return sorted(discover_engines().keys())
```

### 2. Entry Point Discovery for Models

**File:** `exo/models.py`

```python
import importlib.metadata

# Core model definitions (unchanged)
model_cards = {
    # ... existing MLX/tinygrad models ...
}

def discover_models() -> dict:
    """Discover additional models from plugins."""
    models = dict(model_cards)

    try:
        eps = importlib.metadata.entry_points(group="exo.models")
        for ep in eps:
            plugin_models = ep.load()
            if isinstance(plugin_models, dict):
                models.update(plugin_models)
    except Exception:
        pass

    return models


def get_model_cards() -> dict:
    """Get all model cards including plugins."""
    return discover_models()
```

### 3. Entry Point Discovery for Device Detection

**File:** `exo/helpers.py`

```python
import importlib.metadata

def get_system_info() -> str:
    """Detect system type, including plugin-provided detectors."""
    # Built-in detection
    if psutil.MACOS and platform.machine() == "arm64":
        return "Apple Silicon Mac"

    # Plugin detectors
    try:
        eps = importlib.metadata.entry_points(group="exo.device_detectors")
        for ep in eps:
            detector = ep.load()
            result = detector()
            if result:
                return result
    except Exception:
        pass

    return "Linux/Other"
```

### 4. CLI Enhancement

**File:** `exo/main.py`

```python
from exo.inference.inference_engine import list_available_engines

# Dynamic choices based on available engines
parser.add_argument(
    "--inference-engine",
    type=str,
    default=None,
    choices=list_available_engines(),
    help="Inference engine to use"
)

# Add list-engines command
parser.add_argument(
    "--list-engines",
    action="store_true",
    help="List available inference engines and exit"
)
```

## Example Plugin Package

Here's how a third-party engine (like RKLLM) would register:

**pyproject.toml:**
```toml
[project]
name = "exo-rkllm"
version = "0.1.0"
dependencies = ["exo", "aiohttp"]

[project.entry-points."exo.inference_engines"]
rkllm = "exo_rkllm.engine:RKLLMInferenceEngine"

[project.entry-points."exo.models"]
rkllm = "exo_rkllm.models:RKLLM_MODELS"

[project.entry-points."exo.device_detectors"]
rockchip = "exo_rkllm.detection:detect_rockchip_npu"
```

**exo_rkllm/engine.py:**
```python
from exo.inference.inference_engine import InferenceEngine

class RKLLMInferenceEngine(InferenceEngine):
    """Inference engine for Rockchip RK3588/RK3576 NPU."""

    async def encode(self, shard, prompt):
        # Implementation...
        pass

    async def infer_tensor(self, request_id, shard, input_data, inference_state=None):
        # Implementation...
        pass

    # ... other required methods
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **Zero-fork integration** | Third-party engines install via pip |
| **Reduced maintenance** | Core repo stays focused on MLX/tinygrad |
| **Hardware vendor support** | NPU vendors can ship official plugins |
| **Community growth** | Lower barrier for contributors |
| **Optional dependencies** | Users only install what they need |
| **Cleaner architecture** | Follows Python packaging best practices |

## Backward Compatibility

- All existing code continues to work unchanged
- Built-in engines (mlx, tinygrad, dummy) remain in core
- No breaking changes to CLI or API
- Plugin system is additive only

## Implementation Plan

1. **Phase 1:** Add `discover_engines()` and update `get_inference_engine()`
2. **Phase 2:** Add model discovery via entry points
3. **Phase 3:** Add device detector discovery
4. **Phase 4:** Update documentation for plugin authors

## Testing

```python
def test_builtin_engines_available():
    engines = list_available_engines()
    assert "mlx" in engines
    assert "tinygrad" in engines
    assert "dummy" in engines

def test_plugin_discovery(tmp_path):
    # Mock entry point for testing
    # ... test plugin loading ...
```

## Documentation Updates

Add new section to README:

```markdown
## Adding Custom Inference Engines

exo supports plugin inference engines via Python entry points.

### Installing a Plugin Engine

```bash
pip install exo-rkllm  # Example: Rockchip NPU support
exo --inference-engine rkllm
```

### Creating a Plugin Engine

See [docs/plugin-engines.md](docs/plugin-engines.md) for the complete guide.
```

## Related Work

- PyTorch uses entry points for custom backends
- Hugging Face Transformers uses entry points for tokenizers
- pytest uses entry points for plugins
- This is standard Python packaging practice (PEP 621)

## Questions for Maintainers

1. Would you prefer a single `exo.plugins` entry point group or separate groups per type?
2. Should plugin engines be listed in the TUI model selector?
3. Any concerns about lazy importing plugin modules?

---

## Files Changed

| File | Change Type | Lines |
|------|-------------|-------|
| `exo/inference/inference_engine.py` | Modified | +40 |
| `exo/models.py` | Modified | +20 |
| `exo/helpers.py` | Modified | +15 |
| `exo/main.py` | Modified | +10 |
| `docs/plugin-engines.md` | Added | +100 |
| `test/test_plugin_discovery.py` | Added | +50 |

**Total:** ~235 lines added, 0 lines removed

---

## Checklist

- [ ] Code follows project style (YAPF, 2-space indent, 200-char lines)
- [ ] Tests added for plugin discovery
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Entry points use standard `importlib.metadata`

---

*This PR enables the exo ecosystem to grow beyond the core team's bandwidth while maintaining code quality and user experience.*
