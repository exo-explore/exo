"""
Plugin discovery system for exo inference engines.

This module provides entry point-based discovery for:
- Inference engines (exo.inference_engines)
- Model definitions (exo.models)
- Device detectors (exo.device_detectors)

Third-party packages can register plugins via pyproject.toml:

  [project.entry-points."exo.inference_engines"]
  myengine = "mypackage.engine:MyInferenceEngine"

  [project.entry-points."exo.models"]
  myengine = "mypackage.models:MY_MODELS"

  [project.entry-points."exo.device_detectors"]
  mydevice = "mypackage.detection:detect_my_device"
"""

import importlib
import importlib.metadata
from typing import Dict, List, Callable, Any, Optional, Type
import os

# Debug level from environment
DEBUG = int(os.getenv("DEBUG", default="0"))

# =============================================================================
# Built-in Engine Registry
# =============================================================================

# Built-in engines with their import paths
# Format: name -> (module_path, class_name, requires_shard_downloader)
BUILTIN_ENGINES: Dict[str, tuple] = {
  "mlx": ("exo.inference.mlx.sharded_inference_engine", "MLXDynamicShardInferenceEngine", True),
  "tinygrad": ("exo.inference.tinygrad.inference", "TinygradDynamicShardInferenceEngine", True),
  "rkllm": ("exo.inference.rkllm.rkllm_engine", "RKLLMInferenceEngine", True),
  "dummy": ("exo.inference.dummy_inference_engine", "DummyInferenceEngine", False),
}

# =============================================================================
# Entry Point Discovery
# =============================================================================


def _get_entry_points(group: str) -> Dict[str, Any]:
  """
  Get entry points for a given group.

  Args:
    group: Entry point group name (e.g., 'exo.inference_engines')

  Returns:
    Dictionary mapping entry point names to their values
  """
  result = {}
  try:
    # Python 3.10+ style
    eps = importlib.metadata.entry_points(group=group)
    for ep in eps:
      result[ep.name] = ep
  except TypeError:
    # Python 3.9 fallback
    try:
      all_eps = importlib.metadata.entry_points()
      if group in all_eps:
        for ep in all_eps[group]:
          result[ep.name] = ep
    except Exception:
      pass
  except Exception as e:
    if DEBUG >= 2:
      print(f"Error discovering entry points for {group}: {e}")
  return result


def discover_inference_engines() -> Dict[str, tuple]:
  """
  Discover all available inference engines (built-in + plugins).

  Returns:
    Dictionary mapping engine names to (module_path, class_name, requires_downloader)
  """
  engines = dict(BUILTIN_ENGINES)

  # Discover plugin engines via entry points
  plugin_eps = _get_entry_points("exo.inference_engines")
  for name, ep in plugin_eps.items():
    if name not in engines:  # Don't override built-in engines
      # Entry point value format: "module.path:ClassName"
      module_path, class_name = ep.value.rsplit(":", 1)
      engines[name] = (module_path, class_name, True)  # Assume requires downloader
      if DEBUG >= 1:
        print(f"Discovered plugin engine: {name} -> {ep.value}")

  return engines


def discover_models() -> Dict[str, dict]:
  """
  Discover all model definitions from plugins.

  Returns:
    Dictionary of model definitions from plugins
  """
  models = {}

  plugin_eps = _get_entry_points("exo.models")
  for name, ep in plugin_eps.items():
    try:
      plugin_models = ep.load()
      if isinstance(plugin_models, dict):
        models.update(plugin_models)
        if DEBUG >= 1:
          print(f"Discovered {len(plugin_models)} models from plugin: {name}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error loading models from plugin {name}: {e}")

  return models


def discover_device_detectors() -> List[Callable[[], Optional[str]]]:
  """
  Discover device detector functions from plugins.

  Returns:
    List of detector functions that return device type string or None
  """
  detectors = []

  plugin_eps = _get_entry_points("exo.device_detectors")
  for name, ep in plugin_eps.items():
    try:
      detector = ep.load()
      if callable(detector):
        detectors.append(detector)
        if DEBUG >= 1:
          print(f"Discovered device detector: {name}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error loading device detector {name}: {e}")

  return detectors


# =============================================================================
# Engine Loading
# =============================================================================


def load_inference_engine(engine_name: str, shard_downloader: Any) -> Any:
  """
  Load an inference engine by name.

  Args:
    engine_name: Name of the engine (e.g., 'mlx', 'rkllm')
    shard_downloader: ShardDownloader instance for model loading

  Returns:
    Instantiated inference engine

  Raises:
    ValueError: If engine is not found
  """
  engines = discover_inference_engines()

  if engine_name not in engines:
    available = ", ".join(sorted(engines.keys()))
    raise ValueError(f"Unknown inference engine: {engine_name}. Available: {available}")

  module_path, class_name, requires_downloader = engines[engine_name]

  if DEBUG >= 2:
    print(f"Loading engine {engine_name} from {module_path}:{class_name}")

  # Handle special cases for built-in engines
  if engine_name == "tinygrad":
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

  # Import the module and get the class
  module = importlib.import_module(module_path)
  engine_class = getattr(module, class_name)

  # Instantiate with or without shard_downloader
  if requires_downloader:
    return engine_class(shard_downloader)
  else:
    return engine_class()


def list_available_engines() -> List[str]:
  """
  List all available inference engine names.

  Returns:
    Sorted list of engine names
  """
  return sorted(discover_inference_engines().keys())


def get_engine_info(engine_name: str) -> Optional[Dict[str, str]]:
  """
  Get information about an inference engine.

  Args:
    engine_name: Name of the engine

  Returns:
    Dictionary with 'module', 'class', 'type' keys, or None if not found
  """
  engines = discover_inference_engines()
  if engine_name not in engines:
    return None

  module_path, class_name, _ = engines[engine_name]
  engine_type = "builtin" if engine_name in BUILTIN_ENGINES else "plugin"

  return {
    "name": engine_name,
    "module": module_path,
    "class": class_name,
    "type": engine_type,
  }
