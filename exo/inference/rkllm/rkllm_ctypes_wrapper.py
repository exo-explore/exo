"""
ctypes wrapper for RKLLM runtime library (librkllmrt.so).

Provides Python bindings to the Rockchip RKLLM inference runtime for RK3588 NPU.
"""

import ctypes
import os
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from exo.helpers import DEBUG


# RKLLM Call States
class LLMCallState:
  RKLLM_RUN_NORMAL = 0
  RKLLM_RUN_WAITING = 1
  RKLLM_RUN_FINISH = 2
  RKLLM_RUN_ERROR = 3
  RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4


# RKLLM Input Modes
class RKLLMInputMode:
  RKLLM_INPUT_PROMPT = 0
  RKLLM_INPUT_TOKEN = 1
  RKLLM_INPUT_EMBED = 2
  RKLLM_INPUT_MULTIMODAL = 3


# RKLLM Inference Modes
class RKLLMInferMode:
  RKLLM_INFER_GENERATE = 0
  RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1


# ctypes Structure Definitions
class RKLLMExtendParam(ctypes.Structure):
  _fields_ = [
    ("base_domain_id", ctypes.c_int32),
    ("reserved", ctypes.c_uint8 * 112)
  ]


class RKLLMParam(ctypes.Structure):
  _fields_ = [
    ("model_path", ctypes.c_char_p),
    ("max_context_len", ctypes.c_int32),
    ("max_new_tokens", ctypes.c_int32),
    ("top_k", ctypes.c_int32),
    ("top_p", ctypes.c_float),
    ("temperature", ctypes.c_float),
    ("repeat_penalty", ctypes.c_float),
    ("frequency_penalty", ctypes.c_float),
    ("presence_penalty", ctypes.c_float),
    ("mirostat", ctypes.c_int32),
    ("mirostat_tau", ctypes.c_float),
    ("mirostat_eta", ctypes.c_float),
    ("skip_special_token", ctypes.c_bool),
    ("is_async", ctypes.c_bool),
    ("img_start", ctypes.c_char_p),
    ("img_end", ctypes.c_char_p),
    ("img_content", ctypes.c_char_p),
    ("extend_param", RKLLMExtendParam),
  ]


class RKLLMLoraAdapter(ctypes.Structure):
  _fields_ = [
    ("lora_adapter_path", ctypes.c_char_p),
    ("lora_adapter_name", ctypes.c_char_p),
    ("scale", ctypes.c_float)
  ]


class RKLLMEmbedInput(ctypes.Structure):
  _fields_ = [
    ("embed", ctypes.POINTER(ctypes.c_float)),
    ("n_tokens", ctypes.c_size_t)
  ]


class RKLLMTokenInput(ctypes.Structure):
  _fields_ = [
    ("input_ids", ctypes.POINTER(ctypes.c_int32)),
    ("n_tokens", ctypes.c_size_t)
  ]


class RKLLMMultiModelInput(ctypes.Structure):
  _fields_ = [
    ("prompt", ctypes.c_char_p),
    ("image_embed", ctypes.POINTER(ctypes.c_float)),
    ("n_image_tokens", ctypes.c_size_t)
  ]


class RKLLMInputUnion(ctypes.Union):
  _fields_ = [
    ("prompt_input", ctypes.c_char_p),
    ("embed_input", RKLLMEmbedInput),
    ("token_input", RKLLMTokenInput),
    ("multimodal_input", RKLLMMultiModelInput)
  ]


class RKLLMInput(ctypes.Structure):
  _fields_ = [
    ("input_mode", ctypes.c_int),
    ("input_data", RKLLMInputUnion)
  ]


class RKLLMLoraParam(ctypes.Structure):
  _fields_ = [
    ("lora_adapter_name", ctypes.c_char_p)
  ]


class RKLLMPromptCacheParam(ctypes.Structure):
  _fields_ = [
    ("save_prompt_cache", ctypes.c_int),
    ("prompt_cache_path", ctypes.c_char_p)
  ]


class RKLLMInferParam(ctypes.Structure):
  _fields_ = [
    ("mode", ctypes.c_int),
    ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
    ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam))
  ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
  _fields_ = [
    ("hidden_states", ctypes.POINTER(ctypes.c_float)),
    ("embd_size", ctypes.c_int),
    ("num_tokens", ctypes.c_int)
  ]


class RKLLMResult(ctypes.Structure):
  _fields_ = [
    ("text", ctypes.c_char_p),
    ("size", ctypes.c_int),
    ("last_hidden_layer", RKLLMResultLastHiddenLayer)
  ]


# Handle type
RKLLM_Handle_t = ctypes.c_void_p

# Callback type: void (*RKLLMCallback)(RKLLMResult* result, void* userdata, int state)
RKLLMCallback = ctypes.CFUNCTYPE(
  None,
  ctypes.POINTER(RKLLMResult),
  ctypes.c_void_p,
  ctypes.c_int
)


def find_rkllm_library() -> Optional[str]:
  """Search for librkllmrt.so in common locations."""
  search_paths = [
    os.environ.get('RKLLM_LIB_PATH'),
    os.path.expanduser('~/RKLLAMA/lib/librkllmrt.so'),
    '/usr/lib/librkllmrt.so',
    '/usr/local/lib/librkllmrt.so',
    '/usr/lib/aarch64-linux-gnu/librkllmrt.so',
  ]

  for path in search_paths:
    if path and os.path.exists(path):
      return path

  return None


def load_rkllm_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
  """Load the RKLLM runtime library."""
  if lib_path is None:
    lib_path = find_rkllm_library()

  if lib_path is None:
    raise RuntimeError(
      "Could not find librkllmrt.so. "
      "Set RKLLM_LIB_PATH environment variable or install RKLLM runtime."
    )

  if DEBUG >= 2:
    print(f"Loading RKLLM library from: {lib_path}")

  return ctypes.CDLL(lib_path)


class RKLLMWrapper:
  """
  Thread-safe wrapper for RKLLM runtime with hidden state extraction support.

  This wrapper manages the lifecycle of an RKLLM model and provides methods
  for inference with optional hidden state extraction for distributed inference.
  """

  def __init__(self, model_path: str, max_context_len: int = 4096, max_new_tokens: int = 2048):
    self._lib = load_rkllm_library()
    self._handle = RKLLM_Handle_t()
    self._model_path = model_path
    self._lock = threading.Lock()

    # Inference result storage
    self._generated_text: List[str] = []
    self._hidden_states: Optional[np.ndarray] = None
    self._inference_complete = threading.Event()
    self._inference_error: Optional[str] = None

    # Create callback - must be stored as instance variable to prevent GC
    self._callback = RKLLMCallback(self._result_callback)

    # Initialize model
    self._init_model(max_context_len, max_new_tokens)

  def _result_callback(self, result_ptr: ctypes.POINTER(RKLLMResult),
                       userdata: ctypes.c_void_p, state: int):
    """
    Callback invoked by RKLLM runtime with inference results.

    IMPORTANT: Data pointers are ephemeral and must be copied immediately.
    """
    if result_ptr:
      result = result_ptr.contents

      if state == LLMCallState.RKLLM_RUN_NORMAL:
        # Normal token generation - accumulate text
        if result.text:
          text = result.text.decode('utf-8', errors='replace')
          self._generated_text.append(text)

      elif state == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        # Hidden state extraction - copy immediately!
        hidden = result.last_hidden_layer
        if hidden.embd_size > 0 and hidden.num_tokens > 0 and hidden.hidden_states:
          size = hidden.embd_size * hidden.num_tokens
          # Create numpy array from pointer and COPY the data
          arr = np.ctypeslib.as_array(
            hidden.hidden_states,
            shape=(hidden.num_tokens, hidden.embd_size)
          )
          self._hidden_states = arr.copy()  # Copy is essential!
          if DEBUG >= 3:
            print(f"Captured hidden states: shape={self._hidden_states.shape}")

      elif state == LLMCallState.RKLLM_RUN_FINISH:
        self._inference_complete.set()

      elif state == LLMCallState.RKLLM_RUN_ERROR:
        self._inference_error = "RKLLM inference error"
        self._inference_complete.set()

  def _init_model(self, max_context_len: int, max_new_tokens: int):
    """Initialize the RKLLM model."""
    param = RKLLMParam()
    param.model_path = self._model_path.encode('utf-8')
    param.max_context_len = max_context_len
    param.max_new_tokens = max_new_tokens
    param.top_k = 40
    param.top_p = 0.9
    param.temperature = 0.8
    param.repeat_penalty = 1.1
    param.frequency_penalty = 0.0
    param.presence_penalty = 0.0
    param.mirostat = 0
    param.mirostat_tau = 5.0
    param.mirostat_eta = 0.1
    param.skip_special_token = True
    param.is_async = False
    param.img_start = None
    param.img_end = None
    param.img_content = None

    if DEBUG >= 2:
      print(f"Initializing RKLLM model: {self._model_path}")

    # rkllm_init(LLMHandle* handle, RKLLMParam* param, RKLLMCallback callback)
    self._lib.rkllm_init.argtypes = [
      ctypes.POINTER(RKLLM_Handle_t),
      ctypes.POINTER(RKLLMParam),
      RKLLMCallback
    ]
    self._lib.rkllm_init.restype = ctypes.c_int

    ret = self._lib.rkllm_init(
      ctypes.byref(self._handle),
      ctypes.byref(param),
      self._callback
    )

    if ret != 0:
      raise RuntimeError(f"Failed to initialize RKLLM model: error code {ret}")

    if DEBUG >= 1:
      print(f"RKLLM model initialized successfully")

  def _reset_state(self):
    """Reset inference state before new run."""
    self._generated_text = []
    self._hidden_states = None
    self._inference_complete.clear()
    self._inference_error = None

  def run_generate(self, prompt: str, timeout: float = 300.0) -> str:
    """
    Run text generation on prompt.

    Args:
      prompt: Input text prompt
      timeout: Maximum time to wait for completion

    Returns:
      Generated text string
    """
    with self._lock:
      self._reset_state()

      # Prepare input
      rkllm_input = RKLLMInput()
      rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
      rkllm_input.input_data.prompt_input = prompt.encode('utf-8')

      # Prepare inference params
      infer_param = RKLLMInferParam()
      infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
      infer_param.lora_params = None
      infer_param.prompt_cache_params = None

      # rkllm_run(LLMHandle handle, RKLLMInput* input, RKLLMInferParam* param, void* userdata)
      self._lib.rkllm_run.argtypes = [
        RKLLM_Handle_t,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p
      ]
      self._lib.rkllm_run.restype = ctypes.c_int

      ret = self._lib.rkllm_run(
        self._handle,
        ctypes.byref(rkllm_input),
        ctypes.byref(infer_param),
        None
      )

      if ret != 0:
        raise RuntimeError(f"RKLLM run failed: error code {ret}")

      # Wait for completion
      if not self._inference_complete.wait(timeout=timeout):
        raise TimeoutError("RKLLM inference timed out")

      if self._inference_error:
        raise RuntimeError(self._inference_error)

      return ''.join(self._generated_text)

  def run_with_hidden_state(self, prompt: str, timeout: float = 300.0) -> np.ndarray:
    """
    Run inference and extract hidden states for pipeline continuation.

    Args:
      prompt: Input text prompt
      timeout: Maximum time to wait for completion

    Returns:
      Hidden states as numpy array of shape (num_tokens, hidden_dim)
    """
    with self._lock:
      self._reset_state()

      # Prepare input
      rkllm_input = RKLLMInput()
      rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
      rkllm_input.input_data.prompt_input = prompt.encode('utf-8')

      # Prepare inference params for hidden state extraction
      infer_param = RKLLMInferParam()
      infer_param.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
      infer_param.lora_params = None
      infer_param.prompt_cache_params = None

      self._lib.rkllm_run.argtypes = [
        RKLLM_Handle_t,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p
      ]
      self._lib.rkllm_run.restype = ctypes.c_int

      ret = self._lib.rkllm_run(
        self._handle,
        ctypes.byref(rkllm_input),
        ctypes.byref(infer_param),
        None
      )

      if ret != 0:
        raise RuntimeError(f"RKLLM run failed: error code {ret}")

      # Wait for completion
      if not self._inference_complete.wait(timeout=timeout):
        raise TimeoutError("RKLLM inference timed out")

      if self._inference_error:
        raise RuntimeError(self._inference_error)

      if self._hidden_states is None:
        raise RuntimeError("No hidden states captured")

      return self._hidden_states

  def run_from_embeddings(self, embeddings: np.ndarray, timeout: float = 300.0) -> str:
    """
    Continue inference from embedding vectors (for pipeline continuation).

    Args:
      embeddings: Input embeddings of shape (num_tokens, hidden_dim)
      timeout: Maximum time to wait for completion

    Returns:
      Generated text string
    """
    with self._lock:
      self._reset_state()

      # Ensure embeddings are contiguous float32
      embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
      num_tokens, embd_size = embeddings.shape

      # Prepare embed input
      embed_input = RKLLMEmbedInput()
      embed_input.embed = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
      embed_input.n_tokens = num_tokens

      # Prepare input
      rkllm_input = RKLLMInput()
      rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_EMBED
      rkllm_input.input_data.embed_input = embed_input

      # Prepare inference params
      infer_param = RKLLMInferParam()
      infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
      infer_param.lora_params = None
      infer_param.prompt_cache_params = None

      self._lib.rkllm_run.argtypes = [
        RKLLM_Handle_t,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p
      ]
      self._lib.rkllm_run.restype = ctypes.c_int

      ret = self._lib.rkllm_run(
        self._handle,
        ctypes.byref(rkllm_input),
        ctypes.byref(infer_param),
        None
      )

      if ret != 0:
        raise RuntimeError(f"RKLLM run failed: error code {ret}")

      # Wait for completion
      if not self._inference_complete.wait(timeout=timeout):
        raise TimeoutError("RKLLM inference timed out")

      if self._inference_error:
        raise RuntimeError(self._inference_error)

      return ''.join(self._generated_text)

  def run_from_tokens(self, token_ids: np.ndarray, timeout: float = 300.0) -> str:
    """
    Run inference from token IDs.

    Args:
      token_ids: Input token IDs as numpy array
      timeout: Maximum time to wait for completion

    Returns:
      Generated text string
    """
    with self._lock:
      self._reset_state()

      # Ensure token_ids are contiguous int32
      token_ids = np.ascontiguousarray(token_ids.flatten(), dtype=np.int32)

      # Prepare token input
      token_input = RKLLMTokenInput()
      token_input.input_ids = token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
      token_input.n_tokens = len(token_ids)

      # Prepare input
      rkllm_input = RKLLMInput()
      rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN
      rkllm_input.input_data.token_input = token_input

      # Prepare inference params
      infer_param = RKLLMInferParam()
      infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
      infer_param.lora_params = None
      infer_param.prompt_cache_params = None

      self._lib.rkllm_run.argtypes = [
        RKLLM_Handle_t,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p
      ]
      self._lib.rkllm_run.restype = ctypes.c_int

      ret = self._lib.rkllm_run(
        self._handle,
        ctypes.byref(rkllm_input),
        ctypes.byref(infer_param),
        None
      )

      if ret != 0:
        raise RuntimeError(f"RKLLM run failed: error code {ret}")

      # Wait for completion
      if not self._inference_complete.wait(timeout=timeout):
        raise TimeoutError("RKLLM inference timed out")

      if self._inference_error:
        raise RuntimeError(self._inference_error)

      return ''.join(self._generated_text)

  def clear_kv_cache(self):
    """Clear the KV cache for new conversations."""
    with self._lock:
      self._lib.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t, ctypes.c_int]
      self._lib.rkllm_clear_kv_cache.restype = ctypes.c_int
      self._lib.rkllm_clear_kv_cache(self._handle, 1)

  def release(self):
    """Release model resources."""
    if self._handle:
      self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
      self._lib.rkllm_destroy.restype = ctypes.c_int
      self._lib.rkllm_destroy(self._handle)
      self._handle = None
      if DEBUG >= 2:
        print("RKLLM model released")

  def __del__(self):
    self.release()
