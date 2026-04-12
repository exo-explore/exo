"""
RKLLM Inference Engine for Rockchip RK3588 NPU.

This engine provides inference capabilities using the RKLLM runtime
for Rockchip NPU devices. It supports two modes:

1. HTTP mode (default): Connects to a running rkllama server
2. Direct mode: Uses ctypes bindings to librkllmrt.so directly

HTTP mode is recommended as the rkllama server handles frequency
optimization and proper initialization.
"""

import os
import time
import numpy as np
import asyncio
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.download.shard_download import ShardDownloader
from exo.helpers import DEBUG
from exo.api.prometheus_metrics import MODEL_INFO, INFERENCE_ENGINE_INFO
from exo.inference.rkllm.metrics import (
  RKLLM_SERVER_UP, RKLLM_INFERENCE_SECONDS, RKLLM_MODEL_LOAD_SECONDS,
  RKLLM_HTTP_REQUESTS
)

from .rkllm_http_client import RKLLMHTTPClient, RKLLMServerConfig


# Model name to RKLLAMA directory mapping
RKLLM_MODEL_MAPPING = {
  "deepseek-r1-1.5b-rkllm": "DeepSeek-R1-1.5B",
  "qwen2.5-1.5b-rkllm": "Qwen2.5-1.5B",
  "qwen2.5-1.5b-instruct-rkllm": "Qwen2.5-1.5B-Instruct",
  "qwen2.5-3b-rkllm": "Qwen2.5-3B",
  "phi-3-mini-rkllm": "Phi-3-mini",
}

# HuggingFace tokenizer repos for RKLLM models
RKLLM_TOKENIZER_REPOS = {
  "deepseek-r1-1.5b-rkllm": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "qwen2.5-1.5b-rkllm": "Qwen/Qwen2.5-1.5B-Instruct",
  "qwen2.5-1.5b-instruct-rkllm": "Qwen/Qwen2.5-1.5B-Instruct",
  "qwen2.5-3b-rkllm": "Qwen/Qwen2.5-3B-Instruct",
  "phi-3-mini-rkllm": "microsoft/Phi-3-mini-4k-instruct",
}

# Models that benefit from streaming (long chain-of-thought)
STREAMING_MODELS = {
  "deepseek-r1-1.5b-rkllm",
  "deepseek-r1-7b-rkllm",
  "deepseek-r1-14b-rkllm",
}


class RKLLMInferenceEngine(InferenceEngine):
  """
  RKLLM-based inference engine for Rockchip RK3588 NPU.

  Key characteristics:
  - Loads complete .rkllm models (no partial layer loading)
  - Uses HTTP client to connect to rkllama server (recommended)
  - Thread-safe with dedicated executor for blocking operations
  - Supports per-core NPU pinning via RKNN_CORE_MASK env var

  Core mask values for RK3588 (3 NPU cores, 6 TOPS total):
    0x1 = core 0 only (~2 TOPS)
    0x2 = core 1 only (~2 TOPS)
    0x4 = core 2 only (~2 TOPS)
    0x3 = cores 0+1 (~4 TOPS)
    0x7 = all cores (~6 TOPS, default)

  Set RKNN_CORE_MASK to pin this engine instance to specific NPU cores.
  This enables running multiple exo nodes on the same board, each on
  a different core, for distributed inference testing and multi-tenant
  workloads.
  """

  def __init__(
    self,
    shard_downloader: ShardDownloader,
    server_host: str = "localhost",
    server_port: int = 8080
  ):
    self.shard: Optional[Shard] = None
    self.shard_downloader = shard_downloader
    self._tokenizer = None
    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkllm")
    self._shard_lock = asyncio.Lock()
    self.session = {}

    # NPU core pinning: RKNN_CORE_MASK selects which NPU cores this
    # engine instance is allowed to use. When running multiple exo
    # nodes on the same RK3588 (e.g. in separate LXC containers),
    # each node should get its own core to avoid contention.
    self.core_mask = int(os.environ.get("RKNN_CORE_MASK", "0"), 0)
    self.npu_cores = self._count_cores(self.core_mask) if self.core_mask else 3

    # HTTP client configuration
    self._server_config = RKLLMServerConfig(
      host=os.environ.get("RKLLM_SERVER_HOST", server_host),
      port=int(os.environ.get("RKLLM_SERVER_PORT", server_port))
    )
    self._http_client = RKLLMHTTPClient(self._server_config)
    self._model_loaded = False
    self._use_streaming = False  # Will be set based on model
    self._stream_tasks = {}  # Active streaming tasks per request

    if DEBUG >= 1:
      core_desc = f"cores=all" if not self.core_mask else f"core_mask=0x{self.core_mask:x} ({self.npu_cores} core(s))"
      print(f"RKLLM engine initialized (HTTP mode: {self._server_config.base_url}, {core_desc})")

    # Set inference engine info metric
    INFERENCE_ENGINE_INFO.info({
      'engine': 'RKLLMInferenceEngine',
      'mode': 'http',
      'server': self._server_config.base_url,
      'core_mask': hex(self.core_mask) if self.core_mask else 'all',
      'npu_cores': str(self.npu_cores),
    })

  @staticmethod
  def _count_cores(mask: int) -> int:
    """Count set bits in the core mask."""
    count = 0
    while mask:
      count += mask & 1
      mask >>= 1
    return count

  def get_capability_descriptor(self) -> dict:
    """Return a descriptor of this engine's NPU capabilities.

    Used by the topology manager to weigh this node's capacity
    relative to other nodes in the cluster. A single-core node
    gets roughly 1/3 the weight of a full 3-core node.
    """
    return {
      "accelerator": "rk3588-npu",
      "core_mask": self.core_mask,
      "npu_cores": self.npu_cores,
      "tops_estimate": self.npu_cores * 2,  # ~2 TOPS per core
    }

  @property
  def tokenizer(self):
    """Return the tokenizer for compatibility with exo framework."""
    return self._tokenizer

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """Encode prompt to tokens using model's tokenizer."""
    await self.ensure_shard(shard)
    if self._tokenizer:
      tokens = await asyncio.get_running_loop().run_in_executor(
        self._executor,
        self._tokenizer.encode,
        prompt
      )
      return np.array(tokens)
    # Fallback: return prompt as bytes if no tokenizer
    return np.array(list(prompt.encode('utf-8')), dtype=np.int32)

  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    """
    Sample next token from logits or return pre-sampled token.

    For RKLLM, infer_tensor returns token IDs directly (shape 1,1),
    not logits. This method detects this and returns the token as-is.
    """
    # RKLLM returns token IDs directly, not logits
    # Detect this by checking if the output is a single token
    if x.ndim == 2 and x.shape == (1, 1):
      # Already a token ID, return as-is
      return x.astype(np.int32)

    # Fallback for raw logits (shouldn't happen with RKLLM HTTP mode)
    logits = x[:, -1, :] if x.ndim == 3 else x

    if temp == 0:
      return np.argmax(logits, axis=-1, keepdims=True).astype(np.int32)

    # Apply temperature
    logits = logits / max(temp, 1e-8)

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
      sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
      sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
      cumulative_probs = np.cumsum(sorted_probs, axis=-1)
      mask = cumulative_probs > top_p
      mask[:, 1:] = mask[:, :-1].copy()
      mask[:, 0] = False
      sorted_probs[mask] = 0.0
      probs = np.zeros_like(probs)
      np.put_along_axis(probs, sorted_indices, sorted_probs, axis=-1)
      probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # Sample
    sampled = np.array([
      np.random.choice(len(p), p=p) for p in probs
    ]).reshape(-1, 1)

    return sampled.astype(np.int32)

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    """Decode tokens to string."""
    await self.ensure_shard(shard)
    if self._tokenizer:
      token_list = tokens.flatten().tolist()
      return await asyncio.get_running_loop().run_in_executor(
        self._executor,
        self._tokenizer.decode,
        token_list
      )
    # Fallback: decode as UTF-8 bytes
    return bytes(tokens.flatten().tolist()).decode('utf-8', errors='replace')

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Run inference on input tensor.

    RKLLM generates complete responses in one shot, but exo expects
    token-by-token generation. This method handles the mismatch by:
    1. On first call (full prompt): generate complete response, cache it
    2. On subsequent calls: return next token from cache
    3. When cache exhausted: return EOS token

    For streaming models (DeepSeek), tokens are streamed as they're generated.

    Args:
      request_id: Unique identifier for this request
      shard: Model shard specification
      input_data: Token IDs (shape: batch, seq_len)
      inference_state: Optional state dict for continuation

    Returns:
      Tuple of (output tensor, updated inference_state)
    """
    await self.ensure_shard(shard)

    # Initialize or get inference state
    if inference_state is None:
      inference_state = {}

    # Get EOS token ID
    eos_token_id = 151643  # Default for Qwen models
    if self._tokenizer and hasattr(self._tokenizer, 'eos_token_id'):
      eos_token_id = self._tokenizer.eos_token_id or eos_token_id

    # Check if we have cached response tokens for this request
    cache_key = f"response_tokens_{request_id}"
    index_key = f"token_index_{request_id}"
    stream_key = f"stream_task_{request_id}"
    stream_queue_key = f"stream_queue_{request_id}"
    stream_done_key = f"stream_done_{request_id}"

    # Handle streaming mode for DeepSeek models
    if self._use_streaming:
      return await self._infer_tensor_streaming(
        request_id, shard, input_data, inference_state,
        eos_token_id, cache_key, index_key, stream_key,
        stream_queue_key, stream_done_key
      )

    # Non-streaming mode (original behavior)
    if cache_key in self.session:
      # Continuation: return next token from cache
      cached_tokens = self.session[cache_key]
      token_index = self.session.get(index_key, 0)

      if token_index < len(cached_tokens):
        # Return next token
        next_token = cached_tokens[token_index]
        self.session[index_key] = token_index + 1

        if DEBUG >= 2:
          print(f"RKLLM returning cached token {token_index}/{len(cached_tokens)}: {next_token}")

        # Return as logits-like array (token ID)
        output = np.array([[next_token]], dtype=np.int32)
        return output, inference_state
      else:
        # All tokens returned, signal end with EOS
        if DEBUG >= 2:
          print(f"RKLLM generation complete, returning EOS")

        # Clean up session
        del self.session[cache_key]
        if index_key in self.session:
          del self.session[index_key]

        output = np.array([[eos_token_id]], dtype=np.int32)
        return output, inference_state

    # First call: generate complete response
    if DEBUG >= 2:
      print(f"RKLLM infer_tensor: request_id={request_id}, "
            f"input_shape={input_data.shape}, dtype={input_data.dtype}")

    # Decode input tokens to text
    if input_data.dtype in [np.int32, np.int64]:
      if self._tokenizer:
        prompt = self._tokenizer.decode(input_data.flatten().tolist())
      else:
        prompt = bytes(input_data.flatten().tolist()).decode('utf-8', errors='replace')
    else:
      raise ValueError(
        "RKLLM HTTP mode only supports token input, not hidden states."
      )

    if DEBUG >= 2:
      print(f"RKLLM prompt: {prompt[:100]}...")

    # Generate via HTTP API with timing
    infer_start = time.time()
    result_text = await self._http_client.generate_from_prompt(prompt)
    RKLLM_INFERENCE_SECONDS.observe(time.time() - infer_start)
    RKLLM_HTTP_REQUESTS.labels(endpoint='/generate', status='success' if result_text else 'error').inc()

    if DEBUG >= 2:
      print(f"RKLLM response: {result_text[:200]}...")

    # Encode result to tokens and cache
    if self._tokenizer and result_text:
      result_tokens = await asyncio.get_running_loop().run_in_executor(
        self._executor,
        self._tokenizer.encode,
        result_text
      )
      # Cache the tokens for subsequent calls
      self.session[cache_key] = result_tokens
      self.session[index_key] = 1  # Start from second token

      if len(result_tokens) > 0:
        # Return first token
        output = np.array([[result_tokens[0]]], dtype=np.int32)
        if DEBUG >= 2:
          print(f"RKLLM cached {len(result_tokens)} tokens, returning first: {result_tokens[0]}")
        return output, inference_state

    # No tokens generated, return EOS
    output = np.array([[eos_token_id]], dtype=np.int32)
    return output, inference_state

  async def _infer_tensor_streaming(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: dict,
    eos_token_id: int,
    cache_key: str,
    index_key: str,
    stream_key: str,
    stream_queue_key: str,
    stream_done_key: str
  ) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Streaming inference for models like DeepSeek that generate long responses.

    Tokens are streamed from the server and returned as they become available.
    """
    # Check if we have tokens waiting in the queue
    if stream_queue_key in self.session:
      queue = self.session[stream_queue_key]
      is_done = self.session.get(stream_done_key, False)

      # Try to get a token from the queue
      if queue:
        token_text = queue.pop(0)

        # Encode the token text
        if self._tokenizer:
          tokens = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._tokenizer.encode,
            token_text
          )
          if tokens:
            if DEBUG >= 2:
              print(f"RKLLM stream token: {repr(token_text)} -> {tokens[0]}")
            output = np.array([[tokens[0]]], dtype=np.int32)
            return output, inference_state

      # Queue is empty
      if is_done:
        # Stream finished, cleanup and return EOS
        if DEBUG >= 2:
          print(f"RKLLM stream complete, returning EOS")
        self._cleanup_stream_session(request_id)
        output = np.array([[eos_token_id]], dtype=np.int32)
        return output, inference_state
      else:
        # Wait for more tokens
        await asyncio.sleep(0.01)
        # Return a placeholder token to keep the loop going
        # This will be called again to get the actual token
        if queue:
          token_text = queue.pop(0)
          if self._tokenizer:
            tokens = await asyncio.get_running_loop().run_in_executor(
              self._executor,
              self._tokenizer.encode,
              token_text
            )
            if tokens:
              output = np.array([[tokens[0]]], dtype=np.int32)
              return output, inference_state

        # Still waiting, return a space token to keep things moving
        output = np.array([[eos_token_id]], dtype=np.int32)
        return output, inference_state

    # First call: start streaming
    if DEBUG >= 1:
      print(f"RKLLM starting streaming inference for request {request_id}")

    # Decode input tokens to text
    if input_data.dtype in [np.int32, np.int64]:
      if self._tokenizer:
        prompt = self._tokenizer.decode(input_data.flatten().tolist())
      else:
        prompt = bytes(input_data.flatten().tolist()).decode('utf-8', errors='replace')
    else:
      raise ValueError("RKLLM HTTP mode only supports token input")

    if DEBUG >= 2:
      print(f"RKLLM stream prompt: {prompt[:100]}...")

    # Initialize the token queue
    self.session[stream_queue_key] = []
    self.session[stream_done_key] = False

    # Start the streaming task
    async def stream_tokens():
      try:
        async for token_text, is_finished in self._http_client.generate_from_prompt_stream(prompt):
          if stream_queue_key in self.session:
            self.session[stream_queue_key].append(token_text)
            if DEBUG >= 3:
              print(f"RKLLM queued token: {repr(token_text)}")
          if is_finished:
            break
      except Exception as e:
        if DEBUG >= 1:
          print(f"RKLLM stream error: {e}")
      finally:
        if stream_done_key in self.session:
          self.session[stream_done_key] = True

    # Start the streaming task in the background
    task = asyncio.create_task(stream_tokens())
    self._stream_tasks[request_id] = task

    # Wait for the first token with timeout
    max_wait = 60  # 60 second timeout for first token
    wait_interval = 0.1
    total_wait = 0

    while total_wait < max_wait:
      await asyncio.sleep(wait_interval)
      total_wait += wait_interval

      # Check if we got a token
      if self.session.get(stream_queue_key):
        token_text = self.session[stream_queue_key].pop(0)
        if self._tokenizer:
          tokens = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._tokenizer.encode,
            token_text
          )
          if tokens:
            if DEBUG >= 2:
              print(f"RKLLM first stream token: {repr(token_text)} -> {tokens[0]}")
            output = np.array([[tokens[0]]], dtype=np.int32)
            return output, inference_state

      # Check if streaming is done (error or completion)
      if self.session.get(stream_done_key, False):
        if DEBUG >= 1:
          print(f"RKLLM stream finished before producing tokens")
        break

    # Stream finished or timed out without tokens
    self._cleanup_stream_session(request_id)
    output = np.array([[eos_token_id]], dtype=np.int32)
    return output, inference_state

  def _cleanup_stream_session(self, request_id: str):
    """Clean up streaming session state."""
    keys_to_remove = [
      f"stream_queue_{request_id}",
      f"stream_done_{request_id}",
      f"response_tokens_{request_id}",
      f"token_index_{request_id}",
    ]
    for key in keys_to_remove:
      if key in self.session:
        del self.session[key]

    if request_id in self._stream_tasks:
      task = self._stream_tasks[request_id]
      if not task.done():
        task.cancel()
      del self._stream_tasks[request_id]

  async def load_checkpoint(self, shard: Shard, path: str):
    """Load model from checkpoint path via HTTP API."""
    async with self._shard_lock:
      model_name = self._get_rkllama_model_name(shard)

      if DEBUG >= 1:
        print(f"Loading RKLLM model via HTTP: {model_name}")

      success = await self._http_client.load_model(model_name)
      if success:
        self._model_loaded = True
        self.shard = shard
      else:
        raise RuntimeError(f"Failed to load RKLLM model: {model_name}")

  async def ensure_shard(self, shard: Shard):
    """
    Ensure the model for the given shard is loaded.

    Note: RKLLM loads complete models, so we normalize any partial
    shard to cover all layers.
    """
    async with self._shard_lock:
      if self.shard == shard and self._model_loaded:
        return

      # Check if server is available
      server_available = await self._http_client.health_check()
      RKLLM_SERVER_UP.set(1 if server_available else 0)
      RKLLM_HTTP_REQUESTS.labels(endpoint='/health', status='success' if server_available else 'error').inc()

      if not server_available:
        raise RuntimeError(
          f"RKLLM server not available at {self._server_config.base_url}. "
          f"Please start the rkllama server with: "
          f"python server.py --target_platform rk3588 --port {self._server_config.port}"
        )

      # RKLLM requires full model loading
      if not (shard.start_layer == 0 and shard.end_layer == shard.n_layers - 1):
        if DEBUG >= 1:
          print(f"RKLLM loads complete models. "
                f"Requested shard {shard.start_layer}-{shard.end_layer}/{shard.n_layers} "
                f"will load full model.")

      # Get RKLLAMA model name
      model_name = self._get_rkllama_model_name(shard)

      if DEBUG >= 1:
        print(f"Loading RKLLM model: {model_name}")

      # Load model via HTTP API with timing, passing core_mask if set
      load_start = time.time()
      success = await self._http_client.load_model(model_name, core_mask=self.core_mask)
      load_duration = time.time() - load_start
      RKLLM_MODEL_LOAD_SECONDS.observe(load_duration)
      RKLLM_HTTP_REQUESTS.labels(endpoint='/load_model', status='success' if success else 'error').inc()

      if not success:
        # Try listing available models for debugging
        available = await self._http_client.list_models()
        raise RuntimeError(
          f"Failed to load RKLLM model: {model_name}. "
          f"Available models: {available}"
        )

      self._model_loaded = True

      # Update model info metric
      MODEL_INFO.info({
        'model_id': shard.model_id,
        'rkllm_name': model_name,
        'layers': str(shard.n_layers)
      })

      # Load tokenizer from HuggingFace
      tokenizer_repo = RKLLM_TOKENIZER_REPOS.get(shard.model_id.lower())
      if tokenizer_repo:
        try:
          self._tokenizer = await resolve_tokenizer(tokenizer_repo)
          if DEBUG >= 1:
            print(f"Loaded tokenizer from: {tokenizer_repo}")
        except Exception as e:
          if DEBUG >= 1:
            print(f"Failed to load tokenizer from {tokenizer_repo}: {e}")
          self._tokenizer = None
      else:
        # Try to resolve from model path
        try:
          # Use shard_downloader to get model path for tokenizer
          model_path = await self.shard_downloader.ensure_shard(
            shard, self.__class__.__name__
          )
          self._tokenizer = await resolve_tokenizer(model_path)
        except Exception as e:
          if DEBUG >= 1:
            print(f"Failed to load tokenizer: {e}")
          self._tokenizer = None

      # Store normalized shard (full model)
      self.shard = Shard(
        shard.model_id,
        0,
        shard.n_layers - 1,
        shard.n_layers
      )

      self.session = {}

      # Enable streaming for models that benefit from it
      self._use_streaming = shard.model_id.lower() in STREAMING_MODELS
      if self._use_streaming and DEBUG >= 1:
        print(f"RKLLM streaming enabled for model: {shard.model_id}")

      if DEBUG >= 1:
        print(f"RKLLM model loaded: {shard.model_id}")

  def _get_rkllama_model_name(self, shard: Shard) -> str:
    """Get RKLLAMA model directory name from shard."""
    model_id = shard.model_id.lower()

    # Check explicit mapping first
    if model_id in RKLLM_MODEL_MAPPING:
      return RKLLM_MODEL_MAPPING[model_id]

    # Try to derive from model_id
    # Remove common suffixes
    name = model_id.replace('-rkllm', '').replace('_rkllm', '')

    # Title case each part
    parts = name.replace('_', '-').split('-')
    name = '-'.join(p.capitalize() for p in parts)

    return name

  async def cleanup(self):
    """Release all resources."""
    if self._model_loaded:
      try:
        await self._http_client.unload_model()
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error unloading model: {e}")
      self._model_loaded = False

    await self._http_client.close()
    self._executor.shutdown(wait=True)

  def __del__(self):
    # Cleanup is async, so we can't do much here
    pass
