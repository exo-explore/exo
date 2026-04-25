"""
HTTP Client for RKLLM/RKLLama server.

Provides an async HTTP client to interact with the rkllama server
which exposes the RKLLM runtime via a Flask API.
"""

import aiohttp
import asyncio
import json
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from dataclasses import dataclass

from exo.helpers import DEBUG


@dataclass
class RKLLMServerConfig:
  """Configuration for connecting to rkllama server."""
  host: str = "localhost"
  port: int = 8080
  timeout: float = 300.0  # 5 minute timeout for generation
  ollama_compat: bool = True  # Use Ollama-compatible API (/api/generate)

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"


class RKLLMHTTPClient:
  """
  Async HTTP client for rkllama server.

  The rkllama server handles:
  - Model loading/unloading
  - Tokenization (via HuggingFace tokenizers)
  - Inference on the RKLLM runtime
  """

  def __init__(self, config: Optional[RKLLMServerConfig] = None):
    self.config = config or RKLLMServerConfig()
    self._session: Optional[aiohttp.ClientSession] = None
    self._current_model: Optional[str] = None

  async def _get_session(self) -> aiohttp.ClientSession:
    """Get or create aiohttp session."""
    if self._session is None or self._session.closed:
      timeout = aiohttp.ClientTimeout(total=self.config.timeout)
      self._session = aiohttp.ClientSession(timeout=timeout)
    return self._session

  async def close(self):
    """Close the HTTP session."""
    if self._session and not self._session.closed:
      await self._session.close()
      self._session = None

  async def health_check(self) -> bool:
    """Check if the rkllama server is running."""
    try:
      session = await self._get_session()
      async with session.get(f"{self.config.base_url}/") as resp:
        return resp.status == 200
    except aiohttp.ClientError:
      return False
    except Exception as e:
      if DEBUG >= 2:
        print(f"RKLLM health check failed: {e}")
      return False

  async def list_models(self) -> List[str]:
    """Get list of available models on the server."""
    try:
      session = await self._get_session()
      if self.config.ollama_compat:
        async with session.get(f"{self.config.base_url}/api/tags") as resp:
          if resp.status == 200:
            data = await resp.json()
            return [m.get("name", m.get("model", "")) for m in data.get("models", [])]
          return []
      async with session.get(f"{self.config.base_url}/models") as resp:
        if resp.status == 200:
          data = await resp.json()
          return data.get("models", [])
        return []
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to list RKLLM models: {e}")
      return []

  async def get_current_model(self) -> Optional[str]:
    """Get the currently loaded model name."""
    if self.config.ollama_compat:
      return self._current_model
    try:
      session = await self._get_session()
      async with session.get(f"{self.config.base_url}/current_model") as resp:
        if resp.status == 200:
          data = await resp.json()
          return data.get("model_name")
        return None
    except Exception as e:
      if DEBUG >= 2:
        print(f"Failed to get current model: {e}")
      return None

  async def load_model(
    self,
    model_name: str,
    huggingface_path: Optional[str] = None,
    from_file: Optional[str] = None,
    core_mask: int = 0
  ) -> bool:
    """
    Load a model on the rkllama server.

    Args:
      model_name: Name of the model directory in ~/RKLLAMA/models/
      huggingface_path: Optional HuggingFace repo for tokenizer
      core_mask: NPU core mask (0 = all cores, 0x1/0x2/0x4 = single core)
      from_file: Optional .rkllm filename

    Returns:
      True if model loaded successfully
    """
    # Check if model is already loaded
    current = await self.get_current_model()
    if current == model_name:
      if DEBUG >= 2:
        print(f"Model {model_name} already loaded")
      return True

    if self.config.ollama_compat:
      # Ollama-compat rkllama loads models on first use via /api/generate.
      # Just verify the model exists in the list, then mark it as current.
      available = await self.list_models()
      if model_name in available:
        self._current_model = model_name
        if DEBUG >= 1:
          print(f"RKLLM model {model_name} available (Ollama compat, lazy load)")
        return True
      if DEBUG >= 1:
        print(f"RKLLM model {model_name} not in available list: {available}")
      return False

    # Unload current model if one is loaded
    if current:
      await self.unload_model()

    try:
      session = await self._get_session()
      payload: Dict[str, Any] = {"model_name": model_name}

      if huggingface_path:
        payload["huggingface_path"] = huggingface_path
      if from_file:
        payload["from"] = from_file
      if core_mask:
        payload["core_mask"] = core_mask

      async with session.post(
        f"{self.config.base_url}/load_model",
        json=payload
      ) as resp:
        if resp.status == 200:
          self._current_model = model_name
          if DEBUG >= 1:
            print(f"RKLLM model {model_name} loaded successfully")
          return True
        else:
          error = await resp.json()
          if DEBUG >= 1:
            print(f"Failed to load model: {error}")
          return False
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to load RKLLM model: {e}")
      return False

  async def unload_model(self) -> bool:
    """Unload the current model."""
    try:
      session = await self._get_session()
      async with session.post(f"{self.config.base_url}/unload_model") as resp:
        if resp.status == 200:
          self._current_model = None
          return True
        return False
    except Exception as e:
      if DEBUG >= 2:
        print(f"Failed to unload model: {e}")
      return False

  async def generate(
    self,
    messages: List[Dict[str, str]],
    stream: bool = False
  ) -> str:
    """
    Generate text from messages.

    Args:
      messages: List of message dicts with 'role' and 'content' keys
                e.g., [{"role": "user", "content": "Hello"}]
      stream: Whether to stream the response

    Returns:
      Generated text response
    """
    try:
      session = await self._get_session()
      payload = {
        "messages": messages,
        "stream": stream
      }

      async with session.post(
        f"{self.config.base_url}/generate",
        json=payload
      ) as resp:
        if resp.status == 200:
          if stream:
            # Handle streaming response
            full_text = ""
            async for line in resp.content:
              if line:
                try:
                  import json
                  data = json.loads(line.decode('utf-8').strip())
                  if data.get("choices"):
                    content = data["choices"][0].get("content", "")
                    full_text += content
                except (json.JSONDecodeError, KeyError):
                  continue
            return full_text
          else:
            data = await resp.json()
            if data.get("choices"):
              return data["choices"][0].get("content", "")
            return ""
        else:
          error = await resp.text()
          if DEBUG >= 1:
            print(f"Generate failed: {error}")
          return ""
    except asyncio.TimeoutError:
      if DEBUG >= 1:
        print("RKLLM generate timed out")
      return ""
    except Exception as e:
      if DEBUG >= 1:
        print(f"RKLLM generate failed: {e}")
      return ""

  async def generate_from_prompt(self, prompt: str) -> str:
    """
    Generate from a prompt string, handling pre-templated prompts.

    If the prompt contains chat template markers (e.g., from exo),
    extract the user content to avoid double-templating.

    Args:
      prompt: The user prompt text (may be pre-templated)

    Returns:
      Generated text response
    """
    import re

    # Check if prompt is already templated (exo uses special tokens)
    # Common patterns: <｜User｜>, <|user|>, [INST], etc.
    user_markers = [
      (r'<｜User｜>(.*?)<｜Assistant｜>', 1),
      (r'<\|user\|>(.*?)<\|assistant\|>', 1),
      (r'\[INST\](.*?)\[/INST\]', 1),
      (r'<\|im_start\|>user\n(.*?)<\|im_end\|>', 1),
    ]

    extracted_content = None
    for pattern, group in user_markers:
      match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
      if match:
        extracted_content = match.group(group).strip()
        break

    if extracted_content:
      # Use extracted content to avoid double-templating
      if DEBUG >= 2:
        print(f"Extracted user content from template: {extracted_content[:100]}...")
      messages = [{"role": "user", "content": extracted_content}]
    else:
      # Use prompt as-is
      messages = [{"role": "user", "content": prompt}]

    if self.config.ollama_compat:
      return await self._generate_ollama(prompt if not extracted_content else extracted_content)
    return await self.generate(messages, stream=False)

  async def _generate_ollama(self, prompt: str) -> str:
    """Generate via the Ollama-compatible /api/generate endpoint."""
    try:
      session = await self._get_session()
      payload = {
        "model": self._current_model or "",
        "prompt": prompt,
        "stream": False,
      }
      async with session.post(
        f"{self.config.base_url}/api/generate",
        json=payload
      ) as resp:
        if resp.status == 200:
          data = await resp.json()
          return data.get("response", "")
        else:
          error = await resp.text()
          if DEBUG >= 1:
            print(f"Ollama generate failed ({resp.status}): {error[:200]}")
          return ""
    except asyncio.TimeoutError:
      if DEBUG >= 1:
        print("Ollama generate timed out")
      return ""
    except Exception as e:
      if DEBUG >= 1:
        print(f"Ollama generate failed: {e}")
      return ""

  async def generate_stream(
    self,
    messages: List[Dict[str, str]],
    callback: Optional[callable] = None
  ) -> AsyncGenerator[Tuple[str, bool], None]:
    """
    Stream tokens from the rkllama server.

    Args:
      messages: List of message dicts with 'role' and 'content' keys
      callback: Optional callback function called for each token

    Yields:
      Tuple of (token_text, is_finished) for each token
    """
    try:
      session = await self._get_session()
      payload = {
        "messages": messages,
        "stream": True
      }

      if DEBUG >= 2:
        print(f"RKLLM streaming request: {messages}")

      async with session.post(
        f"{self.config.base_url}/generate",
        json=payload
      ) as resp:
        if DEBUG >= 2:
          print(f"RKLLM stream response status: {resp.status}")

        if resp.status == 200:
          async for line in resp.content:
            if line:
              line_str = line.decode('utf-8').strip()
              if DEBUG >= 3:
                print(f"RKLLM stream raw line: {repr(line_str[:200])}")
              if not line_str:
                continue

              # Handle multiple JSON objects in one line (separated by \n\n)
              for json_str in line_str.split('\n\n'):
                json_str = json_str.strip()
                if not json_str:
                  continue

                try:
                  data = json.loads(json_str)
                  if data.get("choices"):
                    choice = data["choices"][0]
                    content = choice.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    is_finished = finish_reason == "stop"

                    if content:
                      if DEBUG >= 3:
                        print(f"RKLLM stream token: {repr(content)}")
                      if callback:
                        callback(content, is_finished)
                      yield content, is_finished

                    if is_finished:
                      return

                except json.JSONDecodeError as e:
                  if DEBUG >= 2:
                    print(f"RKLLM stream JSON decode error: {e}, line: {json_str[:100]}")
                  continue
        else:
          error = await resp.text()
          if DEBUG >= 1:
            print(f"RKLLM stream failed: {error}")

    except asyncio.TimeoutError:
      if DEBUG >= 1:
        print("RKLLM stream timed out")
    except Exception as e:
      if DEBUG >= 1:
        print(f"RKLLM stream failed: {e}")

  async def generate_from_prompt_stream(
    self,
    prompt: str,
    callback: Optional[callable] = None
  ) -> AsyncGenerator[Tuple[str, bool], None]:
    """
    Stream tokens from a prompt string.

    Handles pre-templated prompts by extracting user content.

    Args:
      prompt: The user prompt text (may be pre-templated)
      callback: Optional callback for each token

    Yields:
      Tuple of (token_text, is_finished) for each token
    """
    import re

    # Check if prompt is already templated
    user_markers = [
      (r'<｜User｜>(.*?)<｜Assistant｜>', 1),
      (r'<\|user\|>(.*?)<\|assistant\|>', 1),
      (r'\[INST\](.*?)\[/INST\]', 1),
      (r'<\|im_start\|>user\n(.*?)<\|im_end\|>', 1),
    ]

    extracted_content = None
    for pattern, group in user_markers:
      match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
      if match:
        extracted_content = match.group(group).strip()
        break

    if extracted_content:
      if DEBUG >= 2:
        print(f"Stream: Extracted user content: {extracted_content[:100]}...")
      messages = [{"role": "user", "content": extracted_content}]
    else:
      messages = [{"role": "user", "content": prompt}]

    async for token, is_finished in self.generate_stream(messages, callback):
      yield token, is_finished
