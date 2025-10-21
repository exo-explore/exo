from typing import AsyncIterator, Optional, List, Dict
from collections import defaultdict
import asyncio

from exo.inference.tokenizers import Tokenizer
from exo.orchestration import Node
from pydantic import BaseModel


class InferenceResultChunk(BaseModel):
  text: str
  tokens: list[int]
  is_finished: bool
  finish_reason: Optional[str]

  def extend(self, other: "InferenceResultChunk"):
    if self.is_finished:
      raise ValueError("Cannot extend a finished chunk")

    self.text += other.text
    self.tokens.extend(other.tokens)
    self.is_finished = other.is_finished
    self.finish_reason = other.finish_reason


class InferenceResultManager:
  node: Node
  tokenizers: dict[str, Tokenizer]
  request_models: dict[str, str]

  """
  Manages inference results and provides an async iterator interface for consuming them.
  This handles buffering tokens and providing them as chunks to clients.
  """

  def __init__(self, node: Node):
    self.node = node
    self.token_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
    self.tokenizers = {}
    self.request_models = {}

    # Register our token handler with the node
    self.token_callback = node.on_token.register("inference-result-manager-token-handler")
    self.token_callback.on_next(self._token_handler_wrapper)

  def _token_handler_wrapper(self, request_id: str, tokens: List[int], is_finished: bool,
                             finish_reason: Optional[str] = None):
    """Wrapper that creates a task for the async handler"""
    asyncio.create_task(self.handle_tokens(request_id, tokens, is_finished, finish_reason))

  def register_tokenizer(self, request_id: str, tokenizer: Tokenizer):
    self.tokenizers[request_id] = tokenizer

  def get_tokenizer(self, request_id: str) -> Tokenizer:
    return self.tokenizers[request_id]

  async def handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool,
                          finish_reason: Optional[str] = None):
    """
    Handle incoming tokens for a specific request and queue them for consumption.

    Args:
        request_id: Unique identifier for the request
        tokens: List of token IDs
        is_finished: Whether this is the final set of tokens
        finish_reason: Reason for finishing, if applicable
    """
    tokenizer = self.get_tokenizer(request_id)

    if len(tokens) > 0 and tokens[-1] == tokenizer.eos_token_id:
      tokens.pop(-1)

    # Skip empty token sets unless it's the final one
    if len(tokens) == 0 and not is_finished:
      return

    await self.token_queues[request_id].put(InferenceResultChunk(
      text=tokenizer.decode(tokens),
      tokens=tokens,
      is_finished=is_finished,
      finish_reason=finish_reason
    ))

  async def get_inference_result(self, request_id: str, timeout: int = 90) -> AsyncIterator[InferenceResultChunk]:
    """
    Get an async iterator that yields inference result chunks as they become available.

    Args:
        request_id: The request ID to get results for
        tokenizer: Optional tokenizer for decoding tokens
        timeout: Maximum time to wait for tokens in seconds

    Yields:
        InferenceResultChunk objects with text and completion status
    """
    while True:
      chunk = await asyncio.wait_for(
        self.token_queues[request_id].get(),
        timeout=timeout
      )

      # Yield the chunk
      yield chunk

      # Exit the loop if this is the final chunk
      if chunk.is_finished:
        break

    # Clean up the queue when done
    if request_id in self.token_queues:
      del self.token_queues[request_id]

    if request_id in self.tokenizers:
      del self.tokenizers[request_id]

  async def get_complete_inference_result(self, request_id: str, timeout: int = 90) -> InferenceResultChunk:
    inference_result = InferenceResultChunk(text="", tokens=[], is_finished=False, finish_reason=None)

    async for chunk in self.get_inference_result(request_id, timeout):
      inference_result.extend(chunk)

    return inference_result

  def register_model_for_request(self, request_id: str, model: str) -> None:
    self.request_models[request_id] = model

  def get_model_for_request(self, request_id: str) -> Optional[str]:
    return self.request_models.get(request_id, None)
