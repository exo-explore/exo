import traceback
from os import PathLike
from aiofiles import os as aios
from typing import Union
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.helpers import DEBUG
from exo.download.new_shard_download import ensure_downloads_dir


class DummyTokenizer:
  def __init__(self):
    self.eos_token_id = 69
    self.vocab_size = 1000

  def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
    return "dummy_tokenized_prompt"

  def encode(self, text):
    return np.array([1])

  def decode(self, tokens):
    return "dummy" * len(tokens)


async def resolve_tokenizer(repo_id: Union[str, PathLike]):
  if repo_id == "dummy":
    return DummyTokenizer()
  local_path = await ensure_downloads_dir()/str(repo_id).replace("/", "--")
  if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
  try:
    if local_path and await aios.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} from {local_path=}")
      return await _resolve_tokenizer(local_path)
  except:
    if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {repo_id=} normally...")
    if DEBUG >= 5: traceback.print_exc()
  return await _resolve_tokenizer(repo_id)


async def _resolve_tokenizer(repo_id_or_local_path: Union[str, PathLike]):
  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {repo_id_or_local_path}")
    processor = AutoProcessor.from_pretrained(repo_id_or_local_path, use_fast=True if "Mistral-Large" in f"{repo_id_or_local_path}" else False, trust_remote_code=True)
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_id_or_local_path}")
    return AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Falling back to tinygrad tokenizer. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"[TODO] Unsupported model: {repo_id_or_local_path}")
