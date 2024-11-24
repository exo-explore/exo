import traceback
from aiofiles import os as aios
from os import PathLike
from pathlib import Path
from typing import Union
from transformers import AutoTokenizer, AutoProcessor
from exo.download.hf.hf_helpers import get_local_snapshot_dir
from exo.helpers import DEBUG


class DummyTokenizer:
  def __init__(self):
    self.eos_token_id = 0

  def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
    return [1, 2, 3]

  def decode(self, tokens):
    return "dummy"


async def resolve_tokenizer(model_id: str):
  if model_id == "dummy":
    return DummyTokenizer()
  local_path = await get_local_snapshot_dir(model_id)
  if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
  try:
    if local_path and await aios.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {model_id=} from {local_path=}")
      return await _resolve_tokenizer(local_path)
  except:
    if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {model_id=} normally...")
    if DEBUG >= 5: traceback.print_exc()
  return await _resolve_tokenizer(model_id)


async def _resolve_tokenizer(model_id_or_local_path: Union[str, PathLike]):
  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {model_id_or_local_path}")
    processor = AutoProcessor.from_pretrained(model_id_or_local_path, use_fast=True if "Mistral-Large" in f"{model_id_or_local_path}" else False, trust_remote_code=True)
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {model_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {model_id_or_local_path}")
    return AutoTokenizer.from_pretrained(model_id_or_local_path, trust_remote_code=True)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {model_id_or_local_path}. Falling back to tinygrad tokenizer. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"[TODO] Unsupported model: {model_id_or_local_path}")
