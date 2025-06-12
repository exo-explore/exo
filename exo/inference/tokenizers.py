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
  # Convert to string early to handle WindowsPath objects
  repo_str = str(repo_id_or_local_path)
  
  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {repo_str}")
    processor = AutoProcessor.from_pretrained(repo_id_or_local_path, use_fast=True if "Mistral-Large" in repo_str else False, trust_remote_code=True)
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {repo_str}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_str}")
    return AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_str}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())
  # Special handling for GGUF models - try to get tokenizer from base model
  if "GGUF" in repo_str:
    try:
      # Handle both repo-level GGUF paths and specific file paths
      base_repo_id = repo_str
      
      # If this is a specific .gguf file path, extract the repository part
      if repo_str.endswith('.gguf'):
        # Split by '/' and take everything except the last part (filename)
        parts = repo_str.split('/')
        if len(parts) > 1:
          base_repo_id = '/'.join(parts[:-1])  # Remove the .gguf filename
          if DEBUG >= 4: print(f"Extracted base repo from GGUF file path: {base_repo_id}")
      
      # For GGUF models, try to get the base model name and load tokenizer from there
      if "unsloth/" in base_repo_id and "-GGUF" in base_repo_id:
        # Extract base model name from unsloth GGUF repo
        base_model = base_repo_id.replace("unsloth/", "").replace("-GGUF", "")
        
        # Try different base model variants
        potential_bases = [
          f"meta-llama/{base_model}",
          f"unsloth/{base_model}",
          base_model,
          base_repo_id  # Also try the GGUF repo itself
        ]
        
        for base_repo in potential_bases:
          try:
            if DEBUG >= 4: print(f"Trying base model tokenizer from {base_repo} for GGUF model {repo_id_or_local_path}")
            return AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True)
          except Exception as base_e:
            if DEBUG >= 4: print(f"Failed to load tokenizer from base repo {base_repo}. Error: {base_e}")
            continue
      else:
        # For other GGUF models, try the base repo directly
        try:
          if DEBUG >= 4: print(f"Trying GGUF repo tokenizer from {base_repo_id} for {repo_id_or_local_path}")
          return AutoTokenizer.from_pretrained(base_repo_id, trust_remote_code=True)
        except Exception as base_e:
          if DEBUG >= 4: print(f"Failed to load tokenizer from GGUF repo {base_repo_id}. Error: {base_e}")
          
    except Exception as gguf_e:
      if DEBUG >= 4: print(f"Failed GGUF fallback for {repo_id_or_local_path}. Error: {gguf_e}")

  raise ValueError(f"[TODO] Unsupported model: {repo_id_or_local_path}")
