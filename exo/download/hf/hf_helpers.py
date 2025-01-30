import aiofiles.os as aios
from typing import Union
import os
from typing import Callable, Optional, Dict, List, Union
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, TypeVar
from exo.helpers import DEBUG
from exo.inference.shard import Shard
import aiofiles

T = TypeVar("T")

def filter_repo_objects(
  items: Iterable[T],
  *,
  allow_patterns: Optional[Union[List[str], str]] = None,
  ignore_patterns: Optional[Union[List[str], str]] = None,
  key: Optional[Callable[[T], str]] = None,
) -> Generator[T, None, None]:
  if isinstance(allow_patterns, str):
    allow_patterns = [allow_patterns]
  if isinstance(ignore_patterns, str):
    ignore_patterns = [ignore_patterns]
  if allow_patterns is not None:
    allow_patterns = [_add_wildcard_to_directories(p) for p in allow_patterns]
  if ignore_patterns is not None:
    ignore_patterns = [_add_wildcard_to_directories(p) for p in ignore_patterns]

  if key is None:
    def _identity(item: T) -> str:
      if isinstance(item, str):
        return item
      if isinstance(item, Path):
        return str(item)
      raise ValueError(f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string.")
    key = _identity

  for item in items:
    path = key(item)
    if allow_patterns is not None and not any(fnmatch(path, r) for r in allow_patterns):
      continue
    if ignore_patterns is not None and any(fnmatch(path, r) for r in ignore_patterns):
      continue
    yield item

def _add_wildcard_to_directories(pattern: str) -> str:
  if pattern[-1] == "/":
    return pattern + "*"
  return pattern

def get_hf_endpoint() -> str:
  return os.environ.get('HF_ENDPOINT', "https://huggingface.co")

def get_hf_home() -> Path:
  """Get the Hugging Face home directory."""
  return Path(os.environ.get("HF_HOME", Path.home()/".cache"/"huggingface"))

async def get_hf_token():
  """Retrieve the Hugging Face token from the user's HF_HOME directory."""
  token_path = get_hf_home()/"token"
  if await aios.path.exists(token_path):
    async with aiofiles.open(token_path, 'r') as f:
      return (await f.read()).strip()
  return None

async def get_auth_headers():
  """Get authentication headers if a token is available."""
  token = await get_hf_token()
  if token:
    return {"Authorization": f"Bearer {token}"}
  return {}

def extract_layer_num(tensor_name: str) -> Optional[int]:
  # This is a simple example and might need to be adjusted based on the actual naming convention
  parts = tensor_name.split('.')
  for part in parts:
    if part.isdigit():
      return int(part)
  return None

def get_allow_patterns(weight_map: Dict[str, str], shard: Shard) -> List[str]:
  default_patterns = set(["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"])
  shard_specific_patterns = set()
  if weight_map:
    for tensor_name, filename in weight_map.items():
      layer_num = extract_layer_num(tensor_name)
      if layer_num is not None and shard.start_layer <= layer_num <= shard.end_layer:
        shard_specific_patterns.add(filename)
    sorted_file_names = sorted(weight_map.values())
    if shard.is_first_layer():
      shard_specific_patterns.add(sorted_file_names[0])
    elif shard.is_last_layer():
      shard_specific_patterns.add(sorted_file_names[-1])
  else:
    shard_specific_patterns = set(["*.safetensors"])
  if DEBUG >= 3: print(f"get_allow_patterns {weight_map=} {shard=} {shard_specific_patterns=}")
  return list(default_patterns | shard_specific_patterns)
