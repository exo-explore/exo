import aiofiles.os as aios
from typing import Union
import asyncio
import aiohttp
import json
import os
import sys
import shutil
from urllib.parse import urljoin
from typing import Callable, Optional, Coroutine, Any, Dict, List, Union, Literal
from datetime import datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, TypeVar, TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from exo.helpers import DEBUG, is_frozen
from exo.download.download_progress import RepoProgressEvent, RepoFileProgressEvent, RepoProgressCallback, RepoFileProgressCallback
from exo.inference.shard import Shard
import aiofiles
from aiofiles import os as aios

import exo.main as exo
# remove function : resolve_revision_to_commit_hash

T = TypeVar("T")
# Ori: get_local_snapshot_dir
async def get_local_model_dir(repo_id: str) -> Optional[Path]:
  refs_dir = get_repo_root(repo_id)
  if os.path.exists(refs_dir):
      return refs_dir
  return None


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

# Ori: get_hf_endpoint
def get_lh_endpoint() -> str:
  # http://<local_model_ip>:52525
  return exo.local_model_api_endpoints[0]


def get_exo_home() -> Path: # Ori: get_hf_home
  """Get the exo home directory."""
  exo_path = Path(Path.home()/".cache"/"exo")
  os.makedirs(exo_path, exist_ok=True)
  if DEBUG >= 1: print(f"Init exo folder: {exo_path}")
  return exo_path


async def get_lh_token(): # Ori: get_hf_token
  """get Localhost user's token.
  
  if await aios.path.exists(token_path):
    async with aiofiles.open(token_path, 'r') as f:
      return (await f.read()).strip()
  """
  return None

async def get_auth_headers():
  """Get/Custom authentication headers if a token is available."""
  token = await get_lh_token()
  if token:
    # custom auth header
    return {"Authorization": f"Bearer {token}"}
  return {}


def get_repo_root(model_id: str) -> Path:
  """Get the model directory for a given model ID."""
  sanitized_model_name = str(model_id).split("/")[-1]
  return get_exo_home()/f"{sanitized_model_name}"

async def move_models_to_exo(seed_dir: Union[str, Path]): # Ori: move_models_to_hf
  """Move model in resources folder of app to .cache/exo/"""
  source_dir = Path(seed_dir)
  dest_dir = get_exo_home()
  async for path in source_dir.iterdir():
    print("Start Move")
    if path.is_dir():
      dest_path = dest_dir / path.name
      if dest_path.exists():
        if DEBUG>=1: print(f"skipping moving {dest_path}. File already exists")
      else:
        await aios.rename(str(path), str(dest_path))
  
  print("Done Move")

# Ori: fetch_file_list
@retry(
  stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def fetch_lh_file_list(session, repo_id, path=""):
  model_name = repo_id.split("/")[-1]
  if path :
    url = f"{get_lh_endpoint()}/models/{model_name}/{path}/list"   
  else:
    url = f"{get_lh_endpoint()}/models/{model_name}/list"
  
  headers = await get_auth_headers()

  async with session.get(url, headers=headers, timeout=10) as response:
    if response.status == 200:
      data = await response.json()
      model_datas = data['items']
      files = []
      for item in model_datas:
        if item["type"] == "file":
          files.append({"path": item["name"], "size": item.get("size", "Unknow")})
        elif item["type"] == "directory":
          subfolders_path = f'{path}/{item["name"]}'
          subfiles = await fetch_lh_file_list(session, repo_id, subfolders_path)
          files.extend(subfiles)
      return files
    else:
      raise Exception(f"Failed to fetch file list: {response.status}")
    
@retry(
  stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError)), reraise=True
)
async def download_file(
  session: aiohttp.ClientSession, repo_id: str, revision: str, file_path: str, save_directory: str, progress_callback: Optional[RepoFileProgressCallback] = None, use_range_request: bool = True
):
  #base_url = f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/"
  repo_id = repo_id.split("/")[-1]
  base_url = f'{get_lh_endpoint()}/models/{repo_id}/download/'
  url = urljoin(base_url, file_path)
  local_path = os.path.join(save_directory, file_path)

  await aios.makedirs(os.path.dirname(local_path), exist_ok=True)

  # Check if file already exists and get its size
  local_file_size = await aios.path.getsize(local_path) if await aios.path.exists(local_path) else 0

  headers = await get_auth_headers()
  if use_range_request:
    headers["Range"] = f"bytes={local_file_size}-"

  async with session.get(url, headers=headers) as response:
    total_size = int(response.headers.get('Content-Length', 0))
    downloaded_size = local_file_size
    downloaded_this_session = 0
    mode = 'ab' if use_range_request else 'wb'
    if downloaded_size == total_size:
      if DEBUG >= 2: print(f"File already downloaded: {file_path}")
      if progress_callback:
        await progress_callback(RepoFileProgressEvent(repo_id, revision, file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
      return

    if response.status == 200:
      # File doesn't support range requests or we're not using them, start from beginning
      mode = 'wb'
      downloaded_size = 0
    elif response.status == 206:
      # Partial content, resume download
      content_range = response.headers.get('Content-Range', '')
      try:
        total_size = int(content_range.split('/')[-1])
      except ValueError:
        if DEBUG >= 1: print(f"Failed to parse Content-Range header: {content_range}. Starting download from scratch...")
        return await download_file(session, repo_id, revision, file_path, save_directory, progress_callback, use_range_request=False)
    elif response.status == 416:
      # Range not satisfiable, get the actual file size
      content_range = response.headers.get('Content-Range', '')
      try:
        total_size = int(content_range.split('/')[-1])
        if downloaded_size == total_size:
          if DEBUG >= 2: print(f"File fully downloaded on first pass: {file_path}")
          if progress_callback:
            await progress_callback(RepoFileProgressEvent(repo_id, revision, file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
          return
      except ValueError:
        if DEBUG >= 1: print(f"Failed to parse Content-Range header: {content_range}. Starting download from scratch...")
        return await download_file(session, repo_id, revision, file_path, save_directory, progress_callback, use_range_request=False)
    else:
      raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message=f"Failed to download {file_path}: {response.status}")

    if downloaded_size == total_size:
      print(f"File already downloaded: {file_path}")
      if progress_callback:
        await progress_callback(RepoFileProgressEvent(repo_id, revision, file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
      return

    DOWNLOAD_CHUNK_SIZE = 32768
    start_time = datetime.now()
    async with aiofiles.open(local_path, mode) as f:
      async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
        await f.write(chunk)
        downloaded_size += len(chunk)
        downloaded_this_session += len(chunk)
        if progress_callback and total_size:
          elapsed_time = (datetime.now() - start_time).total_seconds()
          speed = int(downloaded_this_session/elapsed_time) if elapsed_time > 0 else 0
          remaining_size = total_size - downloaded_size
          eta = timedelta(seconds=remaining_size/speed) if speed > 0 else timedelta(0)
          status = "in_progress" if downloaded_size < total_size else "complete"
          if DEBUG >= 8: print(f"HF repo file download progress: {file_path=} {elapsed_time=} {speed=} Downloaded={downloaded_size}/{total_size} {remaining_size=} {eta=} {status=}")
          await progress_callback(RepoFileProgressEvent(repo_id, revision, file_path, downloaded_size, downloaded_this_session, total_size, speed, eta, status))
    if DEBUG >= 2: print(f"Downloaded: {file_path}")

# Ori: download_repo_files
async def download_model_dir(
  repo_id: str,
  revision: str = "Local",
  progress_callback: Optional[RepoProgressCallback] = None,
  allow_patterns: Optional[Union[List[str], str]] = None,
  ignore_patterns: Optional[Union[List[str], str]] = None,
  max_parallel_downloads: int = 4
) -> Path:
  repo_root = get_repo_root(repo_id)
  cachedreqs_dir = get_exo_home()/"cachedreqs"/repo_root

  # Ensure directories exist
  await aios.makedirs(repo_root, exist_ok=True)
  await aios.makedirs(cachedreqs_dir, exist_ok=True)

  # Set up the cached file list directory
  cached_file_list_dir = cachedreqs_dir
  await aios.makedirs(cached_file_list_dir, exist_ok=True)
  cached_file_list_path = cached_file_list_dir/"fetch_file_list.json"

  async with aiohttp.ClientSession() as session:
    # Check if we have a cached file list
    if await aios.path.exists(cached_file_list_path):
      async with aiofiles.open(cached_file_list_path, 'r') as f:
        file_list = json.loads(await f.read())
      if DEBUG >= 2: print(f"Using cached file list from {cached_file_list_path}")
    else:
      file_list = await fetch_lh_file_list(session, repo_id)
      # Cache the file list
      async with aiofiles.open(cached_file_list_path, 'w') as f:
        await f.write(json.dumps(file_list))
      if DEBUG >= 2: print(f"Cached file list at {cached_file_list_path}")
    
    #filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, key=lambda x: x["path"]))
    
    total_files = len(file_list)
    total_bytes = sum(file["size"] for file in file_list)
    file_progress: Dict[str, RepoFileProgressEvent] = {
      file["path"]: RepoFileProgressEvent(repo_id, revision, file["path"], 0, 0, file["size"], 0, timedelta(0), "not_started")
      for file in file_list
    }
    start_time = datetime.now()

    async def download_with_progress(file_info, progress_state):
      local_path = repo_root/file_info["path"]
      if await aios.path.exists(local_path) and (await aios.stat(local_path)).st_size == file_info["size"]:
        if DEBUG >= 2: print(f"File already fully downloaded: {file_info['path']}")
        progress_state['completed_files'] += 1
        progress_state['downloaded_bytes'] += file_info["size"]
        file_progress[file_info["path"]] = RepoFileProgressEvent(repo_id, revision, file_info["path"], file_info["size"], 0, file_info["size"], 0, timedelta(0), "complete")
        if progress_callback:
          elapsed_time = (datetime.now() - start_time).total_seconds()
          overall_speed = int(progress_state['downloaded_bytes_this_session']/elapsed_time) if elapsed_time > 0 else 0
          remaining_bytes = total_bytes - progress_state['downloaded_bytes']
          overall_eta = timedelta(seconds=remaining_bytes/overall_speed) if overall_speed > 0 else timedelta(seconds=0)
          status = "in_progress" if progress_state['completed_files'] < total_files else "complete"
          await progress_callback(
            RepoProgressEvent(
              repo_id, revision, progress_state['completed_files'], total_files, progress_state['downloaded_bytes'], progress_state['downloaded_bytes_this_session'], total_bytes, overall_speed,
              overall_eta, file_progress, status
            )
          )
        return

      async def file_progress_callback(event: RepoFileProgressEvent):
        progress_state['downloaded_bytes'] += event.downloaded - file_progress[event.file_path].downloaded
        progress_state['downloaded_bytes_this_session'] += event.downloaded_this_session - file_progress[event.file_path].downloaded_this_session
        file_progress[event.file_path] = event
        if progress_callback:
          elapsed_time = (datetime.now() - start_time).total_seconds()
          overall_speed = int(progress_state['downloaded_bytes_this_session']/elapsed_time) if elapsed_time > 0 else 0
          remaining_bytes = total_bytes - progress_state['downloaded_bytes']
          overall_eta = timedelta(seconds=remaining_bytes/overall_speed) if overall_speed > 0 else timedelta(seconds=0)
          status = "in_progress" if progress_state['downloaded_bytes'] < total_bytes else "complete"
          await progress_callback(
            RepoProgressEvent(
              repo_id, revision, progress_state['completed_files'], total_files, progress_state['downloaded_bytes'], progress_state['downloaded_bytes_this_session'], total_bytes, overall_speed,
              overall_eta, file_progress, status
            )
          )

      await download_file(session, repo_id, revision, file_info["path"], repo_root, file_progress_callback)
      progress_state['completed_files'] += 1
      file_progress[
        file_info["path"]
      ] = RepoFileProgressEvent(repo_id, revision, file_info["path"], file_info["size"], file_progress[file_info["path"]].downloaded_this_session, file_info["size"], 0, timedelta(0), "complete")
      if progress_callback:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        overall_speed = int(progress_state['downloaded_bytes_this_session']/elapsed_time) if elapsed_time > 0 else 0
        remaining_bytes = total_bytes - progress_state['downloaded_bytes']
        overall_eta = timedelta(seconds=remaining_bytes/overall_speed) if overall_speed > 0 else timedelta(seconds=0)
        status = "in_progress" if progress_state['completed_files'] < total_files else "complete"
        await progress_callback(
          RepoProgressEvent(
            repo_id, revision, progress_state['completed_files'], total_files, progress_state['downloaded_bytes'], progress_state['downloaded_bytes_this_session'], total_bytes, overall_speed,
            overall_eta, file_progress, status
          )
        )

    progress_state = {'completed_files': 0, 'downloaded_bytes': 0, 'downloaded_bytes_this_session': 0}

    semaphore = asyncio.Semaphore(max_parallel_downloads)

    async def download_with_semaphore(file_info):
      async with semaphore:
        await download_with_progress(file_info, progress_state)

    tasks = [asyncio.create_task(download_with_semaphore(file_info)) for file_info in file_list]
    await asyncio.gather(*tasks)

  return repo_root

# Ori: get_weight_map
async def get_lh_weight_map(repo_id: str) -> Optional[Dict[str, str]]:
  """
    Retrieve the weight map from the model.safetensors.index.json file.

    Args:
        repo_id (str): The Hugging Face repository ID.
        revision (str): The revision of the repository to use.

    Returns:
        Optional[Dict[str, str]]: The weight map if it exists, otherwise None.
    """

  # Download the index file
  await download_model_dir(repo_id=repo_id, allow_patterns="model.safetensors.index.json")

  # Check if the file exists
  repo_root = get_repo_root(repo_id)
  index_file = next((f for f in await aios.listdir(repo_root) if f.endswith("model.safetensors.index.json")), None)

  if index_file:
    index_file_path = repo_root/index_file
    if await aios.path.exists(index_file_path):
      async with aiofiles.open(index_file_path, 'r') as f:
        index_data = json.loads(await f.read())
      return index_data.get("weight_map")

  return None


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
  if DEBUG >= 2: print(f"get_allow_patterns {weight_map=} {shard=} {shard_specific_patterns=}")
  return list(default_patterns | shard_specific_patterns)

async def has_exo_home_read_access() -> bool: # Ori: has_hf_home_read_access
  lh_home = get_exo_home()
  try: return await aios.access(lh_home, os.R_OK)
  except OSError: return False

async def has_exo_home_write_access() -> bool: # Ori: has_hf_home_write_access
  lh_home = get_exo_home()
  try: return await aios.access(lh_home, os.W_OK)
  except OSError: return False

async def downlaod_tokenizer_config(repo_id: str):
  await download_model_dir(repo_id=repo_id, allow_patterns="tokenizer_config.json")
