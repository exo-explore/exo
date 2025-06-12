from exo.inference.shard import Shard
from exo.models import get_repo
from pathlib import Path
from exo.download.hf.hf_helpers import get_hf_endpoint, get_auth_headers, filter_repo_objects, get_allow_patterns
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent, RepoFileProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.models import get_supported_models, build_full_shard
import os
import aiofiles.os as aios
import aiohttp
import aiofiles
from urllib.parse import urljoin
from typing import Callable, Union, Tuple, Dict, List, Optional, Literal, AsyncIterator
import time
from datetime import timedelta
import asyncio
import json
import traceback
import shutil
import tempfile
import hashlib
import psutil
import platform

def exo_home() -> Path:
  return Path(os.environ.get("EXO_HOME", Path.home()/".cache"/"exo"))

def exo_tmp() -> Path:
  return Path(tempfile.gettempdir())/"exo"

async def ensure_exo_home() -> Path:
  await aios.makedirs(exo_home(), exist_ok=True)
  return exo_home()

async def ensure_exo_tmp() -> Path:
  await aios.makedirs(exo_tmp(), exist_ok=True)
  return exo_tmp()

async def has_exo_home_read_access() -> bool:
  try: return await aios.access(exo_home(), os.R_OK)
  except OSError: return False

async def has_exo_home_write_access() -> bool:
  try: return await aios.access(exo_home(), os.W_OK)
  except OSError: return False

async def ensure_downloads_dir() -> Path:
  downloads_dir = exo_home()/"downloads"
  await aios.makedirs(downloads_dir, exist_ok=True)
  return downloads_dir

async def delete_model(model_id: str, inference_engine_name: str) -> bool:
  repo_id = get_repo(model_id, inference_engine_name)
  
  # Handle GGUF file paths that include specific files
  actual_repo_id = repo_id
  specific_file = None
  if repo_id and repo_id.endswith('.gguf'):
    # Split repo_id like "unsloth/Qwen3-0.6B-GGUF/Q4_K_M.gguf" into repo and file
    parts = repo_id.split('/')
    if len(parts) >= 3:
      actual_repo_id = '/'.join(parts[:-1])  # "unsloth/Qwen3-0.6B-GGUF"
      specific_file = parts[-1]  # "Q4_K_M.gguf"
  
  model_dir = await ensure_downloads_dir()/actual_repo_id.replace("/", "--")
  found_files = False
  
  if specific_file:
    # Delete specific file if it exists
    specific_file_path = model_dir / specific_file
    if await aios.path.exists(specific_file_path):
      await aios.remove(specific_file_path)
      found_files = True
    # Also check for partial file
    partial_file_path = model_dir / f"{specific_file}.partial"
    if await aios.path.exists(partial_file_path):
      await aios.remove(partial_file_path)
      found_files = True
  else:
    # Check if model directory exists and remove it
    if await aios.path.exists(model_dir):
      await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
      found_files = True
  
  # Also check for any partial files that might exist
  # This handles cases where only partial downloads exist
  downloads_dir = await ensure_downloads_dir()
  if await aios.path.exists(downloads_dir):
    try:
      async for entry in aios.scandir(downloads_dir):
        entry_path = downloads_dir / entry.name
        # Look for directories that match our model and any partial files
        if entry.is_dir() and entry.name.startswith(actual_repo_id.replace("/", "--")):
          # Check if this directory has any partial files
          try:
            async for subentry in aios.scandir(entry_path):
              if subentry.name.endswith(".partial"):
                await aios.remove(entry_path / subentry.name)
                found_files = True
          except Exception:
            pass
    except Exception:
      pass  # Ignore scanning errors
  
  return found_files

async def seed_models(seed_dir: Union[str, Path]):
  """Move model in resources folder of app to .cache/huggingface/hub"""
  source_dir = Path(seed_dir)
  dest_dir = await ensure_downloads_dir()
  for path in source_dir.iterdir():
    if path.is_dir() and path.name.startswith("models--"):
      dest_path = dest_dir/path.name
      if await aios.path.exists(dest_path): print('Skipping moving model to .cache directory')
      else:
        try: await aios.rename(str(path), str(dest_path))
        except:
          print(f"Error seeding model {path} to {dest_path}")
          traceback.print_exc()

async def fetch_file_list_with_cache(repo_id: str, revision: str = "main") -> List[Dict[str, Union[str, int]]]:
  cache_file = (await ensure_exo_tmp())/f"{repo_id.replace('/', '--')}--{revision}--file_list.json"
  if await aios.path.exists(cache_file):
    async with aiofiles.open(cache_file, 'r') as f: return json.loads(await f.read())
  file_list = await fetch_file_list_with_retry(repo_id, revision)
  await aios.makedirs(cache_file.parent, exist_ok=True)
  async with aiofiles.open(cache_file, 'w') as f: await f.write(json.dumps(file_list))
  return file_list

async def fetch_file_list_with_retry(repo_id: str, revision: str = "main", path: str = "") -> List[Dict[str, Union[str, int]]]:
  n_attempts = 30
  for attempt in range(n_attempts):
    try: 
      result = await _fetch_file_list(repo_id, revision, path)
      if result is None:
        # Repository or path doesn't exist (404 error)
        if DEBUG >= 1: print(f"Repository '{repo_id}' not found or inaccessible (404) - skipping this model")
        return []
      return result
    except Exception as e:
      if attempt == n_attempts - 1: raise e
      await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

async def _fetch_file_list(repo_id: str, revision: str = "main", path: str = "") -> Optional[List[Dict[str, Union[str, int]]]]:
  api_url = f"{get_hf_endpoint()}/api/models/{repo_id}/tree/{revision}"
  url = f"{api_url}/{path}" if path else api_url
  headers = await get_auth_headers()  # Use optimized connector for file list requests
  connector = aiohttp.TCPConnector(
    limit=100, limit_per_host=50, enable_cleanup_closed=True,
    keepalive_timeout=120, ttl_dns_cache=600, use_dns_cache=True
  )
  timeout = aiohttp.ClientTimeout(total=60, connect=20, sock_read=60, sock_connect=20)
  async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
    async with session.get(url, headers=headers) as response:
      if response.status == 200:
        data = await response.json()
        files = []
        for item in data:
          if item["type"] == "file":
            files.append({"path": item["path"], "size": item["size"]})
          elif item["type"] == "directory":
            subfiles = await _fetch_file_list(repo_id, revision, item["path"])
            if subfiles is not None:  # Handle case where subdirectory fetch failed
              files.extend(subfiles)
        return files
      elif response.status == 404:
        # Return None for 404 errors (repository or path doesn't exist)
        return None
      else:
        raise Exception(f"Failed to fetch file list: {response.status}")

async def calc_hash(path: Path, type: Literal["sha1", "sha256"] = "sha1") -> str:
  hash = hashlib.sha1() if type == "sha1" else hashlib.sha256()
  if type == "sha1":
    header = f"blob {(await aios.stat(path)).st_size}\0".encode()
    hash.update(header)
  async with aiofiles.open(path, 'rb') as f:
    while chunk := await f.read(8 * 1024 * 1024):
      hash.update(chunk)
  return hash.hexdigest()

async def file_meta(repo_id: str, revision: str, path: str) -> Tuple[int, str]:
  url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
  headers = await get_auth_headers()  # Use optimized connector for metadata requests too
  connector = aiohttp.TCPConnector(
    limit=100, limit_per_host=50, enable_cleanup_closed=True,
    keepalive_timeout=120, ttl_dns_cache=600, use_dns_cache=True
  )
  timeout = aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)
  async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
    async with session.head(url, headers=headers) as r:
      content_length = int(r.headers.get('x-linked-size') or r.headers.get('content-length') or 0)
      etag = r.headers.get('X-Linked-ETag') or r.headers.get('ETag') or r.headers.get('Etag')
      assert content_length > 0, f"No content length for {url}"
      assert etag is not None, f"No remote hash for {url}"
      if  (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"): etag = etag[1:-1]
      return content_length, etag

async def download_file_with_retry(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
  n_attempts = 30
  for attempt in range(n_attempts):
    try: return await _download_file(repo_id, revision, path, target_dir, on_progress)
    except Exception as e:
      if isinstance(e, FileNotFoundError) or attempt == n_attempts - 1: raise e
      print(f"Download error on attempt {attempt}/{n_attempts} for {repo_id=} {revision=} {path=} {target_dir=}")
      traceback.print_exc()
      await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

async def _download_file(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
  if await aios.path.exists(target_dir/path): return target_dir/path
  await aios.makedirs((target_dir/path).parent, exist_ok=True)
  length, etag = await file_meta(repo_id, revision, path)
  remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
  partial_path = target_dir/f"{path}.partial"
  resume_byte_pos = (await aios.stat(partial_path)).st_size if (await aios.path.exists(partial_path)) else None
  if resume_byte_pos != length:
    url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
    headers = await get_auth_headers()
    if resume_byte_pos: headers['Range'] = f'bytes={resume_byte_pos}-'
    n_read = resume_byte_pos or 0    # Highly optimized connector settings for maximum download speeds
    connector = aiohttp.TCPConnector(
        limit=500,  # Significantly increased total connection pool size
        limit_per_host=100,  # Increased per-host connections for parallel downloads
        enable_cleanup_closed=True, 
        keepalive_timeout=120,  # Longer keepalive for better connection reuse
        ttl_dns_cache=600,  # Cache DNS lookups longer
        use_dns_cache=True,
        # TCP socket options for maximum throughput
        family=0,  # Allow both IPv4 and IPv6
        ssl=False,  # Disable SSL verification for speed (if applicable)
        force_close=False  # Reuse connections
    )
    # Optimized timeout values for high-speed downloads
    timeout = aiohttp.ClientTimeout(
        total=7200,  # 2 hours for very large files
        connect=60,  # Longer connect timeout for stability
        sock_read=600,  # 10 minutes read timeout for large chunks
        sock_connect=60
    )
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        # Additional session optimizations
        read_timeout=None,  # Disable read timeout
        connector_owner=False  # Don't close connector automatically
    ) as session:
      async with session.get(url, headers=headers, timeout=timeout) as r:
        if r.status == 404: raise FileNotFoundError(f"File not found: {url}")
        assert r.status in [200, 206], f"Failed to download {path} from {url}: {r.status}"
        async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
          # Adaptive chunk size based on file size for maximum throughput
          # Start with 32MB chunks for better bandwidth utilization
          base_chunk_size = 32 * 1024 * 1024  # 32MB base
          # Increase chunk size for larger files to reduce overhead
          file_size_mb = length / (1024 * 1024)
          if file_size_mb > 1000:  # Files over 1GB
            chunk_size = 64 * 1024 * 1024  # 64MB chunks
          elif file_size_mb > 100:  # Files over 100MB
            chunk_size = 48 * 1024 * 1024  # 48MB chunks
          else:
            chunk_size = base_chunk_size  # 32MB chunks
          
          while chunk := await r.content.read(chunk_size): 
            on_progress(n_read := n_read + await f.write(chunk), length)

  final_hash = await calc_hash(partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
  integrity = final_hash == remote_hash
  if not integrity:
    try: await aios.remove(partial_path)
    except Exception as e: print(f"Error removing partial file {partial_path}: {e}")
    raise Exception(f"Downloaded file {target_dir/path} has hash {final_hash} but remote hash is {remote_hash}")
  await aios.rename(partial_path, target_dir/path)
  return target_dir/path

async def download_file_multipart(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None, parts: int = 4) -> Path:
  """
  Download a file using multiple parallel connections for maximum speed.
  This is especially effective for very large files (>1GB).
  """
  if await aios.path.exists(target_dir/path): 
    return target_dir/path
    
  await aios.makedirs((target_dir/path).parent, exist_ok=True)
  length, etag = await file_meta(repo_id, revision, path)
  
  # Only use multipart for files larger than 100MB
  if length < 100 * 1024 * 1024:
    return await _download_file(repo_id, revision, path, target_dir, on_progress)
  
  remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
  partial_path = target_dir/f"{path}.partial"
  
  # Check if file already exists and is complete
  if await aios.path.exists(partial_path):
    current_size = (await aios.stat(partial_path)).st_size
    if current_size == length:
      # Verify hash and move if correct
      final_hash = await calc_hash(partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
      if final_hash == remote_hash:
        await aios.rename(partial_path, target_dir/path)
        return target_dir/path
  
  url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
  headers = await get_auth_headers()
  
  # Calculate part sizes
  part_size = length // parts
  ranges = []
  for i in range(parts):
    start = i * part_size
    end = start + part_size - 1 if i < parts - 1 else length - 1
    ranges.append((start, end))
    # Highly optimized connector for multipart downloads
  connector = aiohttp.TCPConnector(
    limit=parts * 2,  # More connections for multipart
    limit_per_host=parts * 2,
    enable_cleanup_closed=True,
    keepalive_timeout=120,
    ttl_dns_cache=600,
    use_dns_cache=True
  )
  
  timeout = aiohttp.ClientTimeout(total=7200, connect=60, sock_read=600, sock_connect=60)
  
  # Create temporary files for each part
  part_files = [target_dir/f"{path}.part{i}" for i in range(parts)]
  total_downloaded = 0
  download_lock = asyncio.Lock()
  
  async def download_part(session, part_idx, start, end):
    nonlocal total_downloaded
    part_headers = headers.copy()
    part_headers['Range'] = f'bytes={start}-{end}'
    
    async with session.get(url, headers=part_headers) as r:
      assert r.status in [200, 206], f"Failed to download part {part_idx}: {r.status}"
      
      async with aiofiles.open(part_files[part_idx], 'wb') as f:
        chunk_size = 16 * 1024 * 1024  # 16MB chunks
        async for chunk in r.content.iter_chunked(chunk_size):
          await f.write(chunk)
          async with download_lock:
            total_downloaded += len(chunk)
            on_progress(total_downloaded, length)
  
  # Download all parts in parallel
  async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
    tasks = [download_part(session, i, start, end) for i, (start, end) in enumerate(ranges)]
    await asyncio.gather(*tasks)
  
  # Combine parts into final file
  async with aiofiles.open(partial_path, 'wb') as final_file:
    for part_file in part_files:
      async with aiofiles.open(part_file, 'rb') as part:
        while chunk := await part.read(32 * 1024 * 1024):  # 32MB chunks for combining
          await final_file.write(chunk)
      # Clean up part file
      await aios.remove(part_file)
  
  # Verify hash
  final_hash = await calc_hash(partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
  if final_hash != remote_hash:
    try: 
      await aios.remove(partial_path)
    except Exception as e: 
      print(f"Error removing partial file {partial_path}: {e}")
    raise Exception(f"Downloaded file {target_dir/path} has hash {final_hash} but remote hash is {remote_hash}")
  
  await aios.rename(partial_path, target_dir/path)
  return target_dir/path


def calculate_repo_progress(shard: Shard, repo_id: str, revision: str, file_progress: Dict[str, RepoFileProgressEvent], all_start_time: float) -> RepoProgressEvent:
  all_total_bytes = sum([p.total for p in file_progress.values()])
  all_downloaded_bytes = sum([p.downloaded for p in file_progress.values()])
  all_downloaded_bytes_this_session = sum([p.downloaded_this_session for p in file_progress.values()])
  elapsed_time = time.time() - all_start_time
  all_speed = all_downloaded_bytes_this_session / elapsed_time if elapsed_time > 0 else 0
  all_eta = timedelta(seconds=(all_total_bytes - all_downloaded_bytes) / all_speed) if all_speed > 0 else timedelta(seconds=0)
  # Handle empty file_progress (no files found) - should be "not_started", not "complete"
  if not file_progress:
    status = "not_started"
  else:
    status = "complete" if all(p.status == "complete" for p in file_progress.values()) else "in_progress" if any(p.status == "in_progress" for p in file_progress.values()) else "not_started"
  return RepoProgressEvent(shard, repo_id, revision, len([p for p in file_progress.values() if p.downloaded == p.total]), len(file_progress), all_downloaded_bytes, all_downloaded_bytes_this_session, all_total_bytes, all_speed, all_eta, file_progress, status)

async def get_weight_map(repo_id: str, revision: str = "main") -> Dict[str, str]:
  target_dir = (await ensure_exo_tmp())/repo_id.replace("/", "--")
  index_file = await download_file_with_retry(repo_id, revision, "model.safetensors.index.json", target_dir)
  async with aiofiles.open(index_file, 'r') as f: index_data = json.loads(await f.read())
  return index_data.get("weight_map")

async def resolve_allow_patterns(shard: Shard, inference_engine_classname: str) -> List[str]:
  # GGUF models (LlamaCpp) don't use safetensors format, so download all files
  if inference_engine_classname == "LlamaCppInferenceEngine":
    if DEBUG >= 2: print(f"Using wildcard patterns for GGUF model {shard.model_id}")
    return ["*"]
  
  # For other inference engines (MLX, Tinygrad), try to get weight map from safetensors index
  try:
    weight_map = await get_weight_map(get_repo(shard.model_id, inference_engine_classname))
    return get_allow_patterns(weight_map, shard)
  except:
    if DEBUG >= 1: print(f"Error getting weight map for {shard.model_id=} and inference engine {inference_engine_classname}")
    if DEBUG >= 1: traceback.print_exc()
    return ["*"]

async def get_downloaded_size(path: Path) -> int:
  partial_path = path.with_suffix(path.suffix + ".partial")
  if await aios.path.exists(path): return (await aios.stat(path)).st_size
  if await aios.path.exists(partial_path): return (await aios.stat(partial_path)).st_size
  return 0

async def is_file_complete(path: Path) -> bool:
  """Check if a file is completely downloaded (not partial)"""
  return await aios.path.exists(path)

async def download_shard(shard: Shard, inference_engine_classname: str, on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]], max_parallel_downloads: int = 32, skip_download: bool = False) -> tuple[Path, RepoProgressEvent]:
  if DEBUG >= 1 and not skip_download: 
    print(f"Starting download for shard {shard.start_layer}-{shard.end_layer} out of {shard.n_layers} layers for model {shard.model_id} using {inference_engine_classname}")
  
  repo_id = get_repo(shard.model_id, inference_engine_classname)
  revision = "main"
  
  if repo_id is None:
    raise ValueError(f"No repo found for {shard.model_id=} and inference engine {inference_engine_classname}")

  # Handle GGUF file paths that include specific files
  actual_repo_id = repo_id
  specific_file = None
  if repo_id.endswith('.gguf'):
    # Split repo_id like "unsloth/Qwen3-0.6B-GGUF/Q4_K_M.gguf" into repo and file
    parts = repo_id.split('/')
    if len(parts) >= 3:
      actual_repo_id = '/'.join(parts[:-1])  # "unsloth/Qwen3-0.6B-GGUF"
      specific_file = parts[-1]  # "Q4_K_M.gguf"
  
  target_dir = await ensure_downloads_dir()/actual_repo_id.replace("/", "--")
  if not skip_download: await aios.makedirs(target_dir, exist_ok=True)

  # Override allow_patterns for specific GGUF files
  if specific_file:
    allow_patterns = [specific_file]
  else:
    allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname)
  
  if DEBUG >= 1: 
    print(f"Downloading {shard.model_id=} with {allow_patterns=} for shard layers {shard.start_layer}-{shard.end_layer}")

  all_start_time = time.time()
  file_list = await fetch_file_list_with_cache(actual_repo_id, revision)
  filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))
  file_progress: Dict[str, RepoFileProgressEvent] = {}
  def on_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
    start_time = file_progress[file["path"]].start_time if file["path"] in file_progress else time.time()
    downloaded_this_session = file_progress[file["path"]].downloaded_this_session + (curr_bytes - file_progress[file["path"]].downloaded) if file["path"] in file_progress else curr_bytes
    speed = downloaded_this_session / (time.time() - start_time) if time.time() - start_time > 0 else 0
    eta = timedelta(seconds=(total_bytes - curr_bytes) / speed) if speed > 0 else timedelta(seconds=0)
    file_progress[file["path"]] = RepoFileProgressEvent(actual_repo_id, revision, file["path"], curr_bytes, downloaded_this_session, total_bytes, speed, eta, "complete" if curr_bytes == total_bytes else "in_progress", start_time)
    on_progress.trigger_all(shard, calculate_repo_progress(shard, actual_repo_id, revision, file_progress, all_start_time))
    if DEBUG >= 6: print(f"Downloading {file['path']} {curr_bytes}/{total_bytes} {speed} {eta}")
  for file in filtered_file_list:
    downloaded_bytes = await get_downloaded_size(target_dir/file["path"])
    is_complete = await is_file_complete(target_dir/file["path"])
    file_progress[file["path"]] = RepoFileProgressEvent(actual_repo_id, revision, file["path"], downloaded_bytes, 0, file["size"], 0, timedelta(0), "complete" if is_complete and downloaded_bytes == file["size"] else "not_started", time.time())
  semaphore = asyncio.Semaphore(max_parallel_downloads)
  async def download_with_semaphore(file):
    async with semaphore:
      # Use multipart download for large GGUF files (>500MB) for maximum speed
      file_size_mb = file["size"] / (1024 * 1024)
      if file["path"].endswith(".gguf") and file_size_mb > 500:
        if DEBUG >= 1: print(f"Using multipart download for large GGUF file: {file['path']} ({file_size_mb:.1f}MB)")
        await download_file_multipart(actual_repo_id, revision, file["path"], target_dir, 
                                     lambda curr_bytes, total_bytes: on_progress_wrapper(file, curr_bytes, total_bytes),
                                     parts=8)  # Use 8 parts for very large files
      else:
        await download_file_with_retry(actual_repo_id, revision, file["path"], target_dir, 
                                     lambda curr_bytes, total_bytes: on_progress_wrapper(file, curr_bytes, total_bytes))
  if not skip_download: await asyncio.gather(*[download_with_semaphore(file) for file in filtered_file_list])
  final_repo_progress = calculate_repo_progress(shard, actual_repo_id, revision, file_progress, all_start_time)
  on_progress.trigger_all(shard, final_repo_progress)
  if gguf := next((f for f in filtered_file_list if f["path"].endswith(".gguf")), None):
    return target_dir/gguf["path"], final_repo_progress
  else:
    return target_dir, final_repo_progress

def new_shard_downloader(max_parallel_downloads: int = 32) -> ShardDownloader:
  return SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads)))

class SingletonShardDownloader(ShardDownloader):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard_downloader = shard_downloader
    self.active_downloads: Dict[Shard, asyncio.Task] = {}

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self.shard_downloader.on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    if shard not in self.active_downloads: self.active_downloads[shard] = asyncio.create_task(self.shard_downloader.ensure_shard(shard, inference_engine_name))
    try: return await self.active_downloads[shard]
    finally:
      if shard in self.active_downloads and self.active_downloads[shard].done(): del self.active_downloads[shard]

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
      yield path, status

class CachedShardDownloader(ShardDownloader):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard_downloader = shard_downloader
    self.cache: Dict[tuple[str, Shard], Path] = {}

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self.shard_downloader.on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    if (inference_engine_name, shard) in self.cache:
      if DEBUG >= 2: print(f"ensure_shard cache hit {shard=} for {inference_engine_name}")
      return self.cache[(inference_engine_name, shard)]
    if DEBUG >= 2: print(f"ensure_shard cache miss {shard=} for {inference_engine_name}")
    target_dir = await self.shard_downloader.ensure_shard(shard, inference_engine_name)
    self.cache[(inference_engine_name, shard)] = target_dir
    return target_dir

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
      yield path, status

class NewShardDownloader(ShardDownloader):
  def __init__(self, max_parallel_downloads: int = 32):
    self.max_parallel_downloads = max_parallel_downloads
    self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self._on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    target_dir, _ = await download_shard(shard, inference_engine_name, self.on_progress, max_parallel_downloads=self.max_parallel_downloads)
    return target_dir
  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if DEBUG >= 2: print("Getting shard download status for", inference_engine_name)
    
    async def download_with_timeout(model_id):
      try:
        # Add 30 second timeout per model to prevent hanging
        return await asyncio.wait_for(
          download_shard(build_full_shard(model_id, inference_engine_name), inference_engine_name, self.on_progress, skip_download=True),
          timeout=30.0
        )
      except asyncio.TimeoutError:
        if DEBUG >= 1: print(f"Timeout checking status for model: {model_id}")
        return None
      except Exception as e:
        if DEBUG >= 1: print(f"Error checking status for model {model_id}: {e}")
        return None
    
    tasks = [download_with_timeout(model_id) for model_id in get_supported_models([[inference_engine_name]])]
    for task in asyncio.as_completed(tasks):
      try:
        result = await task
        if result is not None:
          path, progress = result
          yield (path, progress)
      except Exception as e:
        # Handle any remaining errors gracefully
        error_msg = str(e)
        if "404" in error_msg or "Failed to fetch file list: 404" in error_msg:
          if DEBUG >= 1: print(f"Skipping model due to 404 error (repository not accessible): {error_msg}")
        else:
          # Log other errors as they might be more serious
          if DEBUG >= 1: print(f"Error downloading shard: {error_msg}")
          if DEBUG >= 2: print(f"Full traceback: {traceback.format_exc()}")

async def test_bandwidth_and_optimize() -> dict:
    """
    Perform a quick bandwidth test using a small file download to optimize settings.
    Returns optimized settings based on detected bandwidth.
    """
    try:
        import time
        test_start = time.time()
        
        # Use a small file from HuggingFace for testing (usually fast and reliable)
        test_url = f"{get_hf_endpoint()}/datasets/huggingface/documentation-images/resolve/main/blog/ov_blog_cover.jpg"
        headers = await get_auth_headers()
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(test_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.read()
                    test_end = time.time()
                    
                    # Calculate bandwidth
                    elapsed = test_end - test_start
                    if elapsed > 0:
                        bytes_per_second = len(data) / elapsed
                        mbps = (bytes_per_second * 8) / (1024 * 1024)
                        
                        # Optimize based on bandwidth
                        if mbps >= 100:  # Very fast connection (100+ Mbps)
                            return {
                                "max_parallel_downloads": 96,
                                "multipart_parts": 12,
                                "chunk_size_mb": 64,
                                "bandwidth_mbps": mbps
                            }
                        elif mbps >= 50:  # Fast connection (50-100 Mbps)
                            return {
                                "max_parallel_downloads": 64,
                                "multipart_parts": 8,
                                "chunk_size_mb": 48,
                                "bandwidth_mbps": mbps
                            }
                        elif mbps >= 25:  # Medium connection (25-50 Mbps)
                            return {
                                "max_parallel_downloads": 48,
                                "multipart_parts": 6,
                                "chunk_size_mb": 32,
                                "bandwidth_mbps": mbps
                            }
                        elif mbps >= 10:  # Slower connection (10-25 Mbps)
                            return {
                                "max_parallel_downloads": 32,
                                "multipart_parts": 4,
                                "chunk_size_mb": 24,
                                "bandwidth_mbps": mbps
                            }
                        else:  # Very slow connection (<10 Mbps)
                            return {
                                "max_parallel_downloads": 16,
                                "multipart_parts": 2,
                                "chunk_size_mb": 16,
                                "bandwidth_mbps": mbps
                            }
    except Exception as e:
        if DEBUG >= 1:
            print(f"Bandwidth test failed: {e}")
    
    # Fallback to default optimized settings
    return {
        "max_parallel_downloads": 48,
        "multipart_parts": 6,
        "chunk_size_mb": 32,
        "bandwidth_mbps": None
    }

def get_optimal_download_settings() -> dict:
    """
    Auto-detect optimal download settings based on system capabilities.
    Returns a dictionary with recommended settings for maximum download speed.
    """
    settings = {
        "max_parallel_downloads": 32,
        "chunk_size_mb": 32,
        "use_multipart": True,
        "multipart_threshold_mb": 500,
        "multipart_parts": 4
    }
    
    # Get system info
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adjust based on CPU cores
    if cpu_count >= 16:  # High-end systems
        settings["max_parallel_downloads"] = 64
        settings["multipart_parts"] = 8
        settings["chunk_size_mb"] = 64
    elif cpu_count >= 8:  # Mid-range systems
        settings["max_parallel_downloads"] = 48
        settings["multipart_parts"] = 6
        settings["chunk_size_mb"] = 48
    elif cpu_count >= 4:  # Lower-end systems
        settings["max_parallel_downloads"] = 32
        settings["multipart_parts"] = 4
        settings["chunk_size_mb"] = 32
    else:  # Very low-end systems
        settings["max_parallel_downloads"] = 16
        settings["multipart_parts"] = 2
        settings["chunk_size_mb"] = 16
    
    # Adjust based on available memory
    if memory_gb < 8:  # Low memory systems
        settings["max_parallel_downloads"] = min(settings["max_parallel_downloads"], 16)
        settings["chunk_size_mb"] = min(settings["chunk_size_mb"], 16)
        settings["multipart_parts"] = min(settings["multipart_parts"], 2)
    elif memory_gb >= 32:  # High memory systems
        settings["max_parallel_downloads"] = max(settings["max_parallel_downloads"], 64)
        settings["chunk_size_mb"] = max(settings["chunk_size_mb"], 64)
    
    # Windows-specific optimizations
    if platform.system() == "Windows":
        # Windows can handle more connections efficiently
        settings["max_parallel_downloads"] = min(settings["max_parallel_downloads"] * 1.25, 96)
    
    if DEBUG >= 1:
        print(f"🔧 Auto-detected optimal settings: {settings}")
        print(f"   System: {cpu_count} cores, {memory_gb:.1f}GB RAM")
    
    return settings
