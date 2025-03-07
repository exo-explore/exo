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
  model_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
  if not await aios.path.exists(model_dir): return False
  await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
  return True

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
    try: return await _fetch_file_list(repo_id, revision, path)
    except Exception as e:
      if attempt == n_attempts - 1: raise e
      await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

async def _fetch_file_list(repo_id: str, revision: str = "main", path: str = "") -> List[Dict[str, Union[str, int]]]:
  api_url = f"{get_hf_endpoint()}/api/models/{repo_id}/tree/{revision}"
  url = f"{api_url}/{path}" if path else api_url

  headers = await get_auth_headers()
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)) as session:
    async with session.get(url, headers=headers) as response:
      if response.status == 200:
        data = await response.json()
        files = []
        for item in data:
          if item["type"] == "file":
            files.append({"path": item["path"], "size": item["size"]})
          elif item["type"] == "directory":
            subfiles = await _fetch_file_list(repo_id, revision, item["path"])
            files.extend(subfiles)
        return files
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
  headers = await get_auth_headers()
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
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
    n_read = resume_byte_pos or 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
      async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as r:
        if r.status == 404: raise FileNotFoundError(f"File not found: {url}")
        assert r.status in [200, 206], f"Failed to download {path} from {url}: {r.status}"
        async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
          while chunk := await r.content.read(8 * 1024 * 1024): on_progress(n_read := n_read + await f.write(chunk), length)

  final_hash = await calc_hash(partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
  integrity = final_hash == remote_hash
  if not integrity:
    try: await aios.remove(partial_path)
    except Exception as e: print(f"Error removing partial file {partial_path}: {e}")
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
  status = "complete" if all(p.status == "complete" for p in file_progress.values()) else "in_progress" if any(p.status == "in_progress" for p in file_progress.values()) else "not_started"
  return RepoProgressEvent(shard, repo_id, revision, len([p for p in file_progress.values() if p.downloaded == p.total]), len(file_progress), all_downloaded_bytes, all_downloaded_bytes_this_session, all_total_bytes, all_speed, all_eta, file_progress, status)

async def get_weight_map(repo_id: str, revision: str = "main") -> Dict[str, str]:
  target_dir = (await ensure_exo_tmp())/repo_id.replace("/", "--")
  index_file = await download_file_with_retry(repo_id, revision, "model.safetensors.index.json", target_dir)
  async with aiofiles.open(index_file, 'r') as f: index_data = json.loads(await f.read())
  return index_data.get("weight_map")

async def resolve_allow_patterns(shard: Shard, inference_engine_classname: str) -> List[str]:
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

async def download_shard(shard: Shard, inference_engine_classname: str, on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]], max_parallel_downloads: int = 8, skip_download: bool = False) -> tuple[Path, RepoProgressEvent]:
  if DEBUG >= 2 and not skip_download: print(f"Downloading {shard.model_id=} for {inference_engine_classname}")
  repo_id = get_repo(shard.model_id, inference_engine_classname)
  revision = "main"
  target_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
  if not skip_download: await aios.makedirs(target_dir, exist_ok=True)

  if repo_id is None:
    raise ValueError(f"No repo found for {shard.model_id=} and inference engine {inference_engine_classname}")

  allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname)
  if DEBUG >= 2: print(f"Downloading {shard.model_id=} with {allow_patterns=}")

  all_start_time = time.time()
  file_list = await fetch_file_list_with_cache(repo_id, revision)
  filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))
  file_progress: Dict[str, RepoFileProgressEvent] = {}
  def on_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
    start_time = file_progress[file["path"]].start_time if file["path"] in file_progress else time.time()
    downloaded_this_session = file_progress[file["path"]].downloaded_this_session + (curr_bytes - file_progress[file["path"]].downloaded) if file["path"] in file_progress else curr_bytes
    speed = downloaded_this_session / (time.time() - start_time) if time.time() - start_time > 0 else 0
    eta = timedelta(seconds=(total_bytes - curr_bytes) / speed) if speed > 0 else timedelta(seconds=0)
    file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], curr_bytes, downloaded_this_session, total_bytes, speed, eta, "complete" if curr_bytes == total_bytes else "in_progress", start_time)
    on_progress.trigger_all(shard, calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time))
    if DEBUG >= 6: print(f"Downloading {file['path']} {curr_bytes}/{total_bytes} {speed} {eta}")
  for file in filtered_file_list:
    downloaded_bytes = await get_downloaded_size(target_dir/file["path"])
    file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], downloaded_bytes, 0, file["size"], 0, timedelta(0), "complete" if downloaded_bytes == file["size"] else "not_started", time.time())

  semaphore = asyncio.Semaphore(max_parallel_downloads)
  async def download_with_semaphore(file):
    async with semaphore:
      await download_file_with_retry(repo_id, revision, file["path"], target_dir, lambda curr_bytes, total_bytes: on_progress_wrapper(file, curr_bytes, total_bytes))
  if not skip_download: await asyncio.gather(*[download_with_semaphore(file) for file in filtered_file_list])
  final_repo_progress = calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
  on_progress.trigger_all(shard, final_repo_progress)
  if gguf := next((f for f in filtered_file_list if f["path"].endswith(".gguf")), None):
    return target_dir/gguf["path"], final_repo_progress
  else:
    return target_dir, final_repo_progress

def new_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
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
  def __init__(self, max_parallel_downloads: int = 8):
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
    tasks = [download_shard(build_full_shard(model_id, inference_engine_name), inference_engine_name, self.on_progress, skip_download=True) for model_id in get_supported_models([[inference_engine_name]])]
    for task in asyncio.as_completed(tasks):
      try:
        path, progress = await task
        yield (path, progress)
      except Exception as e:
        print("Error downloading shard:", e)
