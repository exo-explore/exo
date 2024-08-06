import asyncio
import aiohttp
import os
from urllib.parse import urljoin
from typing import Callable, Optional, Coroutine, Any, Dict, List, Union, Literal
from datetime import datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, TypeVar, TypedDict
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from exo.helpers import DEBUG

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

def get_hf_home() -> Path:
    """Get the Hugging Face home directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

def get_hf_token():
    """Retrieve the Hugging Face token from the user's HF_HOME directory."""
    token_path = get_hf_home() / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None

def get_auth_headers():
    """Get authentication headers if a token is available."""
    token = get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def get_repo_root(repo_id: str) -> Path:
    """Get the root directory for a given repo ID in the Hugging Face cache."""
    sanitized_repo_id = repo_id.replace("/", "--")
    return get_hf_home() / "hub" / f"models--{sanitized_repo_id}"

async def fetch_file_list(session, repo_id, revision, path=""):
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}"
    url = f"{api_url}/{path}" if path else api_url

    headers = get_auth_headers()
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            files = []
            for item in data:
                if item["type"] == "file":
                    files.append({"path": item["path"], "size": item["size"]})
                elif item["type"] == "directory":
                    subfiles = await fetch_file_list(session, repo_id, revision, item["path"])
                    files.extend(subfiles)
            return files
        else:
            raise Exception(f"Failed to fetch file list: {response.status}")


@dataclass
class HFRepoFileProgressEvent:
    file_path: str
    downloaded: int
    downloaded_this_session: int
    total: int
    speed: int
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "downloaded": self.downloaded,
            "downloaded_this_session": self.downloaded_this_session,
            "total": self.total,
            "speed": self.speed,
            "eta": self.eta.total_seconds(),
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data):
        # Convert eta from seconds back to timedelta
        if 'eta' in data:
            data['eta'] = timedelta(seconds=data['eta'])
        return cls(**data)

@dataclass
class HFRepoProgressEvent:
    completed_files: int
    total_files: int
    downloaded_bytes: int
    downloaded_bytes_this_session: int
    total_bytes: int
    overall_speed: int
    overall_eta: timedelta
    file_progress: Dict[str, HFRepoFileProgressEvent]
    status: Literal["not_started", "in_progress", "complete"]

    def to_dict(self):
        return {
            "completed_files": self.completed_files,
            "total_files": self.total_files,
            "downloaded_bytes": self.downloaded_bytes,
            "downloaded_bytes_this_session": self.downloaded_bytes_this_session,
            "total_bytes": self.total_bytes,
            "overall_speed": self.overall_speed,
            "overall_eta": self.overall_eta.total_seconds(),
            "file_progress": {k: v.to_dict() for k, v in self.file_progress.items()},
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data):
        # Convert overall_eta from seconds back to timedelta
        if 'overall_eta' in data:
            data['overall_eta'] = timedelta(seconds=data['overall_eta'])

        # Parse file_progress
        if 'file_progress' in data:
            data['file_progress'] = {
                k: HFRepoFileProgressEvent.from_dict(v)
                for k, v in data['file_progress'].items()
            }

        return cls(**data)

HFRepoFileProgressCallback = Callable[[HFRepoFileProgressEvent], Coroutine[Any, Any, None]]
HFRepoProgressCallback = Callable[[HFRepoProgressEvent], Coroutine[Any, Any, None]]

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError)),
    reraise=True
)
async def download_file(session: aiohttp.ClientSession, repo_id: str, revision: str, file_path: str, save_directory: str, progress_callback: Optional[HFRepoFileProgressCallback] = None, use_range_request: bool = True):
    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/"
    url = urljoin(base_url, file_path)
    local_path = os.path.join(save_directory, file_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Check if file already exists and get its size
    local_file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

    headers = get_auth_headers()
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
                await progress_callback(HFRepoFileProgressEvent(file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
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
                        await progress_callback(HFRepoFileProgressEvent(file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
                    return
            except ValueError:
                if DEBUG >= 1: print(f"Failed to parse Content-Range header: {content_range}. Starting download from scratch...")
                return await download_file(session, repo_id, revision, file_path, save_directory, progress_callback, use_range_request=False)
        else:
            raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message=f"Failed to download {file_path}: {response.status}")

        if downloaded_size == total_size:
            print(f"File already downloaded: {file_path}")
            if progress_callback:
                await progress_callback(HFRepoFileProgressEvent(file_path, downloaded_size, downloaded_this_session, total_size, 0, timedelta(0), "complete"))
            return

        DOWNLOAD_CHUNK_SIZE = 32768
        start_time = datetime.now()
        with open(local_path, mode) as f:
            async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                downloaded_size += len(chunk)
                downloaded_this_session += len(chunk)
                if progress_callback and total_size:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    speed = int(downloaded_this_session / elapsed_time) if elapsed_time > 0 else 0
                    remaining_size = total_size - downloaded_size
                    eta = timedelta(seconds=remaining_size / speed) if speed > 0 else timedelta(0)
                    status = "in_progress" if downloaded_size < total_size else "complete"
                    if DEBUG >= 8: print(f"HF repo file download progress: {file_path=} {elapsed_time=} {speed=} Downloaded={downloaded_size}/{total_size} {remaining_size=} {eta=} {status=}")
                    await progress_callback(HFRepoFileProgressEvent(file_path, downloaded_size, downloaded_this_session, total_size, speed, eta, status))
        if DEBUG >= 2: print(f"Downloaded: {file_path}")

async def download_all_files(repo_id: str, revision: str = "main", progress_callback: Optional[HFRepoProgressCallback] = None, allow_patterns: Optional[Union[List[str], str]] = None, ignore_patterns: Optional[Union[List[str], str]] = None):
    repo_root = get_repo_root(repo_id)
    refs_dir = repo_root / "refs"
    snapshots_dir = repo_root / "snapshots"

    # Ensure directories exist
    refs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # Fetch the commit hash for the given revision
        api_url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
        headers = get_auth_headers()
        async with session.get(api_url, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch revision info from {api_url}: {response.status}")
            revision_info = await response.json()
            commit_hash = revision_info['sha']

        # Write the commit hash to the refs file
        refs_file = refs_dir / revision
        refs_file.write_text(commit_hash)

        # Set up the snapshot directory
        snapshot_dir = snapshots_dir / commit_hash
        snapshot_dir.mkdir(exist_ok=True)

        file_list = await fetch_file_list(session, repo_id, revision)
        filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, key=lambda x: x["path"]))
        total_files = len(filtered_file_list)
        total_bytes = sum(file["size"] for file in filtered_file_list)
        file_progress: Dict[str, HFRepoFileProgressEvent] = {file["path"]: HFRepoFileProgressEvent(file["path"], 0, 0, file["size"], 0, timedelta(0), "not_started") for file in filtered_file_list}
        start_time = datetime.now()

        async def download_with_progress(file_info, progress_state):
            async def file_progress_callback(event: HFRepoFileProgressEvent):
                progress_state['downloaded_bytes'] += event.downloaded - file_progress[event.file_path].downloaded
                progress_state['downloaded_bytes_this_session'] += event.downloaded_this_session - file_progress[event.file_path].downloaded_this_session
                file_progress[event.file_path] = event
                if progress_callback:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    overall_speed = int(progress_state['downloaded_bytes_this_session'] / elapsed_time) if elapsed_time > 0 else 0
                    remaining_bytes = total_bytes - progress_state['downloaded_bytes']
                    overall_eta = timedelta(seconds=remaining_bytes / overall_speed) if overall_speed > 0 else timedelta(seconds=0)
                    status = "in_progress" if progress_state['downloaded_bytes'] < total_bytes else "complete"
                    await progress_callback(HFRepoProgressEvent(progress_state['completed_files'], total_files, progress_state['downloaded_bytes'], progress_state['downloaded_bytes_this_session'], total_bytes, overall_speed, overall_eta, file_progress, status))

            await download_file(session, repo_id, revision, file_info["path"], snapshot_dir, file_progress_callback)
            progress_state['completed_files'] += 1
            file_progress[file_info["path"]] = HFRepoFileProgressEvent(file_info["path"], file_info["size"], file_progress[file_info["path"]].downloaded_this_session, file_info["size"], 0, timedelta(0), "complete")
            if progress_callback:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                overall_speed = int(progress_state['downloaded_bytes_this_session'] / elapsed_time) if elapsed_time > 0 else 0
                remaining_bytes = total_bytes - progress_state['downloaded_bytes']
                overall_eta = timedelta(seconds=remaining_bytes / overall_speed) if overall_speed > 0 else timedelta(seconds=0)
                status = "in_progress" if progress_state['completed_files'] < total_files else "complete"
                await progress_callback(HFRepoProgressEvent(progress_state['completed_files'], total_files, progress_state['downloaded_bytes'], progress_state['downloaded_bytes_this_session'], total_bytes, overall_speed, overall_eta, file_progress, status))

        progress_state = {'completed_files': 0, 'downloaded_bytes': 0, 'downloaded_bytes_this_session': 0}
        tasks = [download_with_progress(file_info, progress_state) for file_info in filtered_file_list]
        await asyncio.gather(*tasks)

    return snapshot_dir
