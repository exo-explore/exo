import asyncio
import aiohttp
import os
import argparse
from urllib.parse import urljoin
from typing import Callable, Optional, Coroutine, Any
from datetime import datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, List, TypeVar, Union

T = TypeVar("T")

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.safetensors",
]
# Always ignore `.git` and `.cache/huggingface` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/*",
    "*/.cache/huggingface",
    "**/.cache/huggingface/**",
]

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

async def download_file(session, repo_id, revision, file_path, save_directory, progress_callback: Optional[Callable[[str, int, int, float, timedelta], Coroutine[Any, Any, None]]] = None):
    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/"
    url = urljoin(base_url, file_path)
    local_path = os.path.join(save_directory, file_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Check if file already exists and get its size
    if os.path.exists(local_path):
        local_file_size = os.path.getsize(local_path)
    else:
        local_file_size = 0

    headers = {"Range": f"bytes={local_file_size}-"}
    headers.update(get_auth_headers())
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            # File doesn't support range requests, start from beginning
            mode = 'wb'
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0
        elif response.status == 206:
            # Partial content, resume download
            mode = 'ab'
            content_range = response.headers.get('Content-Range')
            total_size = int(content_range.split('/')[-1])
            downloaded_size = local_file_size
        elif response.status == 416:
            # Range not satisfiable, get the actual file size
            if response.headers.get('Content-Type', '').startswith('text/html'):
                content = await response.text()
                print(f"Response content (HTML):\n{content}")
            else:
                print(response)
            print("Return header: ", response.headers)
            print("Return header: ", response.headers.get('Content-Range').split('/')[-1])
            total_size = int(response.headers.get('Content-Range', '').split('/')[-1])
            if local_file_size == total_size:
                print(f"File already fully downloaded: {file_path}")
                return
            else:
                # Start the download from the beginning
                mode = 'wb'
                downloaded_size = 0
        else:
            print(f"Failed to download {file_path}: {response.status}")
            return

        if downloaded_size == total_size:
            print(f"File already downloaded: {file_path}")
            return

        start_time = datetime.now()
        new_downloaded_size = 0
        with open(local_path, mode) as f:
            async for chunk in response.content.iter_chunked(8192):
                f.write(chunk)
                new_downloaded_size += len(chunk)
                if progress_callback:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    speed = new_downloaded_size / elapsed_time if elapsed_time > 0 else 0
                    eta = timedelta(seconds=(total_size - downloaded_size - new_downloaded_size) / speed) if speed > 0 else timedelta(0)
                    await progress_callback(file_path, new_downloaded_size, total_size - downloaded_size, speed, eta)
        print(f"Downloaded: {file_path}")

async def download_all_files(repo_id, revision="main", progress_callback: Optional[Callable[[int, int, int, int, timedelta, dict], Coroutine[Any, Any, None]]] = None, allow_patterns: Optional[Union[List[str], str]] = None, ignore_patterns: Optional[Union[List[str], str]] = None):
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
                raise Exception(f"Failed to fetch revision info: {response.status}")
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
        completed_files = 0
        total_bytes = sum(file["size"] for file in filtered_file_list)
        downloaded_bytes = 0
        new_downloaded_bytes = 0
        file_progress = {file["path"]: {"status": "not_started", "downloaded": 0, "total": file["size"]} for file in filtered_file_list}
        start_time = datetime.now()

        async def download_with_progress(file_info):
            nonlocal completed_files, downloaded_bytes, new_downloaded_bytes, file_progress

            async def file_progress_callback(path, file_downloaded, file_total, speed, file_eta):
                nonlocal downloaded_bytes, new_downloaded_bytes, file_progress
                new_downloaded_bytes += file_downloaded - file_progress[path]['downloaded']
                downloaded_bytes += file_downloaded - file_progress[path]['downloaded']
                file_progress[path].update({
                    'status': 'in_progress',
                    'downloaded': file_downloaded,
                    'total': file_total,
                    'speed': speed,
                    'eta': file_eta
                })
                if progress_callback:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    overall_speed = new_downloaded_bytes / elapsed_time if elapsed_time > 0 else 0
                    overall_eta = timedelta(seconds=(total_bytes - downloaded_bytes) / overall_speed) if overall_speed > 0 else timedelta(0)
                    await progress_callback(completed_files, total_files, new_downloaded_bytes, total_bytes, overall_eta, file_progress)

            await download_file(session, repo_id, revision, file_info["path"], snapshot_dir, file_progress_callback)
            completed_files += 1
            file_progress[file_info["path"]]['status'] = 'complete'
            if progress_callback:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                overall_speed = new_downloaded_bytes / elapsed_time if elapsed_time > 0 else 0
                overall_eta = timedelta(seconds=(total_bytes - downloaded_bytes) / overall_speed) if overall_speed > 0 else timedelta(0)
                await progress_callback(completed_files, total_files, new_downloaded_bytes, total_bytes, overall_eta, file_progress)

        tasks = [download_with_progress(file_info) for file_info in filtered_file_list]
        await asyncio.gather(*tasks)

async def main(repo_id, revision="main", allow_patterns=None, ignore_patterns=None):
    async def progress_callback(completed_files, total_files, downloaded_bytes, total_bytes, overall_eta, file_progress):
        print(f"Overall Progress: {completed_files}/{total_files} files, {downloaded_bytes}/{total_bytes} bytes")
        print(f"Estimated time remaining: {overall_eta}")
        print("File Progress:")
        for file_path, progress in file_progress.items():
            status_icon = {
                'not_started': '⚪',
                'in_progress': '🔵',
                'complete': '✅'
            }[progress['status']]
            eta_str = str(progress.get('eta', 'N/A'))
            print(f"{status_icon} {file_path}: {progress.get('downloaded', 0)}/{progress['total']} bytes, "
                  f"Speed: {progress.get('speed', 0):.2f} B/s, ETA: {eta_str}")
        print("\n")

    await download_all_files(repo_id, revision, progress_callback, allow_patterns, ignore_patterns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from a Hugging Face model repository.")
    parser.add_argument("--repo-id", help="The repository ID (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct')")
    parser.add_argument("--revision", default="main", help="The revision to download (branch, tag, or commit hash)")
    parser.add_argument("--allow-patterns", nargs="*", default=DEFAULT_ALLOW_PATTERNS, help="Patterns of files to allow (e.g., '*.json' '*.safetensors')")
    parser.add_argument("--ignore-patterns", nargs="*", default=DEFAULT_IGNORE_PATTERNS, help="Patterns of files to ignore (e.g., '.*')")

    args = parser.parse_args()

    asyncio.run(main(args.repo_id, args.revision, args.allow_patterns, args.ignore_patterns))
