import asyncio
import hashlib
import os
import shutil
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urljoin

import aiofiles
import aiofiles.os as aios
import aiohttp
from pydantic import BaseModel, DirectoryPath, Field, PositiveInt, TypeAdapter

from exo.shared.constants import EXO_HOME
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.huggingface_utils import (
    filter_repo_objects,
    get_allow_patterns,
    get_auth_headers,
    get_hf_endpoint,
)


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt


class ModelSafetensorsIndex(BaseModel):
    metadata: Optional[ModelSafetensorsIndexMetadata]
    weight_map: Dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    """Progress information for an individual file within a repository download."""

    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: int
    downloaded_this_session: int
    total: int
    speed: float  # bytes per second
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    class Config:
        frozen = True


class RepoDownloadProgress(BaseModel):
    """Aggregated download progress information for a repository/shard combination.

    This structure captures the overall progress of downloading the files
    required to materialise a particular *shard* of a model.  It purposely
    mirrors the key summary fields emitted by the `RepoProgressEvent` so that
    the event payload can be cleanly projected onto the long-lived cluster
    state.
    """

    repo_id: str
    repo_revision: str
    shard: ShardMetadata

    # progress totals
    completed_files: int
    total_files: int
    downloaded_bytes: int
    downloaded_bytes_this_session: int
    total_bytes: int

    # speed / eta
    overall_speed: float  # bytes per second
    overall_eta: timedelta

    # lifecycle status
    status: Literal["not_started", "in_progress", "complete"]

    # fine-grained file progress keyed by file_path
    file_progress: Dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    class Config:
        frozen = True  # allow use as dict keys if desired


def build_model_path(model_id: str) -> DirectoryPath:
    return EXO_HOME / "models" / model_id.replace("/", "--")


async def resolve_model_path_for_repo(repo_id: str) -> Path:
    return (await ensure_models_dir()) / repo_id.replace("/", "--")


async def ensure_exo_home() -> Path:
    await aios.makedirs(EXO_HOME, exist_ok=True)
    return EXO_HOME


async def has_exo_home_read_access() -> bool:
    try:
        return await aios.access(EXO_HOME, os.R_OK)
    except OSError:
        return False


async def has_exo_home_write_access() -> bool:
    try:
        return await aios.access(EXO_HOME, os.W_OK)
    except OSError:
        return False


async def ensure_models_dir() -> Path:
    models_dir = EXO_HOME / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    return models_dir


async def delete_model(repo_id: str) -> bool:
    model_dir = await ensure_models_dir() / repo_id.replace("/", "--")
    if not await aios.path.exists(model_dir):
        return False
    await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
    return True


async def seed_models(seed_dir: Union[str, Path]):
    """Move model in resources folder of app to .cache/huggingface/hub"""
    source_dir = Path(seed_dir)
    dest_dir = await ensure_models_dir()
    for path in source_dir.iterdir():
        if path.is_dir() and path.name.startswith("models--"):
            dest_path = dest_dir / path.name
            if await aios.path.exists(dest_path):
                print("Skipping moving model to .cache directory")
            else:
                try:
                    await aios.rename(str(path), str(dest_path))
                except Exception:
                    print(f"Error seeding model {path} to {dest_path}")
                    traceback.print_exc()


async def fetch_file_list_with_cache(
    repo_id: str, revision: str = "main", recursive: bool = False
) -> List[FileListEntry]:
    target_dir = (
        (await ensure_models_dir()) / "caches" / str(repo_id).replace("/", "--")
    )
    await aios.makedirs(target_dir, exist_ok=True)
    cache_file = (
        target_dir / f"{repo_id.replace('/', '--')}--{revision}--file_list.json"
    )
    if await aios.path.exists(cache_file):
        async with aiofiles.open(cache_file, "r") as f:
            return TypeAdapter(List[FileListEntry]).validate_json(await f.read())
    file_list = await fetch_file_list_with_retry(repo_id, revision, recursive=recursive)
    await aios.makedirs(cache_file.parent, exist_ok=True)
    async with aiofiles.open(cache_file, "w") as f:
        await f.write(TypeAdapter(List[FileListEntry]).dump_json(file_list).decode())
    return file_list


async def fetch_file_list_with_retry(
    repo_id: str, revision: str = "main", path: str = "", recursive: bool = False
) -> List[FileListEntry]:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            return await _fetch_file_list(repo_id, revision, path, recursive)
        except Exception as e:
            if attempt == n_attempts - 1:
                raise e
            await asyncio.sleep(min(8, 0.1 * float(2.0 ** int(attempt))))
    raise Exception(
        f"Failed to fetch file list for {repo_id=} {revision=} {path=} {recursive=}"
    )


async def _fetch_file_list(
    repo_id: str, revision: str = "main", path: str = "", recursive: bool = False
) -> List[FileListEntry]:
    api_url = f"{get_hf_endpoint()}/api/models/{repo_id}/tree/{revision}"
    url = f"{api_url}/{path}" if path else api_url

    headers = await get_auth_headers()
    async with (
        aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=30, connect=10, sock_read=30, sock_connect=10
            )
        ) as session,
        session.get(url, headers=headers) as response,
    ):
        if response.status == 200:
            data_json = await response.text()
            data = TypeAdapter(list[FileListEntry]).validate_json(data_json)
            files: list[FileListEntry] = []
            for item in data:
                if item.type == "file":
                    files.append(FileListEntry.model_validate(item))
                elif item.type == "directory" and recursive:
                    subfiles = await _fetch_file_list(
                        repo_id, revision, item.path, recursive
                    )
                    files.extend(subfiles)
            return files
        else:
            raise Exception(f"Failed to fetch file list: {response.status}")


async def calc_hash(path: Path, hash_type: Literal["sha1", "sha256"] = "sha1") -> str:
    hasher = hashlib.sha1() if hash_type == "sha1" else hashlib.sha256()
    if hash_type == "sha1":
        header = f"blob {(await aios.stat(path)).st_size}\0".encode()
        hasher.update(header)
    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


async def file_meta(
    repo_id: str, revision: str, path: str, redirected_location: str | None = None
) -> Tuple[int, str]:
    url = (
        urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
        if redirected_location is None
        else f"{get_hf_endpoint()}{redirected_location}"
    )
    headers = await get_auth_headers()
    async with (
        aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800, connect=60, sock_read=1800, sock_connect=60
            )
        ) as session,
        session.head(url, headers=headers) as r,
    ):
        if r.status == 307:
            # Try to extract from X-Linked headers first (common for HF redirects)
            content_length = int(
                r.headers.get("x-linked-size") or r.headers.get("content-length") or 0
            )
            etag = (
                r.headers.get("X-Linked-ETag")
                or r.headers.get("ETag")
                or r.headers.get("Etag")
            )
            if content_length > 0 and etag is not None:
                if (etag[0] == '"' and etag[-1] == '"') or (
                    etag[0] == "'" and etag[-1] == "'"
                ):
                    etag = etag[1:-1]
                return content_length, etag
            # If not available, recurse with the redirect
            redirected_location = r.headers.get("Location")
            return await file_meta(repo_id, revision, path, redirected_location)
        content_length = int(
            r.headers.get("x-linked-size") or r.headers.get("content-length") or 0
        )
        etag = (
            r.headers.get("X-Linked-ETag")
            or r.headers.get("ETag")
            or r.headers.get("Etag")
        )
        assert content_length > 0, f"No content length for {url}"
        assert etag is not None, f"No remote hash for {url}"
        if (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"):
            etag = etag[1:-1]
        return content_length, etag


async def download_file_with_retry(
    repo_id: str,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int], None] = lambda _, __: None,
) -> Path:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            return await _download_file(
                repo_id, revision, path, target_dir, on_progress
            )
        except Exception as e:
            if isinstance(e, FileNotFoundError) or attempt == n_attempts - 1:
                raise e
            print(
                f"Download error on attempt {attempt}/{n_attempts} for {repo_id=} {revision=} {path=} {target_dir=}"
            )
            traceback.print_exc()
            await asyncio.sleep(min(8, 0.1 * (2.0**attempt)))
    raise Exception(
        f"Failed to download file {repo_id=} {revision=} {path=} {target_dir=}"
    )


async def _download_file(
    repo_id: str,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int], None] = lambda _, __: None,
) -> Path:
    if await aios.path.exists(target_dir / path):
        return target_dir / path
    await aios.makedirs((target_dir / path).parent, exist_ok=True)
    length, etag = await file_meta(repo_id, revision, path)
    remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    partial_path = target_dir / f"{path}.partial"
    resume_byte_pos = (
        (await aios.stat(partial_path)).st_size
        if (await aios.path.exists(partial_path))
        else None
    )
    if resume_byte_pos != length:
        url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
        headers = await get_auth_headers()
        if resume_byte_pos:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        n_read = resume_byte_pos or 0
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=1800, connect=60, sock_read=1800, sock_connect=60
                )
            ) as session,
            session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=1800, connect=60, sock_read=1800, sock_connect=60
                ),
            ) as r,
        ):
            if r.status == 404:
                raise FileNotFoundError(f"File not found: {url}")
            assert r.status in [200, 206], (
                f"Failed to download {path} from {url}: {r.status}"
            )
            async with aiofiles.open(
                partial_path, "ab" if resume_byte_pos else "wb"
            ) as f:
                while chunk := await r.content.read(8 * 1024 * 1024):
                    n_read = n_read + (await f.write(chunk))
                    on_progress(n_read, length)

    final_hash = await calc_hash(
        partial_path, hash_type="sha256" if len(remote_hash) == 64 else "sha1"
    )
    integrity = final_hash == remote_hash
    if not integrity:
        try:
            await aios.remove(partial_path)
        except Exception as e:
            print(f"Error removing partial file {partial_path}: {e}")
        raise Exception(
            f"Downloaded file {target_dir / path} has hash {final_hash} but remote hash is {remote_hash}"
        )
    await aios.rename(partial_path, target_dir / path)
    return target_dir / path


def calculate_repo_progress(
    shard: ShardMetadata,
    repo_id: str,
    revision: str,
    file_progress: Dict[str, RepoFileDownloadProgress],
    all_start_time: float,
) -> RepoDownloadProgress:
    all_total_bytes = sum(p.total for p in file_progress.values())
    all_downloaded_bytes = sum(p.downloaded for p in file_progress.values())
    all_downloaded_bytes_this_session = sum(
        p.downloaded_this_session for p in file_progress.values()
    )
    elapsed_time = time.time() - all_start_time
    all_speed = (
        all_downloaded_bytes_this_session / elapsed_time if elapsed_time > 0 else 0
    )
    all_eta = (
        timedelta(seconds=(all_total_bytes - all_downloaded_bytes) / all_speed)
        if all_speed > 0
        else timedelta(seconds=0)
    )
    status = (
        "complete"
        if all(p.status == "complete" for p in file_progress.values())
        else "in_progress"
        if any(p.status == "in_progress" for p in file_progress.values())
        else "not_started"
    )
    return RepoDownloadProgress(
        repo_id=repo_id,
        repo_revision=revision,
        shard=shard,
        completed_files=len(
            [p for p in file_progress.values() if p.downloaded == p.total]
        ),
        total_files=len(file_progress),
        downloaded_bytes=all_downloaded_bytes,
        downloaded_bytes_this_session=all_downloaded_bytes_this_session,
        total_bytes=all_total_bytes,
        overall_speed=all_speed,
        overall_eta=all_eta,
        status=status,
        file_progress=file_progress,
    )


async def get_weight_map(repo_id: str, revision: str = "main") -> Dict[str, str]:
    target_dir = (await ensure_models_dir()) / str(repo_id).replace("/", "--")
    await aios.makedirs(target_dir, exist_ok=True)
    index_file = await download_file_with_retry(
        repo_id, revision, "model.safetensors.index.json", target_dir
    )
    async with aiofiles.open(index_file, "r") as f:
        index_data = ModelSafetensorsIndex.model_validate_json(await f.read())
    return index_data.weight_map


async def resolve_allow_patterns(shard: ShardMetadata) -> List[str]:
    try:
        weight_map = await get_weight_map(str(shard.model_meta.model_id))
        return get_allow_patterns(weight_map, shard)
    except Exception:
        print(f"Error getting weight map for {shard.model_meta.model_id=}")
        traceback.print_exc()
        return ["*"]


async def get_downloaded_size(path: Path) -> int:
    partial_path = path.with_suffix(path.suffix + ".partial")
    if await aios.path.exists(path):
        return (await aios.stat(path)).st_size
    if await aios.path.exists(partial_path):
        return (await aios.stat(partial_path)).st_size
    return 0


async def download_progress_for_local_path(
    repo_id: str, shard: ShardMetadata, local_path: Path
) -> RepoDownloadProgress:
    # Scan local files for accurate progress reporting
    file_progress: Dict[str, RepoFileDownloadProgress] = {}
    total_files = 0
    total_bytes = 0

    if await aios.path.isdir(local_path):
        # Recursively count files and sizes
        for root, _, files in os.walk(local_path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".pt", ".gguf", ".json")):
                    file_path = Path(root) / f
                    size = (await aios.stat(file_path)).st_size
                    rel_path = str(file_path.relative_to(local_path))
                    file_progress[rel_path] = RepoFileDownloadProgress(
                        repo_id=repo_id,
                        repo_revision="local",
                        file_path=rel_path,
                        downloaded=size,
                        downloaded_this_session=0,
                        total=size,
                        speed=0,
                        eta=timedelta(0),
                        status="complete",
                        start_time=time.time(),
                    )
                    total_files += 1
                    total_bytes += size
    else:
        raise ValueError(f"Local path {local_path} is not a directory")

    return RepoDownloadProgress(
        repo_id=repo_id,
        repo_revision="local",
        shard=shard,
        completed_files=total_files,
        total_files=total_files,
        downloaded_bytes=total_bytes,
        downloaded_bytes_this_session=0,
        total_bytes=total_bytes,
        overall_speed=0,
        overall_eta=timedelta(0),
        status="complete",
        file_progress=file_progress,
    )


async def download_shard(
    shard: ShardMetadata,
    on_progress: Callable[[ShardMetadata, RepoDownloadProgress], None],
    max_parallel_downloads: int = 8,
    skip_download: bool = False,
    allow_patterns: List[str] | None = None,
) -> tuple[Path, RepoDownloadProgress]:
    if not skip_download:
        print(f"Downloading {shard.model_meta.model_id=}")

    # Handle local paths
    if await aios.path.exists(str(shard.model_meta.model_id)):
        print(f"Using local model path {shard.model_meta.model_id}")
        local_path = Path(str(shard.model_meta.model_id))
        return local_path, await download_progress_for_local_path(
            str(shard.model_meta.model_id), shard, local_path
        )

    revision = "main"
    target_dir = await ensure_models_dir() / str(shard.model_meta.model_id).replace(
        "/", "--"
    )
    if not skip_download:
        await aios.makedirs(target_dir, exist_ok=True)

    if not allow_patterns:
        allow_patterns = await resolve_allow_patterns(shard)

    print(f"Downloading {shard.model_meta.model_id=} with {allow_patterns=}")

    all_start_time = time.time()
    # TODO: currently not recursive. Some models might require subdirectories - thus this will need to be changed.
    file_list = await fetch_file_list_with_cache(
        str(shard.model_meta.model_id), revision, recursive=False
    )
    filtered_file_list = list(
        filter_repo_objects(
            file_list, allow_patterns=allow_patterns, key=lambda x: x.path
        )
    )
    file_progress: Dict[str, RepoFileDownloadProgress] = {}

    def on_progress_wrapper(file: FileListEntry, curr_bytes: int, total_bytes: int):
        start_time = (
            file_progress[file.path].start_time
            if file.path in file_progress
            else time.time()
        )
        downloaded_this_session = (
            file_progress[file.path].downloaded_this_session
            + (curr_bytes - file_progress[file.path].downloaded)
            if file.path in file_progress
            else curr_bytes
        )
        speed = (
            downloaded_this_session / (time.time() - start_time)
            if time.time() - start_time > 0
            else 0
        )
        eta = (
            timedelta(seconds=(total_bytes - curr_bytes) / speed)
            if speed > 0
            else timedelta(seconds=0)
        )
        file_progress[file.path] = RepoFileDownloadProgress(
            repo_id=str(shard.model_meta.model_id),
            repo_revision=revision,
            file_path=file.path,
            downloaded=curr_bytes,
            downloaded_this_session=downloaded_this_session,
            total=total_bytes,
            speed=speed,
            eta=eta,
            status="complete" if curr_bytes == total_bytes else "in_progress",
            start_time=start_time,
        )
        on_progress(
            shard,
            calculate_repo_progress(
                shard,
                str(shard.model_meta.model_id),
                revision,
                file_progress,
                all_start_time,
            ),
        )

    for file in filtered_file_list:
        downloaded_bytes = await get_downloaded_size(target_dir / file.path)
        file_progress[file.path] = RepoFileDownloadProgress(
            repo_id=str(shard.model_meta.model_id),
            repo_revision=revision,
            file_path=file.path,
            downloaded=downloaded_bytes,
            downloaded_this_session=0,
            total=file.size or 0,
            speed=0,
            eta=timedelta(0),
            status="complete" if downloaded_bytes == file.size else "not_started",
            start_time=time.time(),
        )

    semaphore = asyncio.Semaphore(max_parallel_downloads)

    async def download_with_semaphore(file: FileListEntry):
        async with semaphore:
            await download_file_with_retry(
                str(shard.model_meta.model_id),
                revision,
                file.path,
                target_dir,
                lambda curr_bytes, total_bytes: on_progress_wrapper(
                    file, curr_bytes, total_bytes
                ),
            )

    if not skip_download:
        await asyncio.gather(
            *[download_with_semaphore(file) for file in filtered_file_list]
        )
    final_repo_progress = calculate_repo_progress(
        shard, str(shard.model_meta.model_id), revision, file_progress, all_start_time
    )
    on_progress(shard, final_repo_progress)
    if gguf := next((f for f in filtered_file_list if f.path.endswith(".gguf")), None):
        return target_dir / gguf.path, final_repo_progress
    else:
        return target_dir, final_repo_progress
