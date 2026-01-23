import asyncio
import hashlib
import os
import shutil
import ssl
import time
import traceback
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urljoin

import aiofiles
import aiofiles.os as aios
import aiohttp
import certifi
from huggingface_hub import (
    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
)
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    TypeAdapter,
)

from exo.download.huggingface_utils import (
    filter_repo_objects,
    get_allow_patterns,
    get_auth_headers,
    get_hf_endpoint,
    get_hf_token,
)
from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadProgressData,
    FileListEntry,
    ModelSafetensorsIndex,
    RepoDownloadProgress,
    RepoFileDownloadProgress,
)
from exo.shared.types.worker.shards import ShardMetadata


class HuggingFaceAuthenticationError(Exception):
    """Raised when HuggingFace returns 401/403 for a model download."""


class ModelStoreFileMetadata(BaseModel):
    etag: str
    size: int

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _model_store_metadata_path(model_dir: Path, rel_path: str) -> Path:
    safe_rel = Path(rel_path)
    if safe_rel.is_absolute() or ".." in safe_rel.parts:
        raise ValueError(f"Invalid relative path: {rel_path}")
    return model_dir / ".exo" / "download_metadata" / f"{safe_rel}.json"


async def _write_model_store_metadata(
    model_dir: Path,
    rel_path: str,
    *,
    etag: str,
    size: int,
) -> None:
    """
    Write per-file metadata used by the cluster model-store server.

    Rationale for error handling:
    - This metadata is an optimization for local-network distribution (it avoids
      expensive per-request hashing on the master).
    - A failure to write metadata must not fail the primary download path, so
      callers are expected to treat failures as best-effort and continue.
    """
    meta_path = _model_store_metadata_path(model_dir, rel_path)
    await aios.makedirs(meta_path.parent, exist_ok=True)
    async with aiofiles.open(meta_path, "w") as f:
        await f.write(ModelStoreFileMetadata(etag=etag, size=size).model_dump_json())


async def _build_auth_error_message(status_code: int, model_id: ModelId) -> str:
    token = await get_hf_token()
    if status_code == 401 and token is None:
        return (
            f"Model '{model_id}' requires authentication. "
            f"Set HF_TOKEN in the app's Advanced settings, set the HF_TOKEN environment variable, or run `hf auth login`. "
            f"Get a token at https://huggingface.co/settings/tokens"
        )
    elif status_code == 403:
        return (
            f"Access denied to '{model_id}'. "
            f"Please accept the model terms at https://huggingface.co/{model_id}"
        )
    else:
        return f"Authentication failed for '{model_id}' (HTTP {status_code})"


def trim_etag(etag: str) -> str:
    if (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"):
        return etag[1:-1]
    return etag


def map_repo_file_download_progress_to_download_progress_data(
    repo_file_download_progress: RepoFileDownloadProgress,
) -> DownloadProgressData:
    return DownloadProgressData(
        downloaded_bytes=repo_file_download_progress.downloaded,
        downloaded_bytes_this_session=repo_file_download_progress.downloaded_this_session,
        total_bytes=repo_file_download_progress.total,
        completed_files=1 if repo_file_download_progress.status == "complete" else 0,
        total_files=1,
        speed=repo_file_download_progress.speed,
        eta_ms=int(repo_file_download_progress.eta.total_seconds() * 1000),
        files={},
    )


def map_repo_download_progress_to_download_progress_data(
    repo_download_progress: RepoDownloadProgress,
) -> DownloadProgressData:
    return DownloadProgressData(
        total_bytes=repo_download_progress.total_bytes,
        downloaded_bytes=repo_download_progress.downloaded_bytes,
        downloaded_bytes_this_session=repo_download_progress.downloaded_bytes_this_session,
        completed_files=repo_download_progress.completed_files,
        total_files=repo_download_progress.total_files,
        speed=repo_download_progress.overall_speed,
        eta_ms=int(repo_download_progress.overall_eta.total_seconds() * 1000),
        files={
            file_path: map_repo_file_download_progress_to_download_progress_data(
                file_progress
            )
            for file_path, file_progress in repo_download_progress.file_progress.items()
        },
    )


def build_model_path(model_id: ModelId) -> DirectoryPath:
    return EXO_MODELS_DIR / model_id.normalize()


async def resolve_model_path_for_repo(model_id: ModelId) -> Path:
    return (await ensure_models_dir()) / model_id.normalize()


async def ensure_models_dir() -> Path:
    await aios.makedirs(EXO_MODELS_DIR, exist_ok=True)
    return EXO_MODELS_DIR


async def delete_model(model_id: ModelId) -> bool:
    model_dir = await ensure_models_dir() / model_id.normalize()
    if not await aios.path.exists(model_dir):
        return False
    await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
    return True


async def seed_models(seed_dir: str | Path):
    """Move model in resources folder of app to .cache/huggingface/hub"""
    source_dir = Path(seed_dir)
    dest_dir = await ensure_models_dir()
    for path in source_dir.iterdir():
        if path.is_dir() and path.name.startswith("models--"):
            dest_path = dest_dir / path.name
            if await aios.path.exists(dest_path):
                logger.info("Skipping moving model to .cache directory")
            else:
                try:
                    await aios.rename(str(path), str(dest_path))
                except Exception:
                    logger.error(f"Error seeding model {path} to {dest_path}")
                    logger.error(traceback.format_exc())


async def fetch_file_list_with_cache(
    model_id: ModelId,
    revision: str = "main",
    recursive: bool = False,
    *,
    endpoint: str | None = None,
) -> list[FileListEntry]:
    target_dir = (await ensure_models_dir()) / "caches" / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    cache_file = target_dir / f"{model_id.normalize()}--{revision}--file_list.json"
    if await aios.path.exists(cache_file):
        async with aiofiles.open(cache_file, "r") as f:
            return TypeAdapter(list[FileListEntry]).validate_json(await f.read())
    file_list = await fetch_file_list_with_retry(
        model_id, revision, recursive=recursive, endpoint=endpoint
    )
    await aios.makedirs(cache_file.parent, exist_ok=True)
    async with aiofiles.open(cache_file, "w") as f:
        await f.write(TypeAdapter(list[FileListEntry]).dump_json(file_list).decode())
    return file_list


async def fetch_file_list_with_retry(
    model_id: ModelId,
    revision: str = "main",
    path: str = "",
    recursive: bool = False,
    *,
    endpoint: str | None = None,
) -> list[FileListEntry]:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            return await _fetch_file_list(
                model_id, revision, path, recursive, endpoint=endpoint
            )
        except HuggingFaceAuthenticationError:
            raise
        except Exception as e:
            if attempt == n_attempts - 1:
                raise e
            await asyncio.sleep(min(8, 0.1 * float(2.0 ** int(attempt))))
    raise Exception(
        f"Failed to fetch file list for {model_id=} {revision=} {path=} {recursive=}"
    )


async def _fetch_file_list(
    model_id: ModelId,
    revision: str = "main",
    path: str = "",
    recursive: bool = False,
    *,
    endpoint: str | None = None,
) -> list[FileListEntry]:
    resolved_endpoint = endpoint or get_hf_endpoint()
    api_url = f"{resolved_endpoint}/api/models/{model_id}/tree/{revision}"
    url = f"{api_url}/{path}" if path else api_url

    headers = await get_download_headers()
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url, headers=headers) as response,
    ):
        if response.status in [401, 403]:
            msg = await _build_auth_error_message(response.status, model_id)
            raise HuggingFaceAuthenticationError(msg)
        if response.status == 200:
            data_json = await response.text()
            data = TypeAdapter(list[FileListEntry]).validate_json(data_json)
            files: list[FileListEntry] = []
            for item in data:
                if item.type == "file":
                    files.append(FileListEntry.model_validate(item))
                elif item.type == "directory" and recursive:
                    subfiles = await _fetch_file_list(
                        model_id, revision, item.path, recursive
                    )
                    files.extend(subfiles)
            return files
        else:
            raise Exception(f"Failed to fetch file list: {response.status}")


async def get_download_headers() -> dict[str, str]:
    return {**(await get_auth_headers()), "Accept-Encoding": "identity"}


def create_http_session(
    auto_decompress: bool = False,
    timeout_profile: Literal["short", "long"] = "long",
) -> aiohttp.ClientSession:
    if timeout_profile == "short":
        total_timeout = 30
        connect_timeout = 10
        sock_read_timeout = 30
        sock_connect_timeout = 10
    else:
        total_timeout = 1800
        connect_timeout = 60
        sock_read_timeout = 1800
        sock_connect_timeout = 60

    ssl_context = ssl.create_default_context(
        cafile=os.getenv("SSL_CERT_FILE") or certifi.where()
    )
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    return aiohttp.ClientSession(
        auto_decompress=auto_decompress,
        connector=connector,
        proxy=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None,
        timeout=aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_read=sock_read_timeout,
            sock_connect=sock_connect_timeout,
        ),
    )


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
    model_id: ModelId,
    revision: str,
    path: str,
    redirected_location: str | None = None,
    *,
    endpoint: str | None = None,
) -> tuple[int, str]:
    resolved_endpoint = endpoint or get_hf_endpoint()
    url = (
        urljoin(f"{resolved_endpoint}/{model_id}/resolve/{revision}/", path)
        if redirected_location is None
        else f"{resolved_endpoint}{redirected_location}"
    )
    headers = await get_download_headers()
    async with (
        create_http_session(timeout_profile="short") as session,
        session.head(url, headers=headers) as r,
    ):
        if r.status == 404:
            raise FileNotFoundError(f"File not found: {url}")
        if r.status == 307:
            # On redirect, only trust Hugging Face's x-linked-* headers.
            x_linked_size = r.headers.get("x-linked-size")
            x_linked_etag = r.headers.get("x-linked-etag")
            if x_linked_size and x_linked_etag:
                content_length = int(x_linked_size)
                etag = trim_etag(x_linked_etag)
                return content_length, etag
            # Otherwise, follow the redirect to get authoritative size/hash
            redirected_location = r.headers.get("location")
            return await file_meta(
                model_id,
                revision,
                path,
                redirected_location,
                endpoint=endpoint,
            )
        if r.status in [401, 403]:
            msg = await _build_auth_error_message(r.status, model_id)
            raise HuggingFaceAuthenticationError(msg)
        content_length = int(
            r.headers.get("x-linked-size") or r.headers.get("content-length") or 0
        )
        etag = r.headers.get("x-linked-etag") or r.headers.get("etag")
        assert content_length > 0, f"No content length for {url}"
        assert etag is not None, f"No remote hash for {url}"
        etag = trim_etag(etag)
        return content_length, etag


async def download_file_with_retry(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    *,
    endpoint: str | None = None,
    retry_on_not_found: bool = False,
) -> Path:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            return await _download_file(
                model_id,
                revision,
                path,
                target_dir,
                on_progress,
                endpoint=endpoint,
            )
        except HuggingFaceAuthenticationError:
            raise
        except Exception as e:
            if isinstance(e, FileNotFoundError) and not retry_on_not_found:
                raise e
            if attempt == n_attempts - 1:
                raise e
            logger.error(
                f"Download error on attempt {attempt}/{n_attempts} for {model_id=} {revision=} {path=} {target_dir=}"
            )
            logger.error(traceback.format_exc())
            await asyncio.sleep(min(8, 0.1 * (2.0**attempt)))
    raise Exception(
        f"Failed to download file {model_id=} {revision=} {path=} {target_dir=}"
    )


async def _download_file(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    *,
    endpoint: str | None = None,
) -> Path:
    if await aios.path.exists(target_dir / path):
        return target_dir / path
    await aios.makedirs((target_dir / path).parent, exist_ok=True)
    length, etag = await file_meta(model_id, revision, path, endpoint=endpoint)
    remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    partial_path = target_dir / f"{path}.partial"
    resume_byte_pos = (
        (await aios.stat(partial_path)).st_size
        if (await aios.path.exists(partial_path))
        else None
    )
    if resume_byte_pos != length:
        resolved_endpoint = endpoint or get_hf_endpoint()
        url = urljoin(f"{resolved_endpoint}/{model_id}/resolve/{revision}/", path)
        headers = await get_download_headers()
        if resume_byte_pos:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        n_read = resume_byte_pos or 0
        async with (
            create_http_session(timeout_profile="long") as session,
            session.get(url, headers=headers) as r,
        ):
            if r.status == 404:
                raise FileNotFoundError(f"File not found: {url}")
            if r.status in [401, 403]:
                msg = await _build_auth_error_message(r.status, model_id)
                raise HuggingFaceAuthenticationError(msg)
            assert r.status in [200, 206], (
                f"Failed to download {path} from {url}: {r.status}"
            )
            async with aiofiles.open(
                partial_path, "ab" if resume_byte_pos else "wb"
            ) as f:
                while chunk := await r.content.read(8 * 1024 * 1024):
                    n_read = n_read + (await f.write(chunk))
                    on_progress(n_read, length, False)

    final_hash = await calc_hash(
        partial_path, hash_type="sha256" if len(remote_hash) == 64 else "sha1"
    )
    integrity = final_hash == remote_hash
    if not integrity:
        try:
            await aios.remove(partial_path)
        except Exception as e:
            logger.error(f"Error removing partial file {partial_path}: {e}")
        raise Exception(
            f"Downloaded file {target_dir / path} has hash {final_hash} but remote hash is {remote_hash}"
        )
    await aios.rename(partial_path, target_dir / path)
    try:
        await _write_model_store_metadata(
            target_dir, path, etag=remote_hash, size=length
        )
    except Exception as e:
        logger.warning(f"Failed to write model store metadata for {path}: {e}")
    on_progress(length, length, True)
    return target_dir / path


def calculate_repo_progress(
    shard: ShardMetadata,
    model_id: ModelId,
    revision: str,
    file_progress: dict[str, RepoFileDownloadProgress],
    all_start_time: float,
) -> RepoDownloadProgress:
    all_total_bytes = sum((p.total.in_bytes for p in file_progress.values()), 0)
    all_downloaded_bytes = sum(
        (p.downloaded.in_bytes for p in file_progress.values()), 0
    )
    all_downloaded_bytes_this_session = sum(
        (p.downloaded_this_session.in_bytes for p in file_progress.values()), 0
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
        repo_id=model_id,
        repo_revision=revision,
        shard=shard,
        completed_files=len(
            [p for p in file_progress.values() if p.downloaded == p.total]
        ),
        total_files=len(file_progress),
        downloaded_bytes=Memory.from_bytes(all_downloaded_bytes),
        downloaded_bytes_this_session=Memory.from_bytes(
            all_downloaded_bytes_this_session
        ),
        total_bytes=Memory.from_bytes(all_total_bytes),
        overall_speed=all_speed,
        overall_eta=all_eta,
        status=status,
        file_progress=file_progress,
    )


async def get_weight_map(model_id: ModelId, revision: str = "main") -> dict[str, str]:
    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)

    index_files_dir = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        allow_patterns="*.safetensors.index.json",
    )

    index_files = list(Path(index_files_dir).glob("**/*.safetensors.index.json"))

    weight_map: dict[str, str] = {}

    for index_file in index_files:
        relative_dir = index_file.parent.relative_to(index_files_dir)

        async with aiofiles.open(index_file, "r") as f:
            index_data = ModelSafetensorsIndex.model_validate_json(await f.read())

            if relative_dir != Path("."):
                prefixed_weight_map = {
                    f"{relative_dir}/{key}": str(relative_dir / value)
                    for key, value in index_data.weight_map.items()
                }
                weight_map = weight_map | prefixed_weight_map
            else:
                weight_map = weight_map | index_data.weight_map

    return weight_map


async def resolve_allow_patterns(shard: ShardMetadata) -> list[str]:
    # TODO: 'Smart' downloads are disabled because:
    #  (i) We don't handle all kinds of files;
    # (ii) We don't have sticky sessions.
    # (iii) Tensor parallel requires all files.
    return ["*"]
    try:
        weight_map = await get_weight_map(str(shard.model_card.model_id))
        return get_allow_patterns(weight_map, shard)
    except Exception:
        logger.error(f"Error getting weight map for {shard.model_card.model_id=}")
        logger.error(traceback.format_exc())
        return ["*"]


def is_image_model(shard: ShardMetadata) -> bool:
    tasks = shard.model_card.tasks
    return ModelTask.TextToImage in tasks or ModelTask.ImageToImage in tasks


async def get_downloaded_size(path: Path) -> int:
    partial_path = path.with_suffix(path.suffix + ".partial")
    if await aios.path.exists(path):
        return (await aios.stat(path)).st_size
    if await aios.path.exists(partial_path):
        return (await aios.stat(partial_path)).st_size
    return 0


async def download_shard(
    shard: ShardMetadata,
    on_progress: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    max_parallel_downloads: int = 8,
    skip_download: bool = False,
    allow_patterns: list[str] | None = None,
    *,
    endpoint: str | None = None,
    retry_on_not_found: bool = False,
) -> tuple[Path, RepoDownloadProgress]:
    if not skip_download:
        logger.debug(f"Downloading {shard.model_card.model_id=}")

    revision = "main"
    target_dir = await ensure_models_dir() / str(shard.model_card.model_id).replace(
        "/", "--"
    )
    if not skip_download:
        await aios.makedirs(target_dir, exist_ok=True)

    if not allow_patterns:
        allow_patterns = await resolve_allow_patterns(shard)

    if not skip_download:
        logger.debug(f"Downloading {shard.model_card.model_id=} with {allow_patterns=}")

    all_start_time = time.time()
    file_list = await fetch_file_list_with_cache(
        shard.model_card.model_id, revision, recursive=True, endpoint=endpoint
    )
    filtered_file_list = list(
        filter_repo_objects(
            file_list, allow_patterns=allow_patterns, key=lambda x: x.path
        )
    )

    # For image models, skip root-level safetensors files since weights
    # are stored in component subdirectories (e.g., transformer/, vae/)
    if is_image_model(shard):
        filtered_file_list = [
            f
            for f in filtered_file_list
            if "/" in f.path or not f.path.endswith(".safetensors")
        ]
    file_progress: dict[str, RepoFileDownloadProgress] = {}

    async def on_progress_wrapper(
        file: FileListEntry, curr_bytes: int, total_bytes: int, is_renamed: bool
    ) -> None:
        start_time = (
            file_progress[file.path].start_time
            if file.path in file_progress
            else time.time()
        )
        downloaded_this_session = (
            file_progress[file.path].downloaded_this_session.in_bytes
            + (curr_bytes - file_progress[file.path].downloaded.in_bytes)
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
            repo_id=shard.model_card.model_id,
            repo_revision=revision,
            file_path=file.path,
            downloaded=Memory.from_bytes(curr_bytes),
            downloaded_this_session=Memory.from_bytes(downloaded_this_session),
            total=Memory.from_bytes(total_bytes),
            speed=speed,
            eta=eta,
            status="complete"
            if curr_bytes == total_bytes and is_renamed
            else "in_progress",
            start_time=start_time,
        )
        await on_progress(
            shard,
            calculate_repo_progress(
                shard,
                shard.model_card.model_id,
                revision,
                file_progress,
                all_start_time,
            ),
        )

    for file in filtered_file_list:
        downloaded_bytes = await get_downloaded_size(target_dir / file.path)
        file_progress[file.path] = RepoFileDownloadProgress(
            repo_id=shard.model_card.model_id,
            repo_revision=revision,
            file_path=file.path,
            downloaded=Memory.from_bytes(downloaded_bytes),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_bytes(file.size or 0),
            speed=0,
            eta=timedelta(0),
            status="complete" if downloaded_bytes == file.size else "not_started",
            start_time=time.time(),
        )

    semaphore = asyncio.Semaphore(max_parallel_downloads)

    def schedule_progress(
        file: FileListEntry, curr_bytes: int, total_bytes: int, is_renamed: bool
    ) -> None:
        asyncio.create_task(
            on_progress_wrapper(file, curr_bytes, total_bytes, is_renamed)
        )

    async def download_with_semaphore(file: FileListEntry) -> None:
        async with semaphore:
            await download_file_with_retry(
                shard.model_card.model_id,
                revision,
                file.path,
                target_dir,
                lambda curr_bytes, total_bytes, is_renamed: schedule_progress(
                    file, curr_bytes, total_bytes, is_renamed
                ),
                endpoint=endpoint,
                retry_on_not_found=retry_on_not_found,
            )

    if not skip_download:
        await asyncio.gather(
            *[download_with_semaphore(file) for file in filtered_file_list]
        )
    final_repo_progress = calculate_repo_progress(
        shard, shard.model_card.model_id, revision, file_progress, all_start_time
    )
    await on_progress(shard, final_repo_progress)
    if gguf := next((f for f in filtered_file_list if f.path.endswith(".gguf")), None):
        return target_dir / gguf.path, final_repo_progress
    else:
        return target_dir, final_repo_progress
