#!/usr/bin/env python3
"""
Fast parallel downloader for a range of safetensors files from zai-org/GLM-5.
Uses aiohttp with 8 concurrent downloads and 8MB chunks (same approach as exo).

Usage:
  python download_glm5_shard.py <start> <end> [--dir GLM-5] [--jobs 8]

Split across 2 Macs:
  Mac 1: python download_glm5_shard.py 1 141
  Mac 2: python download_glm5_shard.py 142 282

Split across 4 Macs:
  Mac 1: python download_glm5_shard.py 1 71
  Mac 2: python download_glm5_shard.py 72 141
  Mac 3: python download_glm5_shard.py 142 212
  Mac 4: python download_glm5_shard.py 213 282
"""

import argparse
import asyncio
import os
import ssl
import sys
import time

import aiofiles
import aiohttp
import certifi

REPO = "zai-org/GLM-5"
TOTAL_SHARDS = 282
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip() or None
    return None


def make_session() -> aiohttp.ClientSession:
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_ctx, limit=0)
    timeout = aiohttp.ClientTimeout(total=1800, connect=60, sock_read=60)
    return aiohttp.ClientSession(connector=conn, timeout=timeout)


class ProgressTracker:
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.start_time = time.monotonic()
        self.lock = asyncio.Lock()
        # per-file tracking: filename -> (downloaded, total)
        self.active: dict[str, tuple[int, int]] = {}

    async def file_skip(self, filename: str) -> None:
        async with self.lock:
            self.skipped_files += 1
            self._render()

    async def file_fail(self, filename: str) -> None:
        async with self.lock:
            self.active.pop(filename, None)
            self.failed_files += 1
            self._render()

    async def file_start(self, filename: str, total: int, resumed: int) -> None:
        async with self.lock:
            self.total_bytes += total
            self.downloaded_bytes += resumed
            self.active[filename] = (resumed, total)
            self._render()

    async def file_progress(self, filename: str, downloaded: int, total: int) -> None:
        async with self.lock:
            prev, _ = self.active.get(filename, (0, total))
            self.downloaded_bytes += downloaded - prev
            self.active[filename] = (downloaded, total)
            self._render()

    async def file_done(self, filename: str) -> None:
        async with self.lock:
            self.active.pop(filename, None)
            self.completed_files += 1
            self._render()

    def _render(self) -> None:
        elapsed = time.monotonic() - self.start_time
        speed = self.downloaded_bytes / elapsed if elapsed > 0 else 0
        done = self.completed_files + self.skipped_files
        remaining_bytes = self.total_bytes - self.downloaded_bytes
        eta = remaining_bytes / speed if speed > 0 else 0

        # Overall progress bar
        pct = self.downloaded_bytes / self.total_bytes * 100 if self.total_bytes else 0
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "=" * filled + ">" * (1 if filled < bar_width else 0) + " " * (bar_width - filled - 1)

        # Active file names (short)
        active_names = []
        for fn, (dl, tot) in sorted(self.active.items()):
            short = fn.replace("model-", "").replace(f"-of-{TOTAL_SHARDS:05d}.safetensors", "")
            file_pct = dl / tot * 100 if tot else 0
            active_names.append(f"{short}:{file_pct:.0f}%")
        active_str = " ".join(active_names[:8])

        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"

        line = (
            f"\r[{bar}] {pct:5.1f}%  "
            f"{done}/{self.total_files} files  "
            f"{self.downloaded_bytes / 1024**3:.1f}/{self.total_bytes / 1024**3:.1f} GB  "
            f"{speed / 1024**2:.1f} MB/s  "
            f"ETA {eta_str}  "
            f"{active_str}"
        )
        # Pad to clear previous line, truncate to terminal width
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 120
        line = line[:cols].ljust(cols)
        sys.stderr.write(line)
        sys.stderr.flush()

    def final_summary(self) -> None:
        elapsed = time.monotonic() - self.start_time
        speed = self.downloaded_bytes / elapsed if elapsed > 0 else 0
        mins, secs = divmod(int(elapsed), 60)
        sys.stderr.write("\n")
        print(
            f"Done: {self.completed_files} downloaded, {self.skipped_files} skipped, "
            f"{self.failed_files} failed. "
            f"{self.downloaded_bytes / 1024**3:.1f} GB in {mins}m{secs:02d}s "
            f"({speed / 1024**2:.1f} MB/s avg)"
        )


async def download_file(
    session: aiohttp.ClientSession,
    filename: str,
    target_dir: str,
    headers: dict[str, str],
    sem: asyncio.Semaphore,
    progress: ProgressTracker,
) -> None:
    async with sem:
        url = f"{HF_ENDPOINT}/{REPO}/resolve/main/{filename}"
        target = os.path.join(target_dir, filename)
        partial = target + ".partial"
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if os.path.exists(target):
            await progress.file_skip(filename)
            return

        resume_pos = 0
        req_headers = dict(headers)
        if os.path.exists(partial):
            resume_pos = os.path.getsize(partial)
            req_headers["Range"] = f"bytes={resume_pos}-"

        async with session.get(url, headers=req_headers) as r:
            if r.status == 416:
                os.rename(partial, target)
                await progress.file_skip(filename)
                return
            if r.status not in (200, 206):
                await progress.file_fail(filename)
                return

            total = int(r.headers.get("Content-Length", 0)) + resume_pos
            downloaded = resume_pos
            await progress.file_start(filename, total, resume_pos)

            async with aiofiles.open(partial, "ab" if resume_pos else "wb") as f:
                while True:
                    chunk = await r.content.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    await f.write(chunk)
                    downloaded += len(chunk)
                    await progress.file_progress(filename, downloaded, total)

        os.rename(partial, target)
        await progress.file_done(filename)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fast parallel GLM-5 shard downloader")
    parser.add_argument("start", type=int, help="First shard number (1-based)")
    parser.add_argument("end", type=int, help="Last shard number (inclusive)")
    parser.add_argument("--dir", default="GLM-5", help="Target directory (default: GLM-5)")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel downloads (default: 8)")
    args = parser.parse_args()

    files = [
        f"model-{i:05d}-of-{TOTAL_SHARDS:05d}.safetensors"
        for i in range(args.start, args.end + 1)
    ]

    headers: dict[str, str] = {"Accept-Encoding": "identity"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Downloading {len(files)} files ({args.start}-{args.end}) to {args.dir}/ with {args.jobs} parallel jobs")

    progress = ProgressTracker(len(files))
    sem = asyncio.Semaphore(args.jobs)
    async with make_session() as session:
        await asyncio.gather(*[
            download_file(session, f, args.dir, headers, sem, progress)
            for f in files
        ])

    progress.final_summary()


if __name__ == "__main__":
    asyncio.run(main())
