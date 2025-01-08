# Helper functions for pytorch inference
# Some code coming from tinygrad but written towards pytorch

import asyncio
import aiohttp
from tqdm import tqdm
from pathlib import Path
from typing import List

async def fetch_file_async(session, url: str, output_path: Path):
    async with session.get(url) as response:
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            async for chunk in response.content.iter_chunked(8192):
                f.write(chunk)

async def download_files(urls: List[str], output_paths: List[Path]):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, output_path in zip(urls, output_paths):
            tasks.append(fetch_file_async(session, url, output_path))
        
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading files"):
            await f
