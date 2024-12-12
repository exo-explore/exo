from aiohttp import web
import aiohttp_cors
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from exo import DEBUG

async def download_model(model_name, target_dir):
    url = f"http://localhost:52525/models/{model_name}/download"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            current_file = None
            current_writer = None
            
            async for line in response.content:
                try:
                    # Try parsing as metadata
                    metadata = json.loads(line)
                    if current_writer:
                        await current_writer.close()
                    
                    # Setup new file
                    filepath = os.path.join(target_dir, metadata['filename'])
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    current_file = open(filepath, 'wb')
                    current_writer = current_file
                    continue
                except json.JSONDecodeError:
                    pass

                # Check for EOF marker
                if line.strip() == b'EOF':
                    if current_writer:
                        await current_writer.close()
                        current_writer = None
                    continue

                # Write chunk to current file
                if current_writer:
                    current_writer.write(line)