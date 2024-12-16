from aiohttp import web
import aiohttp_cors
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from exo import DEBUG

class LocalModelAPI:
    def __init__(self):
        self.app = web.Application()
        self.cache_dir = os.path.expanduser('~/.cache/exo')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
        # Setup CORS
        cors = aiohttp_cors.setup(self.app)
        cors_options = aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )

        # Add routes with CORS
        routes = [
            ('GET', '/health', self.health_check),
            ('GET', '/files/{filename}', self.handle_download_file),
            ('GET', '/files', self.handle_list_files),
            ('GET', '/models', self.handle_list_models),
            ('GET', '/models/{foldername:.*}/list', self.handle_list_model_items),
            ('GET', '/models/{model_name}/download', self.handle_model_download)
        ]

        for method, path, handler in routes:
            cors.add(self.app.router.add_route(method, path, handler), {"*": cors_options})

        # Add middleware
        self.app.middlewares.append(self.log_request)

    async def log_request(self, app, handler):
        async def middleware(request):
            if DEBUG >= 2:
                print(f"Received request: {request.method} {request.path}")
            return await handler(request)
        return middleware

    async def handle_download_file(self, request):
        filename = request.match_info['filename']
        file_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(file_path):
            return web.json_response({"error": "File not found"}, status=404)
        
        return web.FileResponse(file_path)

    async def handle_list_files(self, request):
        files = []
        for entry in os.scandir(self.cache_dir):
            if entry.is_file() and not entry.name.startswith('.'):
                stat = entry.stat()
                files.append({
                    'name': entry.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        return web.json_response({'files': files})

    async def handle_model_download(self, request):
        model_name = request.match_info['model_name']
        chunk_size = 32768  # 32KB chunks for better performance
        model_path = os.path.join(self.cache_dir, model_name)

        if not os.path.exists(model_path):
            return web.json_response({"error": "Model not found"}, status=404)

        if not os.path.isdir(model_path):
            return web.json_response({"error": "Not a valid model directory"}, status=400)

        # Setup streaming response
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'application/octet-stream'
            }
        )
        await response.prepare(request)

        # Walk through directory and stream files
        for root, _, files in os.walk(model_path):
            for filename in files:
                if filename.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, model_path)
                
                if DEBUG >= 2:
                    print(f"Streaming file: {rel_path}")

                # Send file metadata
                metadata = {
                    'filename': rel_path,
                    'size': os.path.getsize(file_path)
                }
                await response.write(json.dumps(metadata).encode() + b'\n')

                # Stream file content in chunks
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        await response.write(chunk)
                
                # Send end of file marker
                await response.write(b'EOF\n')

        await response.write_eof()
        return response

    async def handle_list_models(self, request):
        folders = []
        for entry in os.scandir(self.cache_dir):
            if entry.is_dir() and not entry.name.startswith('.'):
                folder_size = sum(
                    f.stat().st_size for f in Path(entry.path).rglob('*') if f.is_file()
                )
                folders.append({
                    'name': entry.name,
                    'size': folder_size,
                    'modified': datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                })
        return web.json_response({'models': folders})

    async def handle_list_model_items(self, request):
        # Get the full path including potential subfolders from URL
        path_parts = request.match_info['foldername'].split('/')
        folder_path = os.path.join(self.cache_dir, *path_parts)
        
        if not os.path.exists(folder_path):
            return web.json_response({"error": "Folder not found"}, status=404)
            
        if not os.path.isdir(folder_path):
            return web.json_response({"error": "Not a valid folder"}, status=400)

        items = []
        # List only immediate contents of current directory
        for entry in os.scandir(folder_path):
            if entry.name.startswith('.'):
                continue
                
            if entry.is_dir():
                items.append({
                    'type': 'directory',
                    'name': entry.name,
                    'modified': datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                })
            else:
                items.append({
                    'type': 'file',
                    'name': entry.name,
                    'size': entry.stat().st_size,
                    'modified': datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                })
        
        return web.json_response({'items': items})


    async def health_check(self, request):
        """健康检查端点"""
        return web.Response(text='OK', status=200)

    async def run(self, host: str = "0.0.0.0", port: int = 52525):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
