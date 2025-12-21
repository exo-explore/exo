"""
Llama Server Manager for Android.

Manages a llama-server process and provides HTTP-based inference.
This is more reliable than subprocess calls on Android because:
1. llama-server handles all TTY/stdin/stdout issues
2. Provides OpenAI-compatible API with proper streaming
3. Model stays loaded between requests (faster inference)
"""

import os
import signal
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse


# Server configuration
DEFAULT_PORT = 8080
SERVER_STARTUP_TIMEOUT = 120  # seconds - model loading takes time on Android
SERVER_HOST = "127.0.0.1"


def find_llama_server() -> Path | None:
    """Find the llama-server binary."""
    search_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path.home() / "llama.cpp" / "build" / "bin" / "server",
    ]
    
    for path in search_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path
    
    return None


def get_lib_path() -> str:
    """Get LD_LIBRARY_PATH for llama.cpp libraries."""
    lib_dirs = [
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "lib",
    ]
    return ":".join(str(d) for d in lib_dirs if d.exists())


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((SERVER_HOST, port)) == 0


class LlamaServerManager:
    """
    Manages a llama-server instance for inference.
    
    Handles:
    - Starting the server with a specific model
    - Stopping and restarting for model switches
    - Health checks
    - Graceful shutdown
    """
    
    _instance: "LlamaServerManager | None" = None
    
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.current_model: str | None = None
        self.port = DEFAULT_PORT
        self.server_path = find_llama_server()
        self.lib_path = get_lib_path()
        
        if self.server_path is None:
            raise FileNotFoundError(
                "llama-server not found. Please build it:\n"
                "cd ~/llama.cpp && cmake -B build && cmake --build build --target llama-server"
            )
        
        logger.info(f"LlamaServerManager initialized with server: {self.server_path}")
    
    @classmethod
    def get_instance(cls) -> "LlamaServerManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def is_running(self) -> bool:
        """Check if server is running and healthy."""
        if self.process is None:
            return False
        
        # Check if process is still alive
        if self.process.poll() is not None:
            self.process = None
            return False
        
        # Check if server responds to health check
        try:
            response = requests.get(
                f"http://{SERVER_HOST}:{self.port}/health",
                timeout=2
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def start(self, model_path: str) -> bool:
        """
        Start the llama-server with the specified model.
        
        Returns True if server started successfully.
        """
        model_path = str(model_path)
        
        # Check if already running with same model
        if self.is_running() and self.current_model == model_path:
            logger.info(f"Server already running with model: {model_path}")
            return True
        
        # Stop existing server if running with different model
        if self.is_running():
            logger.info(f"Switching model from {self.current_model} to {model_path}")
            self.stop()
        
        # Check if port is in use by something else
        if is_port_in_use(self.port):
            logger.warning(f"Port {self.port} already in use, attempting to kill existing process")
            self._kill_existing_server()
            time.sleep(1)
        
        # Build command
        cmd = [
            str(self.server_path),
            "-m", model_path,
            "--port", str(self.port),
            "--host", SERVER_HOST,
            "-c", "2048",  # Context size
            "-t", "4",     # Threads
            "--no-mmap",   # Important for Android
        ]
        
        # Set up environment
        env = os.environ.copy()
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = self.lib_path
        
        logger.info(f"Starting llama-server: {' '.join(cmd[:6])}...")
        
        try:
            # Start server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
                if self.is_running():
                    self.current_model = model_path
                    logger.info(f"llama-server started successfully on port {self.port}")
                    return True
                
                # Check if process died
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                    logger.error(f"llama-server died during startup: {stderr[:500]}")
                    self.process = None
                    return False
                
                time.sleep(1)
                logger.debug(f"Waiting for server... ({int(time.time() - start_time)}s)")
            
            logger.error(f"llama-server failed to start within {SERVER_STARTUP_TIMEOUT}s")
            self.stop()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False
    
    def stop(self):
        """Stop the llama-server."""
        if self.process is None:
            return
        
        logger.info("Stopping llama-server...")
        
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server didn't terminate gracefully, killing...")
            self.process.kill()
            self.process.wait()
        
        self.process = None
        self.current_model = None
        logger.info("llama-server stopped")
    
    def _kill_existing_server(self):
        """Try to kill any existing llama-server on the port."""
        try:
            # Try to find and kill process using the port
            os.system(f"pkill -f 'llama-server.*--port {self.port}'")
        except Exception:
            pass
    
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate a response using the llama-server API.
        
        Yields text chunks as they're generated.
        """
        if not self.is_running():
            raise RuntimeError("llama-server is not running")
        
        url = f"http://{SERVER_HOST}:{self.port}/v1/chat/completions"
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        logger.debug(f"Sending request to llama-server: {len(messages)} messages")
        
        try:
            if stream:
                # Streaming response
                response = requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=180,
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    
                    try:
                        import json
                        parsed = json.loads(data)
                        content = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
            else:
                # Non-streaming response
                response = requests.post(
                    url,
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    yield content
                    
        except requests.RequestException as e:
            logger.error(f"llama-server request failed: {e}")
            raise


def server_generate(
    model_path: str,
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate text using llama-server.
    
    This is the main entry point for EXO integration.
    Matches the interface of native_generate for compatibility.
    """
    # Get or create server manager
    manager = LlamaServerManager.get_instance()
    
    # Ensure server is running with the right model
    if not manager.start(model_path):
        raise RuntimeError(f"Failed to start llama-server with model: {model_path}")
    
    # Convert EXO messages to OpenAI format
    messages = []
    for msg in task.messages:
        content = msg.content
        if content is None:
            continue
        if isinstance(content, list):
            if len(content) == 0:
                continue
            content = content[0].text if hasattr(content[0], "text") else str(content[0])
        elif hasattr(content, "text"):
            content = content.text
        
        messages.append({
            "role": msg.role,
            "content": str(content),
        })
    
    max_tokens = task.max_tokens or 256
    temperature = task.temperature if task.temperature is not None else 0.7
    
    logger.info(f"Server generation: {len(messages)} messages, max_tokens={max_tokens}")
    
    try:
        token_idx = 0
        full_response = ""
        
        for text in manager.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            full_response += text
            yield GenerationResponse(
                text=text,
                token=token_idx,
                finish_reason=None,
            )
            token_idx += 1
        
        logger.info(f"Server generation complete: {len(full_response)} chars, {token_idx} chunks")
        
        # Final response with finish reason
        yield GenerationResponse(
            text="",
            token=token_idx,
            finish_reason="stop",
        )
        
    except Exception as e:
        logger.error(f"Server generation error: {e}")
        raise


def server_warmup(model_path: str) -> int:
    """
    Warm up by starting the server and loading the model.
    
    This pre-loads the model so first inference is fast.
    """
    logger.info(f"Warming up llama-server with model: {model_path}")
    
    manager = LlamaServerManager.get_instance()
    
    if manager.start(model_path):
        logger.info("llama-server warmup complete - model loaded")
        return 1
    else:
        logger.error("llama-server warmup failed")
        return 0

