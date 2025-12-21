"""
Native llama-cli wrapper for Android.

Uses the native llama.cpp CLI binary via subprocess instead of Python bindings.
This is more reliable on Android where the Python bindings may segfault.
"""

import os
import subprocess
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

from loguru import logger

from exo.shared.types.api import FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse


def find_llama_cli() -> Path | None:
    """Find the llama-cli binary (the one we tested and works)."""
    search_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]
    
    for path in search_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path
    
    return None


def get_lib_path() -> str:
    """Get the LD_LIBRARY_PATH for llama.cpp libraries."""
    lib_dirs = [
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "lib",
    ]
    return ":".join(str(d) for d in lib_dirs if d.exists())


class NativeLlamaCpp:
    """Wrapper around native llama-cli binary."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        
        self.cli_path = find_llama_cli()
        if self.cli_path is None:
            raise FileNotFoundError(
                "llama-cli not found. Please build llama.cpp first:\n"
                "cd ~/llama.cpp && cmake -B build && cmake --build build"
            )
        
        self.lib_path = get_lib_path()
        logger.info(f"Using native llama-cli: {self.cli_path}")
        logger.info(f"LD_LIBRARY_PATH: {self.lib_path}")
        
        # Verify model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Generate text using llama-cli."""
        
        # Use llama-cli with prompt and immediate exit after generation
        # Flags for non-interactive batch mode:
        # --no-display-prompt: Don't echo the prompt back
        # --log-disable: Disable verbose model loading logs
        cmd = [
            str(self.cli_path),
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--temp", str(temperature),
            "--no-mmap",
            "--no-display-prompt",
            "--log-disable",
        ]
        
        env = os.environ.copy()
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = self.lib_path
        
        logger.info(f"Running: {' '.join(cmd[:8])}...")
        logger.info(f"Full command: {' '.join(cmd)}")
        
        try:
            # Run the command with stdin closed to ensure it doesn't wait for input
            # Using input="" sends empty stdin and closes it
            result = subprocess.run(
                cmd,
                input="",  # Close stdin immediately
                capture_output=True,
                text=True,
                env=env,
                timeout=180,  # Increase timeout for slow Android
            )
            
            full_output = result.stdout
            stderr_output = result.stderr
            
            logger.info(f"llama exit code: {result.returncode}")
            logger.info(f"stdout len: {len(full_output)}, stderr len: {len(stderr_output)}")
            
            if result.returncode != 0:
                logger.warning(f"llama stderr: {stderr_output[:500]}")
            
            # With --no-display-prompt, output should be just the generated text
            response = full_output.strip()
            
            # Clean up any special tokens that might appear
            for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|assistant|>"]:
                response = response.replace(token, "")
            
            response = response.strip()
            
            logger.info(f"Raw output preview: '{full_output[:200]}'")
            
            if response:
                logger.info(f"Extracted response: '{response[:100]}'")
                yield response
            else:
                logger.warning(f"Empty response. stderr: {stderr_output[:300]}")
                
        except subprocess.TimeoutExpired:
            logger.error("llama-cli timed out after 120s")
        except Exception as e:
            logger.error(f"Error running llama: {e}")


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format messages into a chat prompt."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def native_generate(
    model_path: str,
    task: ChatCompletionTaskParams,
    n_ctx: int = 2048,
    n_threads: int = 4,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate text using native llama-cli.
    Matches the interface of llamacpp_generate for compatibility.
    """
    # Extract the last user message - llama-cli works best with simple prompts
    user_message = ""
    for msg in reversed(task.messages):
        content = msg.content
        if content is None:
            continue
        if isinstance(content, list):
            if len(content) == 0:
                continue
            content = content[0].text
        elif hasattr(content, "text"):
            content = content.text
        
        if msg.role == "user":
            user_message = str(content)
            break
    
    # Use just the user message - llama-cli will use the model's chat template
    prompt = user_message
    
    max_tokens = task.max_tokens or 256
    temperature = task.temperature if task.temperature is not None else 0.7
    top_p = task.top_p if task.top_p is not None else 0.9
    
    logger.info(f"Native generation with prompt='{prompt}', max_tokens={max_tokens}")
    
    try:
        cli = NativeLlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
        
        logger.info("Starting llama-cli subprocess...")
        token_idx = 0
        response_count = 0
        
        for text in cli.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            response_count += 1
            logger.info(f"Got response chunk {response_count}: '{text[:50]}...'")
            yield GenerationResponse(
                text=text,
                token=token_idx,
                finish_reason=None,
            )
            token_idx += 1
        
        logger.info(f"Generation complete, yielded {response_count} chunks")
        
        # Final response with finish reason
        yield GenerationResponse(
            text="",
            token=token_idx,
            finish_reason="stop",
        )
        logger.info("Sent final stop signal")
        
    except Exception as e:
        logger.error(f"Native generation error: {e}")
        raise


def native_warmup(model_path: str, n_ctx: int = 512, n_threads: int = 4) -> int:
    """
    Warm up by verifying the binary exists and is executable.
    
    Note: We don't actually generate tokens during warmup for native CLI mode
    because each call to llama-cli loads the model fresh (no persistent state).
    The first real generation will be the warmup.
    """
    logger.info("Warming up native llama-cli (verification only)")
    
    try:
        cli_path = find_llama_cli()
        if cli_path is None:
            raise FileNotFoundError("llama-cli/llama-simple not found")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Native warmup complete: {cli_path} ready, model exists")
        return 1  # Indicate success
        
    except Exception as e:
        logger.warning(f"Native warmup failed: {e}")
        return 0

