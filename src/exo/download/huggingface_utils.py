"""Where the Hugging Face Hub is, and the token to talk to it with."""

import os
from pathlib import Path

import aiofiles
import aiofiles.os as aios


def get_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_home() -> Path:
    """Get the Hugging Face home directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


async def get_hf_token() -> str | None:
    """Retrieve the Hugging Face token from HF_TOKEN env var or HF_HOME directory."""
    # Check environment variable first
    if token := os.environ.get("HF_TOKEN"):
        return token
    # Fall back to file-based token
    token_path = get_hf_home() / "token"
    if await aios.path.exists(token_path):
        async with aiofiles.open(token_path, "r") as f:
            return (await f.read()).strip()
    return None


async def get_auth_headers() -> dict[str, str]:
    """Get authentication headers if a token is available."""
    token = await get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}
