from functools import lru_cache
from pathlib import Path
from vllm.connections import global_http_connection as global_http_connection

VLLM_S3_BUCKET_URL: str

def get_cache_dir() -> Path: ...
@lru_cache
def get_vllm_public_assets(filename: str, s3_prefix: str | None = None) -> Path: ...
