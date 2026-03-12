from botocore.client import BaseClient as BaseClient
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

def glob(
    s3: BaseClient | None = None, path: str = "", allow_pattern: list[str] | None = None
) -> list[str]: ...
def list_files(
    s3: BaseClient,
    path: str,
    allow_pattern: list[str] | None = None,
    ignore_pattern: list[str] | None = None,
) -> tuple[str, str, list[str]]: ...
