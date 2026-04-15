import os


def get_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_endpoints() -> list[str]:
    primary = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    mirror = os.environ.get("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")
    if not mirror or primary == mirror:
        return [primary]
    return [primary, mirror]
