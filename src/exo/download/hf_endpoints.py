import os

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _normalize(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        return endpoint
    if "://" not in endpoint:
        endpoint = "https://" + endpoint
    return endpoint


def get_hf_endpoint() -> str:
    return _normalize(os.environ.get("HF_ENDPOINT") or _DEFAULT_HF_ENDPOINT)


def get_hf_mirror_endpoint() -> str | None:
    mirror = os.environ.get("HF_MIRROR_ENDPOINT", _DEFAULT_MIRROR_ENDPOINT)
    if not mirror:
        return None
    return _normalize(mirror)


def get_hf_endpoints() -> list[str]:
    primary = get_hf_endpoint()
    mirror = get_hf_mirror_endpoint()
    endpoints = [primary]
    if mirror is not None and mirror != primary:
        endpoints.append(mirror)
    return endpoints
