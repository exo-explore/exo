"""Three-tier sampling default resolution: request → instance → cluster → hardcoded.

Each tier supplies optional values; the first non-None wins. Hardcoded defaults
preserve historical Exo behavior when no other tier is set.
"""
import os

# --- Per-cluster defaults (env vars). None = fall through to hardcoded. ---
CLUSTER_DEFAULT_TEMPERATURE: float | None = (
    float(os.environ["EXO_DEFAULT_TEMPERATURE"])
    if os.environ.get("EXO_DEFAULT_TEMPERATURE") else None
)
CLUSTER_DEFAULT_TOP_P: float | None = (
    float(os.environ["EXO_DEFAULT_TOP_P"])
    if os.environ.get("EXO_DEFAULT_TOP_P") else None
)
CLUSTER_DEFAULT_TOP_K: int | None = (
    int(os.environ["EXO_DEFAULT_TOP_K"])
    if os.environ.get("EXO_DEFAULT_TOP_K") else None
)
CLUSTER_DEFAULT_MIN_P: float | None = (
    float(os.environ["EXO_DEFAULT_MIN_P"])
    if os.environ.get("EXO_DEFAULT_MIN_P") else None
)

# --- Hardcoded last-resort fallbacks (existing Exo behavior). ---
HARDCODED_TEMPERATURE: float = 0.7
HARDCODED_TOP_P: float = 1.0
HARDCODED_TOP_K: int = 0
HARDCODED_MIN_P: float = 0.05


def _first_non_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def resolve_sampling(
    *,
    request_temperature: float | None = None,
    request_top_p: float | None = None,
    request_top_k: int | None = None,
    request_min_p: float | None = None,
    instance_temperature: float | None = None,
    instance_top_p: float | None = None,
    instance_top_k: int | None = None,
    instance_min_p: float | None = None,
) -> dict:
    """Return a kwargs dict for `make_sampler(temp, top_p, top_k, min_p)`.

    Resolution order per field: request → instance → cluster env → hardcoded.
    """
    return {
        "temp": _first_non_none(
            request_temperature, instance_temperature,
            CLUSTER_DEFAULT_TEMPERATURE, HARDCODED_TEMPERATURE,
        ),
        "top_p": _first_non_none(
            request_top_p, instance_top_p,
            CLUSTER_DEFAULT_TOP_P, HARDCODED_TOP_P,
        ),
        "top_k": _first_non_none(
            request_top_k, instance_top_k,
            CLUSTER_DEFAULT_TOP_K, HARDCODED_TOP_K,
        ),
        "min_p": _first_non_none(
            request_min_p, instance_min_p,
            CLUSTER_DEFAULT_MIN_P, HARDCODED_MIN_P,
        ),
    }
