"""Warn-on-extra-fields mixin for public API request models.

Pydantic v2 defaults to ``extra="ignore"`` for ``BaseModel``, which silently
drops unknown fields. For internal data structures this is fine, but for
public API request bodies it makes mistyped or out-of-spec fields invisible:
the request is accepted, the offending field is dropped, and the request
runs with whatever defaults the matching declared fields had.

In practice this has bitten us at the placement layer (``/place_instance``):
a mistyped ``instance_meta`` field was silently dropped, falling back to the
default ``MlxRing`` instead of the requested ``MlxJaccl``, and the only
symptom was lower decode throughput. No warning, no log line, no 4xx.

This module exposes :class:`WarnExtraModel`, a drop-in base class that keeps
``extra="ignore"`` semantics (so the change is non-breaking for existing
clients) but logs a one-line warning the first time a given
``(model_name, unknown_field)`` pair is seen, rate-limited thereafter to
avoid log-spam from repeated requests.

A future change can flip ``extra="forbid"`` once the warning has had time
to surface miswired clients in the wild.
"""

from threading import Lock
from time import monotonic
from typing import Any

from loguru import logger
from pydantic import AliasChoices, AliasPath, BaseModel, model_validator

# Default rate-limit window for repeat warnings of the same
# (model_name, field_name) pair. Chosen to be long enough that a hot loop
# of mistyped requests does not flood the log, but short enough that the
# warning re-surfaces after operator attention has likely moved on.
_DEFAULT_RATE_LIMIT_SECONDS: float = 60.0

# (class_name, field_name) -> last-emitted monotonic timestamp.
_last_warned: dict[tuple[str, str], float] = {}
_lock = Lock()


def _should_emit(key: tuple[str, str], now: float, window: float) -> bool:
    """Return True if a warning for ``key`` should be emitted now.

    Rate-limited per key with a fixed window. Threadsafe; intended to be
    called from request validation paths that may run concurrently under
    asyncio + thread-pool offloading.
    """
    with _lock:
        last = _last_warned.get(key)
        if last is not None and (now - last) < window:
            return False
        _last_warned[key] = now
        return True


def _reset_rate_limit_state() -> None:
    """Test helper: clear the rate-limit cache.

    Not part of the public API. Tests use this to assert per-key behavior
    without coupling to wall-clock timing.
    """
    with _lock:
        _last_warned.clear()


class WarnExtraModel(BaseModel):
    """Base class for public API request models that warns on extra fields.

    Subclasses keep the default ``extra="ignore"`` behavior — unknown keys
    are still dropped — but a warning is logged when this happens, rate
    limited to once per ``(model_name, field_name)`` pair per
    ``_DEFAULT_RATE_LIMIT_SECONDS``.

    Why not ``extra="forbid"``?
        Existing clients may already be sending fields we silently drop
        (whether a typo or a future-spec field they expect us to ignore).
        Flipping to ``forbid`` is a breaking change. Warning gives us a
        deprecation path: surface the problem in logs, then forbid in a
        follow-up.
    """

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, data: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]
        # Validators see whatever the caller passed. For public API requests
        # this is normally a dict from JSON, but pydantic also dispatches
        # this hook for nested model instances and other types — those are
        # not the case we care about, so bail cleanly.
        if not isinstance(data, dict):
            return data

        known: set[str] = set()
        for name, field in cls.model_fields.items():
            known.add(name)
            if field.alias is not None:
                known.add(field.alias)
            va = field.validation_alias
            if isinstance(va, str):
                known.add(va)
            elif isinstance(va, AliasChoices):
                for choice in va.choices:
                    if isinstance(choice, str):
                        known.add(choice)
                    elif isinstance(choice, AliasPath):
                        # AliasPath targets a nested location; the top-level
                        # key it reads from is the first path element.
                        first = choice.path[0] if choice.path else None
                        if isinstance(first, str):
                            known.add(first)
            elif isinstance(va, AliasPath):
                first = va.path[0] if va.path else None
                if isinstance(first, str):
                    known.add(first)

        unknown = [k for k in data if k not in known]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        if not unknown:
            return data

        now = monotonic()
        cls_name = cls.__name__
        for key in unknown:
            rate_key = (cls_name, str(key))
            if _should_emit(rate_key, now, _DEFAULT_RATE_LIMIT_SECONDS):
                logger.warning(
                    "Dropping unknown field {field!r} on {model} request — "
                    "this field is not declared on the model and will be "
                    "ignored. Check for a typo or an out-of-spec field. "
                    "(Subsequent occurrences of this same field on this "
                    "model are rate-limited to once per {window:.0f}s.)",
                    field=key,
                    model=cls_name,
                    window=_DEFAULT_RATE_LIMIT_SECONDS,
                )

        return data
