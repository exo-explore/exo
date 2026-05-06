"""Pluggable model-source detection.

Each :class:`~exo.sources.base.ModelSource` knows how to enumerate locally-installed
models in one external tool (HuggingFace cache, LM Studio, Ollama, llama.cpp) plus
exo's own writable cache. The registry below picks the set of sources to enable for
the current process; the per-worker scanner service walks them on a periodic interval
and emits :class:`~exo.shared.types.events.LocalModelsScanned` events for the UI.

Adding a new source = write one class implementing :class:`ModelSource` and append it
to :func:`default_sources`.
"""

from exo.sources.base import ModelSource
from exo.sources.exo_native import ExoNativeSource
from exo.sources.huggingface import HuggingFaceSource
from exo.sources.llamacpp import LlamaCppSource
from exo.sources.lmstudio import LMStudioSource
from exo.sources.ollama import OllamaSource


def default_sources() -> list[ModelSource]:
    """The full list of sources exo enables by default.

    Each source's ``is_available()`` is consulted by the scanner before invoking
    ``scan()`` — sources whose cache dir does not exist are silently skipped.
    """
    return [
        ExoNativeSource(),
        HuggingFaceSource(),
        LMStudioSource(),
        OllamaSource(),
        LlamaCppSource(),
    ]


__all__ = [
    "ExoNativeSource",
    "HuggingFaceSource",
    "LMStudioSource",
    "LlamaCppSource",
    "ModelSource",
    "OllamaSource",
    "default_sources",
]
