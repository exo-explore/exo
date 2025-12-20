"""
llama.cpp inference engine for exo.

This engine provides cross-platform LLM inference using llama-cpp-python,
enabling exo to run on Android/Termux, Linux, and other non-Apple platforms.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama

__all__ = ["Llama"]

