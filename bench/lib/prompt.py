"""Typed prompt-sizing utilities for benchmarks.

Wraps the HuggingFace ``transformers`` tokenizer (a fundamentally dynamic
object — different models return different types from
``apply_chat_template``) behind a small typed API so the rest of the bench
library can stay strict-typed.

``PromptSizer.build(target)`` returns a ``(content, exact_token_count)``
pair. Internally it:
  1. Tokenises the empty user message to learn the chat-template overhead
     (``base_tokens``).
  2. Estimates tokens-per-atom from a 100-atom sample.
  3. Binary-searches over the atom count so the resulting message
     tokenises to *exactly* ``target`` tokens.

Callers downstream (``run_one_completion`` etc.) receive the verified
token count, so analysis can confirm the prompt hit its target.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast


def _coerce_token_ids(raw: object) -> list[int]:
    """Normalise ``apply_chat_template`` output to a flat list of token ids.

    transformers' ``apply_chat_template`` may return:
      - ``list[int]`` (slow tokenizers, ``tokenize=True``)
      - a ``BatchEncoding`` with ``.input_ids`` (fast tokenizers)
      - a tensor wrapped object (some models)

    We only need ``len(.)`` of the result, so we just need to flatten to a
    list and return it.
    """
    if isinstance(raw, list):
        return cast("list[int]", raw)
    input_ids = getattr(raw, "input_ids", None)
    if isinstance(input_ids, list):
        return cast("list[int]", input_ids)
    raise TypeError(
        f"Unsupported tokenizer output type {type(raw).__name__}; "
        "expected list[int] or BatchEncoding-like with .input_ids."
    )


def _build_token_counter(tokenizer: object) -> Callable[[str], int]:
    """Return a closure that counts tokens for a user message.

    Tries ``apply_chat_template`` first; falls back to the DeepSeek-V4
    Python encoder for models that don't ship a Jinja chat template.
    """
    apply_chat_template = cast(
        Callable[..., object],
        tokenizer.apply_chat_template,  # type: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    encode = cast(
        Callable[..., list[int]],
        tokenizer.encode,  # type: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )

    def count_fn(user_content: str) -> int:
        messages = [{"role": "user", "content": user_content}]
        try:
            raw = apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except ValueError:
            # Models without a Jinja chat template (e.g. DeepSeek V4 which
            # ships its own Python encoder). Use the exo-side V4 encoder.
            from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (  # type: ignore[reportMissingTypeStubs]
                encode_messages as encode_v4,
            )

            prompt = cast(str, encode_v4(messages, thinking_mode="thinking"))  # type: ignore[reportUnknownArgumentType]
            raw = encode(prompt, add_special_tokens=False)
        return len(_coerce_token_ids(raw))

    return count_fn


class PromptSizer:
    """Build a chat-completion content string of an exact token length."""

    DEFAULT_ATOM: Final[str] = "a "

    def __init__(self, tokenizer: object, atom: str = DEFAULT_ATOM):
        self._tokenizer = tokenizer
        self.atom = atom
        self._count_fn = _build_token_counter(tokenizer)
        self.base_tokens = self._count_fn("")

    def count(self, content: str) -> int:
        """Return the token count for ``content`` after chat-template expansion."""
        return self._count_fn(content)

    def build(self, target_prompt_tokens: int) -> tuple[str, int]:
        """Return ``(content, exact_token_count)`` summing to ``target``.

        Raises ``RuntimeError`` if the chosen ``atom`` overshoots the target
        (try a different atom — see ``DEFAULT_ATOM``).
        """
        target = int(target_prompt_tokens)
        if target < self.base_tokens:
            raise RuntimeError(
                f"Target ({target}) is smaller than template overhead "
                f"({self.base_tokens})."
            )

        # Estimate tokens per atom using a sample.
        sample_count = 100
        sample_tokens = self._count_fn(self.atom * sample_count) - self.base_tokens
        tokens_per_atom = sample_tokens / sample_count
        needed_tokens = target - self.base_tokens
        estimated_atoms = int(needed_tokens / tokens_per_atom)

        # Binary search to find exact atom count.
        low, high = 0, estimated_atoms * 2 + 100
        while low < high:
            mid = (low + high) // 2
            if self._count_fn(self.atom * mid) < target:
                low = mid + 1
            else:
                high = mid

        content = self.atom * low
        actual = self._count_fn(content)
        if actual != target:
            raise RuntimeError(
                f"Overshot: got {actual} tokens (target {target}). "
                f"Pick a different atom (try ' a' or '\\n' or '0 ')."
            )
        return content, actual


def _load_kimi_tokenizer(model_id: str) -> object:
    """Special-case Kimi K2's custom TikTokenTokenizer (transformers 5.x quirk)."""
    from huggingface_hub import (
        snapshot_download,  # type: ignore[reportUnknownVariableType]
    )

    raw_path = snapshot_download(
        model_id,
        allow_patterns=[
            "*.json",
            "*.py",
            "*.tiktoken",
            "*.model",
            "*.jinja",
        ],
        dry_run=False,
    )
    model_path = Path(raw_path)
    sys.path.insert(0, str(model_path))

    tool_decl_path = model_path / "tool_declaration_ts.py"
    if tool_decl_path.exists():
        spec = importlib.util.spec_from_file_location(
            "tool_declaration_ts", tool_decl_path
        )
        if spec is not None and spec.loader is not None:
            tool_decl_module = importlib.util.module_from_spec(spec)
            sys.modules["tool_declaration_ts"] = tool_decl_module
            spec.loader.exec_module(tool_decl_module)

    tok_path = model_path / "tokenization_kimi.py"
    source = tok_path.read_text().replace(
        "from .tool_declaration_ts", "from tool_declaration_ts"
    )
    tok_module = types.ModuleType("tokenization_kimi")
    tok_module.__file__ = str(tok_path)
    sys.modules["tokenization_kimi"] = tok_module
    exec(compile(source, str(tok_path), "exec"), tok_module.__dict__)  # noqa: S102

    tik_token_cls = cast(Any, tok_module).TikTokenTokenizer  # type: ignore[reportAny]
    hf_tokenizer = cast(Any, tik_token_cls.from_pretrained(model_path))  # type: ignore[reportAny]

    # Patch encode to use internal tiktoken model directly (transformers 5.x
    # bug in the encode→pad path for slow tokenizers).
    def _patched_encode(text: str, **_kwargs: object) -> list[int]:
        return list(
            hf_tokenizer.model.encode(text, allowed_special="all")  # type: ignore[reportAny, reportUnknownMemberType]
        )

    hf_tokenizer.encode = _patched_encode
    return cast(object, hf_tokenizer)


def load_tokenizer_for_bench(model_id: str) -> object:
    """Load a HuggingFace tokenizer with bench-specific compatibility shims.

    Returns the tokenizer as ``object`` because transformers' types are
    fundamentally dynamic (concrete class depends on the model). Callers
    should pass the result straight to :class:`PromptSizer`.
    """
    # Monkey-patch for transformers 5.x: Kimi's tokenization_kimi.py imports
    # bytes_to_unicode from gpt2_tokenization which moved.
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
            gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[reportAttributeAccessIssue]
    except ImportError:
        pass

    if "kimi-k2" in model_id.lower():
        return _load_kimi_tokenizer(model_id)

    from transformers import AutoTokenizer

    try:
        return cast(
            object,
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),  # type: ignore[reportUnknownMemberType]
        )
    except (AttributeError, ValueError):
        # Some models ship a Jinja template / encoder that AutoTokenizer
        # can't introspect from HF directly — download artefacts and load
        # from the local snapshot path.
        from huggingface_hub import (
            snapshot_download,  # type: ignore[reportUnknownVariableType]
        )
        from transformers import PretrainedConfig

        raw_full_path = snapshot_download(
            model_id,
            allow_patterns=[
                "*.json",
                "*.py",
                "tokenizer.model",
                "*.tiktoken",
                "tiktoken.model",
                "*.txt",
                "*.jsonl",
                "*.jinja",
            ],
            dry_run=False,
        )
        model_path = Path(raw_full_path)
        stub_kwargs: dict[str, Any] = {}
        config_file = model_path / "config.json"
        if config_file.exists():
            with config_file.open() as f:
                raw_config: dict[str, Any] = json.load(f)  # type: ignore[reportAny]
            for key in (
                "model_type",
                "max_position_embeddings",
                "vocab_size",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
            ):
                if key in raw_config:
                    stub_kwargs[key] = raw_config[key]
        return cast(
            object,
            AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
                str(model_path),
                config=PretrainedConfig(**stub_kwargs),  # type: ignore[reportArgumentType, reportAny]
                trust_remote_code=True,
            ),
        )
