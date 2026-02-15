import sys
from pathlib import Path
from typing import Any

from exo.shared.models.model_cards import ModelId


def _apply_transformers_5x_compact() -> None:
    """Monkey patch for transformers 5.x compatibility (Kimi tokenizer)."""
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_tok
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        if not hasattr(gpt2_tok, "bytes_to_unicode"):
            gpt2_tok.bytes_to_unicode = bytes_to_unicode  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError:
        pass


def load_tokenizer_for_model(model_id: ModelId, model_path: Path) -> Any:  # pyright: ignore[reportAny]
    model_id_lower = model_id.lower()

    if "kimi-k2" in model_id_lower:
        _apply_transformers_5x_compact()
        sys.path.insert(0, str(model_path))

        from tokenization_kimi import (  # pyright: ignore[reportMissingImports]
            TikTokenTokenizer,  # pyright: ignore[reportUnknownVariableType]
        )
        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return hf_tokenizer  # pyright: ignore[reportUnknownVariableType]

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
