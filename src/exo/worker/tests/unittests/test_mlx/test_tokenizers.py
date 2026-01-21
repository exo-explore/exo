"""
Unit tests for tokenizer loading and functionality across all supported models.

This test downloads only tokenizer-related files (not full model weights) to verify
that tokenizers can be loaded and used correctly for encoding/decoding.
"""

import asyncio
import contextlib
from pathlib import Path

import pytest

from exo.shared.models.model_cards import MODEL_CARDS, ModelCard, ModelId
from exo.worker.download.download_utils import (
    download_file_with_retry,
    ensure_models_dir,
    fetch_file_list_with_cache,
)
from exo.worker.engines.mlx.utils_mlx import (
    get_eos_token_ids_for_model,
    load_tokenizer_for_model_id,
)

# Files needed for tokenizer functionality
TOKENIZER_FILE_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tiktoken.model",
    "added_tokens.json",
    "tokenizer.model",
    "tokenization_*.py",  # Custom tokenizer implementations
]


def is_tokenizer_file(filename: str) -> bool:
    """Check if a file is needed for tokenizer functionality."""
    for pattern in TOKENIZER_FILE_PATTERNS:
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            suffix = pattern.split("*")[1]
            if filename.startswith(prefix) and filename.endswith(suffix):
                return True
        elif filename == pattern:
            return True
    return False


async def download_tokenizer_files(model_id: ModelId) -> Path:
    """Download only the tokenizer-related files for a model."""
    target_dir = await ensure_models_dir() / model_id.normalize()
    target_dir.mkdir(parents=True, exist_ok=True)

    file_list = await fetch_file_list_with_cache(model_id, "main", recursive=True)

    tokenizer_files = [f for f in file_list if is_tokenizer_file(f.path)]

    if not tokenizer_files:
        pytest.skip(f"No tokenizer files found for {model_id}")

    for file_entry in tokenizer_files:
        with contextlib.suppress(FileNotFoundError):
            await download_file_with_retry(
                model_id, "main", file_entry.path, target_dir
            )

    return target_dir


# Get a sample of models to test (one per family to keep tests fast)
def get_test_models() -> list[ModelCard]:
    """Get a representative sample of models to test."""
    # Pick one model from each family to test
    families: dict[str, ModelCard] = {}
    for card in MODEL_CARDS.values():
        # Extract family name (e.g., "llama-3.1" from "llama-3.1-8b")
        parts = card.model_id.short().split("-")
        family = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]

        if family not in families:
            families[family] = card

    return list(families.values())


TEST_MODELS: list[ModelCard] = get_test_models()

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_encode_decode(model_card: ModelCard) -> None:
    """Test that tokenizer can encode and decode text correctly."""
    model_id = model_card.model_id

    # Download tokenizer files
    model_path = await download_tokenizer_files(model_id)

    # Verify required files exist
    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    if not has_tokenizer:
        pytest.skip(f"Required tokenizer files not found for {model_id}")

    # Load tokenizer
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)

    # Test basic encoding
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    assert isinstance(encoded, list), f"encode() should return a list for {model_id}"
    assert len(encoded) > 0, f"encode() should return non-empty list for {model_id}"
    assert all(isinstance(t, int) for t in encoded), (
        f"All tokens should be integers for {model_id}"
    )

    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str), f"decode() should return a string for {model_id}"
    assert test_text in decoded or decoded.strip() == test_text.strip(), (
        f"decode(encode(x)) should preserve text for {model_id}: got {decoded!r}"
    )

    # Test with longer text
    long_text = "The quick brown fox jumps over the lazy dog. " * 10
    long_encoded = tokenizer.encode(long_text)
    assert len(long_encoded) > len(encoded), (
        f"Longer text should produce more tokens for {model_id}"
    )

    # Test empty string
    empty_encoded = tokenizer.encode("")
    assert isinstance(empty_encoded, list), (
        f"encode('') should return a list for {model_id}"
    )

    # Test special characters
    special_text = 'Hello!\n\tWorld? <test> & "quotes"'
    special_encoded = tokenizer.encode(special_text)
    assert len(special_encoded) > 0, f"Special chars should encode for {model_id}"

    # Test unicode
    unicode_text = "Hello 世界 🌍"
    unicode_encoded = tokenizer.encode(unicode_text)
    assert len(unicode_encoded) > 0, f"Unicode should encode for {model_id}"


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_has_required_attributes(model_card: ModelCard) -> None:
    """Test that tokenizer has required attributes for inference."""
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    if not has_tokenizer:
        pytest.skip(f"Required tokenizer files not found for {model_id}")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Check for vocabulary size
    empty_vocab: dict[str, int] = {}
    vocab_size: int = getattr(tokenizer, "vocab_size", None) or len(
        getattr(tokenizer, "get_vocab", lambda: empty_vocab)()
    )
    assert vocab_size > 0, f"Tokenizer should have vocab_size > 0 for {model_id}"

    # Check for EOS token (either from tokenizer or explicitly provided)
    has_eos = (
        eos_token_ids is not None
        or getattr(tokenizer, "eos_token_id", None) is not None
        or getattr(tokenizer, "eos_token", None) is not None
    )
    assert has_eos, f"Tokenizer should have EOS token for {model_id}"


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_special_tokens(model_card: ModelCard) -> None:
    """Test that tokenizer can encode text containing special tokens.

    This is critical because the actual inference path uses prompts with
    special tokens from chat templates. If special tokens aren't handled
    correctly, encoding will fail.
    """
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    assert has_tokenizer, f"Required tokenizer files not found for {model_id}"

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)

    # Get special tokens from the tokenizer
    special_tokens: list[str] = []

    # Try to get special tokens from various sources
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens.extend(tokenizer.all_special_tokens)
    elif hasattr(tokenizer, "_tokenizer") and hasattr(
        tokenizer._tokenizer,
        "all_special_tokens",
    ):
        special_tokens.extend(tokenizer._tokenizer.all_special_tokens)

    # Also check for common special token attributes
    for attr in [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
    ]:
        token = getattr(tokenizer, attr, None)
        if token is None and hasattr(tokenizer, "_tokenizer"):
            token = getattr(tokenizer._tokenizer, attr, None)
        if token and isinstance(token, str) and token not in special_tokens:
            special_tokens.append(token)

    # If we found special tokens, test encoding text that contains them
    if special_tokens:
        # Create text with special tokens interspersed
        test_with_special = f"{special_tokens[0]}Hello world"
        if len(special_tokens) > 1:
            test_with_special += f"{special_tokens[1]}"

        encoded = tokenizer.encode(test_with_special)
        assert isinstance(encoded, list), (
            f"encode() with special tokens should return list for {model_id}"
        )
        assert len(encoded) > 0, (
            f"encode() with special tokens should return non-empty list for {model_id}"
        )
        assert all(isinstance(t, int) for t in encoded), (
            f"All tokens should be integers for {model_id}"
        )

        # Verify we can decode
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str), f"decode() should return string for {model_id}"

    # Test with angle-bracket tokens (common format for special tokens)
    # These should not raise errors even if they're not actual special tokens
    angle_bracket_text = "<|test|>Hello<|end|>"
    encoded = tokenizer.encode(angle_bracket_text)
    assert isinstance(encoded, list), (
        f"encode() with angle brackets should return list for {model_id}"
    )
    assert len(encoded) > 0, (
        f"encode() with angle brackets should be non-empty for {model_id}"
    )


# Specifically test Kimi tokenizer since it has special handling
@pytest.mark.asyncio
async def test_kimi_tokenizer_specifically():
    """Test Kimi tokenizer with its specific patches and quirks."""
    kimi_models = [
        card for card in MODEL_CARDS.values() if "kimi" in card.model_id.lower()
    ]

    if not kimi_models:
        pytest.skip("No Kimi models found in MODEL_CARDS")

    model_card = kimi_models[0]
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    # Ensure the custom tokenizer file exists
    if not (model_path / "tokenization_kimi.py").exists():
        pytest.skip("tokenization_kimi.py not found")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Test encode/decode cycle
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) > 0, "Kimi tokenizer should encode text"
    assert isinstance(decoded, str), "Kimi tokenizer should decode to string"

    # Test that the patched encode works (returns list of ints)
    assert all(isinstance(t, int) for t in encoded), "Tokens should be integers"

    # Test encoding text with special tokens (like from chat templates)
    # This is critical - the warmup inference uses prompts with special tokens
    special_token_text = "<|im_user|>user<|im_middle|>Hello<|im_end|><|im_assistant|>"
    special_encoded = tokenizer.encode(special_token_text)
    assert len(special_encoded) > 0, "Kimi tokenizer should handle special tokens"
    assert all(isinstance(t, int) for t in special_encoded), (
        "Special token encoding should return integers"
    )

    # Verify EOS token is set
    assert eos_token_ids == [163586], "Kimi EOS token should be [163586]"


# Test GLM tokenizer since it also has special handling
@pytest.mark.asyncio
async def test_glm_tokenizer_specifically():
    """Test GLM tokenizer with its specific EOS tokens."""
    glm_model_cards = [
        card for card in MODEL_CARDS.values() if "glm" in card.model_id.lower()
    ]

    if not glm_model_cards:
        pytest.skip("No GLM models found in MODEL_CARDS")

    model_card = glm_model_cards[0]
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (model_path / "tokenizer.json").exists() or (
        model_path / "tokenizer_config.json"
    ).exists()
    if not has_tokenizer:
        pytest.skip("GLM tokenizer files not found")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Test encode/decode
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) > 0, "GLM tokenizer should encode text"
    assert isinstance(decoded, str), "GLM tokenizer should decode to string"

    # Verify EOS tokens
    assert eos_token_ids == [
        151336,
        151329,
        151338,
    ], "GLM EOS tokens should be correct"
