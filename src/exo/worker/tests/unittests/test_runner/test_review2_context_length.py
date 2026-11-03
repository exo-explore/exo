"""review2: the context check trusts the model card's context_length, which can be below what
the model itself takes, and mishandles Ollama's num_predict=-1.

Adapted: the runner's limit is model_context_length (the most the card or config.json says),
so the first test goes through it rather than passing the card's value straight in."""

from exo.worker.runner.context_length import context_length_error, model_context_length

# mlx-community/Qwen3-30B-A3B-4bit: the card says 32768; the model's config.json says
# max_position_embeddings = 40960 (so does Qwen3-0.6B, the other 32768 card).
QWEN3_CARD_CONTEXT = 32768
QWEN3_MODEL_POSITIONS = 40960


def test_a_request_that_fits_the_model_is_not_refused_because_of_the_card() -> None:
    prompt, max_tokens = 30_000, 8_192
    assert prompt + max_tokens <= QWEN3_MODEL_POSITIONS  # ran before the fix
    limit = model_context_length(
        QWEN3_CARD_CONTEXT, {"max_position_embeddings": QWEN3_MODEL_POSITIONS}
    )
    assert context_length_error(prompt, max_tokens, limit) is None


def test_ollama_unlimited_num_predict_is_not_reported_as_minus_one_tokens() -> None:
    # Ollama's num_predict=-1 means "no limit"; the adapter passes it through as
    # max_output_tokens, so the message tells the client it asked for -1 tokens.
    message = context_length_error(40_000, -1, QWEN3_CARD_CONTEXT)
    assert message is not None
    assert "-1" not in message, message
