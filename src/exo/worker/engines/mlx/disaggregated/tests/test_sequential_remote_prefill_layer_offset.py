"""The sequential generation path (EXO_NO_BATCH) must pass the rank's start_layer
to remote_prefill as layer_offset, just like the batch path.

Every rank of a multi-node decode instance fetches the remote prefill on its own and
the prefill server numbers layers globally. A pipeline rank holds only layers
[start_layer, end_layer), so without the offset it would ingest global layers 0..k into
its local layers (wrong attention, no error). This test drives mlx_generate (the
sequential path) for a rank holding layers 3..5 and asserts the mocked remote_prefill
receives layer_offset=3.
"""

from typing import cast
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.types import Model

# The sequential path only fetches remote prefills for prompts longer than this.
_REMOTE_PREFILL_MIN_TOKENS = 1000


def _run_mlx_generate(monkeypatch: pytest.MonkeyPatch, start_layer: int) -> int:
    generate_module = "exo.worker.engines.mlx.generator.generate"

    def _identity(tokens: mx.array, tokenizer: TokenizerWrapper) -> mx.array:
        return tokens

    mock_remote: Mock = Mock(return_value=(1.0, 100, []))
    monkeypatch.setattr(f"{generate_module}.remote_prefill", mock_remote)

    long_tokens: mx.array = mx.arange(_REMOTE_PREFILL_MIN_TOKENS + 500)
    monkeypatch.setattr(
        f"{generate_module}.encode_prompt",
        Mock(return_value=long_tokens),
    )
    monkeypatch.setattr(
        f"{generate_module}.fix_unmatched_think_end_tokens",
        Mock(side_effect=_identity),
    )
    monkeypatch.setattr(
        f"{generate_module}.system_prompt_token_count",
        Mock(return_value=0),
    )
    monkeypatch.setattr(
        f"{generate_module}.make_kv_cache",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        f"{generate_module}.stream_generate",
        Mock(return_value=iter(())),
    )

    task = TextGenerationTaskParams(
        model=ModelId("test/model"),
        input=[],
        max_output_tokens=1,
        prefill_endpoint="http://localhost:9999",
    )
    generator = mlx_generate(
        model=cast(Model, object()),
        tokenizer=cast(TokenizerWrapper, object()),
        task=task,
        prompt="x",
        kv_prefix_cache=None,
        group=None,
        start_layer=start_layer,
    )
    # The generator hits remote_prefill during the first step, then finishes.
    for _ in generator:  # noqa: B007 - just consume
        pass

    return cast(int, mock_remote.call_args.kwargs["layer_offset"])


def test_sequential_path_passes_layer_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_mlx_generate(monkeypatch, start_layer=3) == 3
