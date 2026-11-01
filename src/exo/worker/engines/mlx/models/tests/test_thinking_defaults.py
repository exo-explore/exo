"""What each model's chat template gets for a Chat Completions request that only turns
thinking on (enable_thinking=true, no reasoning_effort). exo leaves the level unset, and
each family keeps the behaviour its own tests were written for: GLM-5.x runs at High and
Inkling at medium (as for "medium"), Hy3 gets high (its template does not think by
default), MiniMax M3 and M3.1 get medium, and Qwen3.8 gets its template's own default
(xhigh)."""

import asyncio

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionRequest
from exo.worker.engines.mlx.models import registry


@pytest.mark.parametrize(
    ("model", "template_effort"),
    [
        ("mlx-community/GLM-5.3-4bit", "high"),
        ("mlx-community/GLM-5.3-Flash-4bit", "high"),
        ("mlx-community/Inkling-Small-mlx-4bit", "medium"),
        ("XavierLocalAI/Hy3-4bit", "high"),
        ("MiniMaxAI/MiniMax-M3.1-Flash", "medium"),
        ("mlx-community/MiniMax-M3-4bit", "medium"),
        ("mlx-community/Qwen3.8-27B-4bit", None),
        ("mlx-community/Qwen3.8-Flash-Next-4bit", None),
    ],
)
def test_thinking_on_without_a_level(model: str, template_effort: str | None) -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": True,
        }
    )
    params = asyncio.run(chat_request_to_text_generation(request))
    assert (params.reasoning_effort, params.enable_thinking) == (None, True)
    hook = registry.reasoning_effort_hook_for(params)
    assert hook is not None
    assert hook.template_effort(params) == template_effort
