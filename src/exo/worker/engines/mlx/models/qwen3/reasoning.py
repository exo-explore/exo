"""reasoning_effort for the Qwen3.8 chat template (Qwen3.8-27B and Qwen3.8-Flash-Next share
it byte for byte).

With thinking on, the template accepts only "xhigh" (its default, which also adds an
instruction to think carefully), "medium" and "low" (which adds an instruction to keep
thinking brief), and raises on any other value, so "minimal" and "high" made requests fail.
With thinking off it ignores the effort.
"""

from exo.shared.types.text_generation import ReasoningEffort, TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import ReasoningEffortHook

_TEMPLATE_EFFORT: dict[ReasoningEffort, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
}


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    model = task_params.model.lower()
    return "qwen3.8" in model or "qwen-3.8" in model


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    """exo's level mapped onto the template's; None (the template's default) when unset or
    when thinking is off."""
    effort = task_params.reasoning_effort
    if effort is None or task_params.enable_thinking is False:
        return None
    return _TEMPLATE_EFFORT.get(effort)


def thinking_off_suffix(_task_params: TextGenerationTaskParams) -> str | None:
    # The template turns thinking off itself (enable_thinking=false)
    return None


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
