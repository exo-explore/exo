"""reasoning_effort for Inkling's chat template.

The template writes "Thinking effort level: X" into a system message, mapping none,
minimal, low, medium, high and max to 0, 0.1, 0.2, 0.7, 0.9 and 0.99 (0.9 when unset), and
fails on any other name. exo's levels are the same up to high, then xhigh, which becomes
max here. enable_thinking=false reaches the template as reasoning_effort="none", so no
suffix is needed.
"""

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import ReasoningEffortHook


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "inkling" in task_params.model.lower()


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    match task_params.reasoning_effort:
        case None:
            return None
        case "xhigh":
            return "max"
        case "none" | "minimal" | "low" | "medium" | "high":
            return task_params.reasoning_effort


def thinking_off_suffix(_task_params: TextGenerationTaskParams) -> str | None:
    return None


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
