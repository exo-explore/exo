"""reasoning_effort for the Muse Spark chat template.

The template takes ``low``, ``medium``, ``high`` and ``xhigh`` (rendered as "Reasoning
strength: 8/32/256/512." in the system message), defaults to ``xhigh`` and raises on any
other value. Neither the template nor the model card has a way to turn reasoning off (no
switch, and no documented prompt without a reasoning message), so the model can't reason
less than ``low``: exo's ``none`` (also what a request that turns thinking off resolves to)
and ``minimal`` give ``low``, and such a request still gets a short reasoning message.
Unset leaves the template's default, ``xhigh``.
"""

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import ReasoningEffortHook


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "muse-spark" in task_params.model.lower()


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    match task_params.reasoning_effort:
        case None:
            return None
        case "none" | "minimal" | "low":
            return "low"
        case "medium" | "high" | "xhigh":
            return task_params.reasoning_effort


def thinking_off_suffix(_task_params: TextGenerationTaskParams) -> str | None:
    return None


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
