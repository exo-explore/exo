"""reasoning_effort and thinking off for the MiniMax M3 and M3.1 chat templates.

- M3.1 renders ``reasoning_effort`` as ``<effort>…</effort>`` in its system prompt and
  documents ``low``, ``medium``, ``high``, ``xhigh`` and ``max``. exo's ``none`` and
  ``minimal`` aren't among them. M3's template has no effort and ignores the variable.
- Both think adaptively by default (the model decides), and a turn without reasoning
  starts with a bare ``</mm:think>``.
  - M3's template has a ``thinking_mode`` switch: ``disabled`` says so in the system
    prompt and ends the generation prompt with ``</mm:think>``. exo sets it when a
    request turns thinking off. (``enabled`` would open the reasoning block in the prompt
    instead; exo leaves thinking on at the adaptive default.)
  - M3.1's template has no switch: exo ends the generation prompt with ``</mm:think>``,
    the prefix its system prompt names for thinking disabled.
"""

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import ReasoningEffortHook, TemplateKwargsHook

THINKING_OFF_SUFFIX = "</mm:think>"


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "minimax-m3" in task_params.model.lower()


def _has_thinking_mode(task_params: TextGenerationTaskParams) -> bool:
    """M3's template has a thinking_mode switch; M3.1's doesn't."""
    return "minimax-m3.1" not in task_params.model.lower()


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    """Unset or none -> none given (the template default); minimal -> low; low, medium,
    high and xhigh as they are. A request that only turns thinking on gets medium."""
    match task_params.reasoning_effort:
        case None:
            return "medium" if task_params.enable_thinking is True else None
        case "none":
            return None
        case "minimal" | "low":
            return "low"
        case "medium" | "high" | "xhigh":
            return task_params.reasoning_effort


def thinking_off_suffix(task_params: TextGenerationTaskParams) -> str | None:
    return None if _has_thinking_mode(task_params) else THINKING_OFF_SUFFIX


def template_kwargs(task_params: TextGenerationTaskParams) -> dict[str, str]:
    if task_params.enable_thinking is False and _has_thinking_mode(task_params):
        return {"thinking_mode": "disabled"}
    return {}


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
KWARGS_HOOK = TemplateKwargsHook(applies_to=applies_to, kwargs=template_kwargs)
