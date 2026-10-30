"""The language model of NVIDIA's Nemotron-H Omni checkpoints (config.json model_type
`nemotron_h_omni`, e.g. Nemotron 3.5 Super-VL).

These checkpoints hold a nemotron_h language model, configured by `llm_config` and stored
under `language_model.`, next to a vision encoder (`vision_model.`) and its projector
(`mlp1.`). exo serves the text path: this is the family's nemotron_h model (model.py)
built from `llm_config`, and loading keeps only the language model's weights.
"""

from dataclasses import dataclass
from typing import Any

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.nemotron_h import ModelArgs as NemotronHModelArgs

from exo.worker.engines.mlx.models.nemotron_h.model import Model as NemotronHModel

LANGUAGE_MODEL_PREFIX = "language_model."


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    llm_config: dict[str, Any]


class Model(NemotronHModel):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(NemotronHModelArgs.from_dict(args.llm_config))

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        language_model = {
            name.removeprefix(LANGUAGE_MODEL_PREFIX): weights[name]
            for name in weights
            if name.startswith(LANGUAGE_MODEL_PREFIX)
        }
        return super().sanitize(language_model)
