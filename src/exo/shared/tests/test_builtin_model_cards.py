import tomllib
from pathlib import Path
from typing import cast

import pytest
from huggingface_hub import model_info

from exo.shared.constants import RESOURCES_DIR

_CARDS = sorted(
    card
    for directory in ("inference_model_cards", "image_model_cards")
    for card in (Path(RESOURCES_DIR) / directory).rglob("*.toml")
)


def _model_id(card: Path) -> str:
    model_id = cast(object, tomllib.loads(card.read_text())["model_id"])
    assert isinstance(model_id, str)
    return model_id


@pytest.mark.slow
@pytest.mark.parametrize("card", _CARDS, ids=[card.stem for card in _CARDS])
def test_builtin_card_names_a_repo_that_exists(card: Path) -> None:
    """A card whose repository was deleted or renamed offers a model nobody can download."""
    model_id = _model_id(card)
    assert model_info(model_id).id == model_id
