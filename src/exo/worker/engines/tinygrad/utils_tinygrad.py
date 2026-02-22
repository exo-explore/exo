from collections.abc import Callable
from typing import Any

from exo.download.download_utils import build_model_path
from exo.shared.model_config import parse_model_config
from exo.shared.tokenizer.loader import load_tokenizer_for_model
from exo.shared.tokenizer.wrapper import TokenizerWrapper
from exo.shared.types.worker.instances import BoundInstance, TinygradInstance

from .weight_loader import TransformerWeights, load_transformer_weights


def initialize_tinygrad(bound_instance: BoundInstance) -> None:
    instance = bound_instance.instance
    assert isinstance(instance, TinygradInstance)
    # For single instance, we can let tinygrad instance decide device.
    # When sharding models, we may have to add further explicit initialization
    # routines to ensure effective sharding.


def load_tinygrad_items(
    bound_instance: BoundInstance, group: None,
    on_timeout: Callable[[], None] | None = None,
) -> tuple[TransformerWeights, Any]:
    shard = bound_instance.bound_shard
    model_id = shard.model_card.model_id
    model_path = build_model_path(model_id)
    config = parse_model_config(model_path / "config.json")
    weights = load_transformer_weights(
        model_path=model_path, config=config,
        start_layer=shard.start_layer, end_layer=shard.end_layer,
    )

    tokenizer = TokenizerWrapper(load_tokenizer_for_model(model_id, model_path))

    return weights, tokenizer
