from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.download.download_utils import build_model_path


def initialize_mflux(bound_instance: BoundInstance) -> Flux1:
    model_id = bound_instance.bound_shard.model_meta.model_id
    model_path = build_model_path(model_id)
    model = Flux1(
        model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
        local_path=str(model_path),
        # quantize=8,
    )

    return model
