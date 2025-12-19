from exo.shared.types.worker.instances import BoundInstance
from exo.worker.engines.mflux.distributed_model import DistributedImageModel


def initialize_mflux(bound_instance: BoundInstance) -> DistributedImageModel:
    """Initialize DistributedImageModel from a BoundInstance."""
    return DistributedImageModel.from_bound_instance(bound_instance)
