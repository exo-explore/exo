from exo.shared.types.worker.instances import BoundInstance
from exo.worker.engines.mflux.distributed_flux import DistributedFlux1


def initialize_mflux(bound_instance: BoundInstance) -> DistributedFlux1:
    """Initialize DistributedFlux1 from a BoundInstance."""
    return DistributedFlux1.from_bound_instance(bound_instance)
