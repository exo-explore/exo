from typing import Literal

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import Id
from exo.shared.types.worker.instances import InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import FrozenModel


class DeploymentId(Id):
    pass


class Deployment(FrozenModel):
    """A request to keep a model running.

    While a deployment exists, the master's keeper places an instance of its model whenever the
    cluster has none, with the same options a `/place_instance` request takes. Instances know
    nothing about deployments: any instance of the model counts, however it was created.
    """

    deployment_id: DeploymentId
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    # The instance the keeper placed last; deleting the deployment deletes it
    instance_id: InstanceId | None = None
    # Why the keeper's last placement failed, until a placement succeeds
    placement_error: str | None = None


# serving: an instance of the model has every runner ready; starting: one is on its way up;
# placing: the keeper will place one shortly; cant_place: no placement fits the cluster right now
DeploymentStatus = Literal["serving", "starting", "placing", "cant_place"]
