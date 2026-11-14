from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from exo.master.deployments import place_deployment, spec_of
from exo.master.placement import launch_refusal, memory_once_loaded, place_instance
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.tensorfold_support import prefill_link_refusal
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import DeploymentId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.placement import PlacementRequest
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.tensorfold import NativeCapabilities, ServeCapabilities
from exo.shared.types.worker.instances import InstanceMeta, TensorFoldInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import Sharding, whole_model_shard


@dataclass(frozen=True)
class Cluster:
    card: ModelCard
    node: NodeId
    state: State

    def place(self, request: PlacementRequest | None = None) -> TensorFoldInstance:
        placed = place_instance(
            request or self.request(),
            self.state.topology,
            self.state.instances,
            self.state.node_memory,
            self.state.node_network,
            self.state.node_backends,
            runners=self.state.runners,
            node_tensorfold=self.state.node_tensorfold,
        )
        (instance,) = placed.values()
        assert isinstance(instance, TensorFoldInstance)
        return instance

    def request(self, **changes: object) -> PlacementRequest:
        return PlacementRequest(
            model_card=self.card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.TensorFold,
            min_nodes=1,
        ).model_copy(update=changes)


@pytest.fixture
def cluster() -> Cluster:
    node = NodeId("native")
    card = ModelCard(
        model_id=ModelId("test/native"),
        storage_size=Memory.from_gb(2),
        n_layers=24,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        model_type="qwen3_5",
        tensorfold_format="mlx-q4g64",
        backends=[Backend.TensorFoldCuda, Backend.TensorFoldMetal],
    )
    capabilities = NativeCapabilities(
        schema=1,
        engine="zig",
        version="test",
        chip="nvidia-sm121",
        backends=["cuda"],
        families={"qwen3_5": ["mlx-q4g64"]},
        token_stream_versions=[1],
        serve=ServeCapabilities(flags={"--unix-socket": {}, "--name": {}}),
    )
    topology = Topology()
    topology.add_node(node)
    memory = MemoryUsage.from_bytes(
        ram_total=128 * 1024**3,
        ram_available=64 * 1024**3,
        swap_total=0,
        swap_available=0,
        gpu_available=Memory.from_gb(8),
    )
    return Cluster(
        card,
        node,
        State(
            topology=topology,
            node_memory={node: memory},
            node_backends={node: [Backend.TensorFoldCuda]},
            node_tensorfold={node: capabilities},
        ),
    )


def test_native_placement_keeps_backend_and_complete_model(cluster: Cluster) -> None:
    instance = cluster.place()
    (shard,) = instance.shard_assignments.runner_to_shard.values()
    assert shard == whole_model_shard(cluster.card)
    assert instance.context_length == 8192
    assert instance.parallel is None
    assert launch_refusal(instance, cluster.state) is None
    assert spec_of(instance).instance_meta == InstanceMeta.TensorFold


@pytest.mark.parametrize(
    "changes",
    [{"min_nodes": 2}, {"sharding": Sharding.Tensor}],
)
def test_native_placement_rejects_unimplemented_distribution(
    cluster: Cluster, changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="one complete model"):
        cluster.place(cluster.request(**changes))


@pytest.mark.parametrize("free", [None, Memory.from_gb(2)])
def test_cuda_placement_does_not_substitute_host_ram(
    cluster: Cluster, free: Memory | None
) -> None:
    state = cluster.state.model_copy(
        update={
            "node_memory": {
                cluster.node: cluster.state.node_memory[cluster.node].model_copy(
                    update={"gpu_available": free}
                )
            }
        }
    )
    with pytest.raises(ValueError, match="GPU memory"):
        Cluster(cluster.card, cluster.node, state).place()


@pytest.mark.parametrize(
    "changes",
    [
        {"token_stream_versions": []},
        {"chip": None},
        {"families": {"qwen3_5": ["bf16"]}},
        {"families": {"llama": ["mlx-q4g64"]}},
    ],
)
def test_native_placement_requires_exact_capabilities(
    cluster: Cluster, changes: dict[str, object]
) -> None:
    capabilities = cluster.state.node_tensorfold[cluster.node].model_copy(
        update=changes
    )
    state = cluster.state.model_copy(
        update={"node_tensorfold": {cluster.node: capabilities}}
    )
    with pytest.raises(ValueError, match="No TensorFold node"):
        Cluster(cluster.card, cluster.node, state).place()


def test_metal_placement_caps_available_memory_at_working_set(cluster: Cluster) -> None:
    capabilities = cluster.state.node_tensorfold[cluster.node].model_copy(
        update={"chip": "apple-m5", "backends": ["metal"]}
    )
    memory = cluster.state.node_memory[cluster.node].model_copy(
        update={"gpu_available": None, "gpu_working_set": Memory.from_gb(2)}
    )
    state = cluster.state.model_copy(
        update={
            "node_tensorfold": {cluster.node: capabilities},
            "node_backends": {cluster.node: [Backend.TensorFoldMetal]},
            "node_memory": {cluster.node: memory},
        }
    )
    with pytest.raises(ValueError, match="GPU memory"):
        Cluster(cluster.card, cluster.node, state).place()
    state = state.model_copy(
        update={
            "node_memory": {
                cluster.node: memory.model_copy(
                    update={"gpu_working_set": Memory.from_gb(8)}
                )
            }
        }
    )
    assert Cluster(cluster.card, cluster.node, state).place().context_length == 8192


def test_pending_native_load_reserves_gpu_memory_once(cluster: Cluster) -> None:
    instance = cluster.place()
    instances = {instance.instance_id: instance}
    memory = memory_once_loaded(cluster.state.node_memory, instances, {})
    assert memory[cluster.node].gpu_available == Memory.from_gb(5)
    (runner,) = instance.shard_assignments.runner_to_shard
    memory = memory_once_loaded(
        cluster.state.node_memory, instances, {runner: RunnerReady()}
    )
    assert memory[cluster.node].gpu_available == Memory.from_gb(8)


def test_launch_rechecks_gpu_memory_after_another_pending_load(
    cluster: Cluster,
) -> None:
    instance = cluster.place()
    memory = cluster.state.node_memory[cluster.node].model_copy(
        update={"gpu_available": Memory.from_gb(5)}
    )
    state = cluster.state.model_copy(
        update={
            "instances": {instance.instance_id: instance},
            "node_memory": {cluster.node: memory},
        }
    )
    assert (
        launch_refusal(instance, state)
        == "The selected TensorFold node has insufficient available GPU memory"
    )


def test_instance_wire_validation_refuses_partial_and_multiple_ranks(
    cluster: Cluster,
) -> None:
    instance = cluster.place()
    runner = RunnerId()
    partial = whole_model_shard(cluster.card).model_copy(update={"end_layer": 12})
    with pytest.raises(ValidationError, match="complete model"):
        TensorFoldInstance(
            instance_id=instance.instance_id,
            shard_assignments=ShardAssignments(
                model_id=cluster.card.model_id,
                runner_to_shard={runner: partial},
                node_to_runner={cluster.node: runner},
            ),
        )
    runner = next(iter(instance.shard_assignments.runner_to_shard))
    with pytest.raises(ValidationError, match="exactly one node"):
        TensorFoldInstance(
            instance_id=instance.instance_id,
            shard_assignments=instance.shard_assignments.model_copy(
                update={
                    "node_to_runner": {cluster.node: runner, NodeId("second"): runner}
                }
            ),
        )


def test_deployment_replacement_preserves_native_options(cluster: Cluster) -> None:
    instance = cluster.place(
        cluster.request(tensorfold_context_length=4096, tensorfold_parallel=2)
    )
    spec = spec_of(instance)
    placed = place_deployment(
        DeploymentId(),
        spec,
        cluster.card,
        cluster.state,
        {},
        allowed_nodes={cluster.node},
    )
    assert isinstance(placed.instance, TensorFoldInstance)
    assert placed.instance.context_length == 4096
    assert placed.instance.parallel == 2


def test_native_instance_cannot_join_prefill_links(cluster: Cluster) -> None:
    instance = cluster.place()
    assert prefill_link_refusal(
        [instance.instance_id], {instance.instance_id: instance}
    )
