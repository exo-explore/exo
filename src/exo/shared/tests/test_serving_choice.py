"""Which instance an API node sends a request to, which of its nodes streams the output, where to
reach that node, and who may call the internal serving routes: pure functions of the node's own
view of the state (exo.shared.serving)."""

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.serving import (
    NoInstanceForModelError,
    candidate_instances,
    is_member_address,
    node_urls,
    output_node,
    prefill_endpoint_for,
)
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import CfgShardMetadata, ShardMetadata
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")
A, B, C = NodeId("a"), NodeId("b"), NodeId("c")


def instance_on(
    instance_id: str, nodes: list[NodeId], model: ModelId = MODEL
) -> tuple[Instance, list[RunnerId]]:
    runners = [RunnerId(f"{instance_id}-{node}") for node in nodes]
    return (
        get_mlx_ring_instance(
            instance_id=InstanceId(instance_id),
            model_id=model,
            node_to_runner=dict(zip(nodes, runners, strict=True)),
            runner_to_shard={
                runner: get_pipeline_shard_metadata(model, rank, len(nodes))
                for rank, runner in enumerate(runners)
            },
        ),
        runners,
    )


def state_with(
    *instances: tuple[Instance, list[RunnerId], RunnerStatus], **fields: object
) -> State:
    return State(
        instances={instance.instance_id: instance for instance, _, _ in instances},
        runners={
            runner: status for _, runners, status in instances for runner in runners
        },
        **fields,  # pyright: ignore[reportArgumentType]
    )


def everything(_instance: Instance) -> bool:
    return True


# --- choosing an instance ---------------------------------------------------------------


def test_a_ready_instance_is_chosen_over_one_still_downloading() -> None:
    """review2: the master used to send a request to the instance with fewer requests in flight
    even when only the other one was ready"""
    ready, ready_runners = instance_on("ready", [A])
    downloading, downloading_runners = instance_on("downloading", [B])
    state = state_with(
        (ready, ready_runners, RunnerReady()),
        (downloading, downloading_runners, RunnerWarmingUp()),
    )
    assert candidate_instances(state, MODEL, {ready.instance_id: 5}, everything) == [
        ready.instance_id
    ]


def test_a_model_with_no_instance_has_no_candidate() -> None:
    with pytest.raises(NoInstanceForModelError) as raised:
        candidate_instances(State(), MODEL, {}, everything)
    assert raised.value.message == f"No instance found for model {MODEL}"


def test_a_model_that_isnt_ready_says_what_it_is_doing() -> None:
    instance, runners = instance_on("loading", [A])
    state = state_with(
        (instance, runners, RunnerWarmingUp()),
        downloads={
            A: [
                DownloadOngoing(
                    node_id=A,
                    shard_metadata=instance.shard_assignments.runner_to_shard[
                        runners[0]
                    ],
                    download_progress=DownloadProgressData(
                        total=Memory.from_gb(10),
                        downloaded=Memory.from_gb(4),
                        downloaded_this_session=Memory.from_gb(4),
                        completed_files=0,
                        total_files=1,
                        speed=1.0,
                        eta_ms=30_000,
                        files={},
                    ),
                )
            ]
        },
    )
    with pytest.raises(NoInstanceForModelError) as raised:
        candidate_instances(state, MODEL, {}, everything)
    assert raised.value.message.startswith("The model is not ready yet: downloading")
    assert raised.value.retry_after_seconds == 30


def test_fewer_requests_in_flight_first_then_idle_runners_first() -> None:
    first, first_runners = instance_on("first", [A])
    second, second_runners = instance_on("second", [B])
    third, third_runners = instance_on("third", [C])
    state = state_with(
        (first, first_runners, RunnerRunning()),
        (second, second_runners, RunnerReady()),
        (third, third_runners, RunnerReady()),
    )
    in_flight = {second.instance_id: 2}
    # first and third have none from this node; third's runner isn't busy
    assert candidate_instances(state, MODEL, in_flight, everything) == [
        third.instance_id,
        first.instance_id,
        second.instance_id,
    ]


def test_equally_good_instances_are_chosen_at_random() -> None:
    """Several API nodes with nothing in flight don't all send their next request to the same
    idle instance (the first in the state's order, before)"""
    first, first_runners = instance_on("first", [A])
    second, second_runners = instance_on("second", [B])
    state = state_with(
        (first, first_runners, RunnerReady()), (second, second_runners, RunnerReady())
    )
    chosen = {candidate_instances(state, MODEL, {}, everything)[0] for _ in range(64)}
    assert chosen == {first.instance_id, second.instance_id}


def test_an_instance_this_node_cant_use_is_left_out() -> None:
    usable, usable_runners = instance_on("usable", [A])
    unreachable, unreachable_runners = instance_on("unreachable", [B])
    state = state_with(
        (usable, usable_runners, RunnerReady()),
        (unreachable, unreachable_runners, RunnerReady()),
    )
    assert candidate_instances(
        state, MODEL, {}, lambda instance: instance.instance_id == usable.instance_id
    ) == [usable.instance_id]


def test_a_prefill_only_instance_never_gets_a_request() -> None:
    decode, decode_runners = instance_on("decode", [A])
    prefill, prefill_runners = instance_on("prefill", [B])
    state = state_with(
        (decode, decode_runners, RunnerReady()),
        (prefill, prefill_runners, RunnerReady()),
        instance_links={
            InstanceLinkId("link"): InstanceLink(
                link_id=InstanceLinkId("link"),
                prefill_instances=[prefill.instance_id],
                decode_instances=[decode.instance_id],
            )
        },
    )
    assert candidate_instances(state, MODEL, {decode.instance_id: 9}, everything) == [
        decode.instance_id
    ]


# --- prefill/decode links ---------------------------------------------------------------


def edge(source: NodeId, sink: NodeId, ip: str) -> Connection:
    return Connection(
        source=source,
        sink=sink,
        edge=SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")),
    )


def topology(*connections: Connection) -> Topology:
    graph = Topology()
    for connection in connections:
        for node in (connection.source, connection.sink):
            if not graph.contains_node(node):
                graph.add_node(node)
        graph.add_connection(connection)
    return graph


def test_a_decode_request_fetches_its_prompt_from_the_least_busy_linked_prefill_server() -> (
    None
):
    decode, decode_runners = instance_on("decode", [A])
    busy, busy_runners = instance_on("busy", [B])
    idle, idle_runners = instance_on("idle", [C])
    state = state_with(
        (decode, decode_runners, RunnerReady()),
        (busy, busy_runners, RunnerReady()),
        (idle, idle_runners, RunnerReady()),
        instance_links={
            InstanceLinkId("link"): InstanceLink(
                link_id=InstanceLinkId("link"),
                prefill_instances=[busy.instance_id, idle.instance_id],
                decode_instances=[decode.instance_id],
            )
        },
        prefill_server_ports={busy_runners[0]: 6001, idle_runners[0]: 6002},
        topology=topology(edge(A, B, "10.0.0.2"), edge(A, C, "10.0.0.3")),
    )
    assert (
        prefill_endpoint_for(state, decode.instance_id, {busy.instance_id: 3})
        == "10.0.0.3:6002"
    )


def test_an_unlinked_instance_has_no_prefill_endpoint() -> None:
    decode, decode_runners = instance_on("decode", [A])
    state = state_with((decode, decode_runners, RunnerReady()))
    assert prefill_endpoint_for(state, decode.instance_id, {}) is None


# --- the output node --------------------------------------------------------------------


def test_text_streams_from_rank_0() -> None:
    instance, _ = instance_on("pipeline", [A, B, C])
    assert output_node(instance, images=False) == A


def test_an_image_model_streams_from_its_last_pipeline_stage() -> None:
    instance, _ = instance_on("pipeline", [A, B, C])
    assert output_node(instance, images=True) == C


def test_a_cfg_image_model_streams_from_the_positive_branch_s_last_stage() -> None:
    card = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(1000),
        n_layers=8,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        backends=[Backend.MlxMetal],
    )

    def shard(rank: int, cfg_rank: int, pipeline_rank: int) -> ShardMetadata:
        return CfgShardMetadata(
            model_card=card,
            device_rank=rank,
            world_size=4,
            start_layer=0,
            end_layer=8,
            n_layers=8,
            cfg_rank=cfg_rank,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=2,
        )

    nodes = [A, B, C, NodeId("d")]
    runners = [RunnerId(f"r{index}") for index in range(4)]
    instance = get_mlx_ring_instance(
        instance_id=InstanceId("cfg"),
        model_id=MODEL,
        node_to_runner=dict(zip(nodes, runners, strict=True)),
        runner_to_shard={
            runners[0]: shard(0, cfg_rank=0, pipeline_rank=0),
            runners[1]: shard(1, cfg_rank=0, pipeline_rank=1),
            runners[2]: shard(2, cfg_rank=1, pipeline_rank=0),
            runners[3]: shard(3, cfg_rank=1, pipeline_rank=1),
        },
    )
    assert output_node(instance, images=True) == B


# --- addresses and membership -------------------------------------------------------------


def network(*addresses: tuple[str, str]) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name=name, ip_address=address)
            for name, address in addresses
        ]
    )


def test_a_node_is_tried_at_its_verified_address_first_and_over_ipv4_only() -> None:
    state = State(
        topology=topology(edge(A, B, "192.168.1.20")),
        node_network={
            B: network(
                ("en1", "10.1.1.20"),
                ("en0", "192.168.1.20"),
                ("en0", "fe80::1%en0"),
                ("en0", "2001:db8::20"),
                ("lo0", "127.0.0.1"),
            )
        },
    )
    assert node_urls(state, A, B, 52415) == [
        "http://192.168.1.20:52415",
        "http://10.1.1.20:52415",
    ]


def test_only_loopback_and_the_cluster_s_nodes_may_call_the_internal_routes() -> None:
    state = State(
        topology=topology(edge(A, B, "192.168.1.20")),
        node_network={
            A: network(("en0", "192.168.1.10"), ("en0", "fe80::10%en0")),
            B: network(("en0", "192.168.1.20")),
            # Known, but no longer in the cluster
            C: network(("en0", "192.168.1.30")),
        },
    )
    assert is_member_address(state, "192.168.1.10")
    assert is_member_address(state, "::ffff:192.168.1.20")
    assert is_member_address(state, "fe80::10")
    assert is_member_address(state, "127.0.0.1")
    assert is_member_address(state, "::1")
    assert not is_member_address(state, "192.168.1.30")
    assert not is_member_address(state, "203.0.113.9")
    assert not is_member_address(state, None)
