from exo.api.dashboard import (
    DashboardRunnerUpdate,
    dashboard_event_from_lv_update,
    dashboard_topology_update,
    lv_update_affects_topology,
)
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import SocketConnection, SocketConnections
from exo.shared.types.worker.runners import RunnerId, RunnerReady


class MockAggregator:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def dump(self) -> dict[str, str]:
        return self._values


def test_dashboard_lv_runner_status_metric_returns_runner_update() -> None:
    runner_id = RunnerId("runner-one")
    runner_status = RunnerReady(prefill_server_port=12345)

    event = dashboard_event_from_lv_update(
        f"node_metrics/node-one/runners/{runner_id}/status",
        runner_status.model_dump_json(),
    )

    assert event is not None
    event_name, payload = event
    assert event_name == "runner_update"
    assert isinstance(payload, DashboardRunnerUpdate)
    assert payload.runner_id == runner_id
    assert payload.runner == runner_status


def test_dashboard_lv_runner_status_delete_removes_runner() -> None:
    event = dashboard_event_from_lv_update(
        "node_metrics/node-one/runners/runner-one/status",
        None,
    )

    assert event is not None
    event_name, payload = event
    assert event_name == "runner_update"
    assert isinstance(payload, DashboardRunnerUpdate)
    assert payload.runner_id == RunnerId("runner-one")
    assert payload.runner is None


def test_dashboard_lv_socket_connections_metric_affects_topology() -> None:
    source_node_id = NodeId("source-node")
    sink_node_id = NodeId("sink-node")
    socket_connection = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/52415")
    )
    socket_connections = SocketConnections(
        connections={sink_node_id: [socket_connection]}
    )

    key = f"node_metrics/{source_node_id}/socket_connections"

    assert dashboard_event_from_lv_update(key, socket_connections.model_dump_json()) is None
    assert lv_update_affects_topology(key)

    update = dashboard_topology_update(
        node_metric_values={key: socket_connections.model_dump_json()}
    )

    assert update.topology.nodes == [source_node_id]
    assert update.topology.connections == {}


def test_state_with_aggregator_reads_runner_status_key() -> None:
    runner_id = RunnerId("runner-one")
    runner_status = RunnerReady(prefill_server_port=12345)

    state = State().with_aggregator(
        MockAggregator(
            {
                f"node-one/runners/{runner_id}/status": runner_status.model_dump_json(),
            }
        )  # pyright: ignore[reportArgumentType]
    )

    assert state.runners[runner_id] == runner_status
