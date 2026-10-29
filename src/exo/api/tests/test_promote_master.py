# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import ModelId
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import ShardAssignments

NODE_A = NodeId("node-a")
NODE_B = NodeId("node-b")


def _make_api(state: State) -> Any:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = state
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/master/promote/{node_id}")(api.promote_master)
    return api


def _idle_topology() -> Topology:
    topology = Topology()
    topology.add_node(NODE_A)
    return topology


def test_promote_master_rejects_unknown_node() -> None:
    api = _make_api(State(topology=_idle_topology(), instances={}))
    client = TestClient(api.app)

    response = client.post(f"/master/promote/{NODE_B}")

    assert response.status_code == 404
    api._send.assert_not_called()


def test_promote_master_rejects_while_instances_running() -> None:
    instance = MlxRingInstance(
        instance_id=InstanceId("instance-a"),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    api = _make_api(
        State(
            topology=_idle_topology(),
            instances={instance.instance_id: instance},
        )
    )
    client = TestClient(api.app)

    response = client.post(f"/master/promote/{NODE_A}")

    assert response.status_code == 409
    api._send.assert_not_called()


def test_promote_master_sends_command_when_idle() -> None:
    api = _make_api(State(topology=_idle_topology(), instances={}))
    client = TestClient(api.app)

    response = client.post(f"/master/promote/{NODE_A}")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["target_node_id"] == str(NODE_A)
    api._send.assert_called_once()
    command = api._send.call_args[0][0]
    assert command.target_node_id == NODE_A
