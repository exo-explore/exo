# pyright: reportPrivateUsage=false
"""The instance endpoints launch and stop deployments, with the same requests and answers as
before: POST /instance, POST /place_instance and DELETE /instance/{id}."""

import anyio
import pytest
from fastapi import HTTPException

import exo.api.main as api_main
from exo.api.tests.applying_master import sent_while_applying
from exo.api.tests.test_place_instance_validation import (
    MODEL,
    NODE,
    known_model,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    single_node_api,
)
from exo.api.types import (
    CreateInstanceParams,
    DeploymentConstraints,
    PlaceInstanceParams,
)
from exo.master.tests.test_requests_to_deleted_instances import deployment_of
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.model_readiness import deployment_readiness
from exo.shared.types.commands import (
    CreateDeployment,
    DeleteDeployment,
    ResumeDeployment,
)
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.deployments import Hold, Waiting
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import Sharding
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def instance_on_one_node() -> Instance:
    runner = RunnerId()
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL.model_id,
        node_to_runner={NodeId("a-node"): runner},
        runner_to_shard={runner: get_pipeline_shard_metadata(MODEL.model_id, 0)},
    )


async def test_a_previewed_instance_is_launched_with_the_users_constraints() -> None:
    api, commands = single_node_api()
    preview = await api.get_placement(MODEL.model_id)

    response, (sent,) = await sent_while_applying(
        api,
        commands,
        lambda: api.create_instance(
            CreateInstanceParams(
                instance=preview,
                constraints=DeploymentConstraints(min_nodes=1, pinned_nodes=[NODE]),
            )
        ),
    )

    assert isinstance(sent, CreateDeployment)
    assert sent.deployment_id == response.deployment_id
    # A preview's own deployment id is replaced: every launch is a new deployment
    assert sent.instance.deployment_id == response.deployment_id
    assert sent.instance.instance_id == preview.instance_id
    assert (
        sent.spec.model_id,
        sent.spec.sharding,
        sent.spec.instance_meta,
        sent.spec.min_nodes,
        sent.spec.pinned_nodes,
    ) == (MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1, [NODE])
    assert response.model_card == MODEL


async def test_an_instance_launched_without_constraints_may_come_back_smaller() -> None:
    api, commands = single_node_api()

    placement = await api.get_placement(MODEL.model_id)
    _, (sent,) = await sent_while_applying(
        api,
        commands,
        lambda: api.create_instance(CreateInstanceParams(instance=placement)),
    )

    assert isinstance(sent, CreateDeployment)
    assert (sent.spec.min_nodes, sent.spec.pinned_nodes) == (1, [])


async def test_place_instance_launches_a_deployment_with_its_node_count() -> None:
    api, commands = single_node_api()

    response, (sent,) = await sent_while_applying(
        api,
        commands,
        lambda: api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id)),
    )

    assert isinstance(sent, CreateDeployment)
    # The placement made here, launched as it is, for this deployment
    assert sent.instance.deployment_id == sent.deployment_id
    assert sent.spec.min_nodes == 1
    assert (response.deployment_id, response.model_card) == (sent.deployment_id, MODEL)


async def test_stopping_by_instance_id_stops_its_deployment() -> None:
    api, commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    api.state = api.state.model_copy(
        update={
            "instances": {instance.instance_id: instance},
            "deployments": {instance.deployment_id: deployment_of(instance)},
        }
    )

    response, (sent,) = await sent_while_applying(
        api, commands, lambda: api.delete_instance(instance.instance_id)
    )

    assert sent == DeleteDeployment(
        command_id=sent.command_id, deployment_id=instance.deployment_id
    )
    assert response.instance_id == instance.instance_id


async def test_a_model_waiting_to_be_placed_again_is_stopped_by_its_deployment_id() -> (
    None
):
    api, commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    # Its instance is gone: the deployment waits
    api.state = api.state.model_copy(
        update={"deployments": {instance.deployment_id: deployment_of(instance)}}
    )

    _, (sent,) = await sent_while_applying(
        api, commands, lambda: api.delete_instance(InstanceId(instance.deployment_id))
    )

    assert isinstance(sent, DeleteDeployment)
    assert sent.deployment_id == instance.deployment_id


async def test_stopping_something_unknown_is_not_found() -> None:
    api, commands = single_node_api()

    with pytest.raises(HTTPException) as missing:
        await api.delete_instance(InstanceId())

    assert missing.value.status_code == 404
    assert commands.collect() == []


async def test_a_deployment_is_resumed_by_its_id() -> None:
    api, commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    api.state = api.state.model_copy(
        update={"deployments": {instance.deployment_id: deployment_of(instance)}}
    )

    response, (sent,) = await sent_while_applying(
        api, commands, lambda: api.resume_deployment(instance.deployment_id)
    )

    assert isinstance(sent, ResumeDeployment)
    assert sent.deployment_id == response.deployment_id == instance.deployment_id
    # It resumes the revision the user saw: a copy sent again changes nothing
    assert sent.revision == 0
    with pytest.raises(HTTPException) as missing:
        await api.resume_deployment(DeploymentId())
    assert missing.value.status_code == 404


async def test_a_request_for_a_model_being_placed_again_gets_503_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its instance went with a device, and exo is placing it again (or waits for the
    device). A 404 told clients the model doesn't exist, and OpenAI's SDKs don't retry a 404;
    they retry a 503 after its Retry-After."""
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.05)
    api, _commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    api.state = api.state.model_copy(
        update={"deployments": {instance.deployment_id: deployment_of(instance)}}
    )

    with pytest.raises(HTTPException) as refused:
        await api._validate_model_has_instance(MODEL.model_id)

    assert refused.value.status_code == 503
    assert refused.value.headers == {"Retry-After": "5"}
    assert "it is being placed again" in refused.value.detail


@pytest.mark.parametrize(
    ("waiting", "says"),
    [
        (
            Waiting(kind="DeviceAway"),
            "it waits for a device it must run on to come back",
        ),
        (Waiting(kind="NoDevicesWithFiles"), "no connected device has its files"),
        (
            Waiting(kind="NotEnoughDevices", needed=2, have=1),
            "it needs 2 devices that have its files, and 1 is connected",
        ),
        (
            Waiting(kind="DoesNotFit"),
            "it doesn't fit in the free memory of the devices that have its files",
        ),
    ],
    ids=["pinned_device_away", "no_files", "not_enough", "does_not_fit"],
)
def test_a_request_for_a_waiting_model_is_told_why_it_waits(
    waiting: Waiting, says: str
) -> None:
    instance = instance_on_one_node()
    deployment = deployment_of(instance).model_copy(update={"waiting": waiting})

    not_ready = deployment_readiness(deployment)

    assert not_ready.status == says


async def test_a_request_for_a_model_kept_for_a_device_it_cant_reach_ends_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its instance has a device this node can't reach. The master keeps that instance while
    the device may be back (a gone node an even split keeps listed, for the split hold), so
    its deployment isn't being placed again: the request ends at once, as for any model
    whose devices can't be reached. Another launch of the model, being placed again, is
    waited for instead."""
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.05)
    api, _commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    api.state = api.state.model_copy(
        update={
            "instances": {instance.instance_id: instance},
            "deployments": {instance.deployment_id: deployment_of(instance)},
        }
    )
    membership = FakeMembership(NodeId("api-node"), NODE)
    await membership.lose(NODE, "Gone")
    api.membership = membership

    with anyio.fail_after(1):
        with pytest.raises(api_main.ModelUnreachableError):
            await api._validate_model_has_instance(MODEL.model_id)

    other = deployment_of(instance).model_copy(update={"deployment_id": DeploymentId()})
    api.state = api.state.model_copy(
        update={"deployments": {**api.state.deployments, other.deployment_id: other}}
    )
    with pytest.raises(HTTPException) as refused:
        await api._validate_model_has_instance(MODEL.model_id)
    assert refused.value.status_code == 503
    assert "it is being placed again" in refused.value.detail


async def test_a_request_for_a_paused_model_gets_503_at_once_saying_why() -> None:
    api, _commands = single_node_api()
    instance = await api.get_placement(MODEL.model_id)
    paused = deployment_of(instance).model_copy(
        update={
            "hold": Hold(
                kind="CrashLoop", message="It lost a device 3 times in 10 minutes"
            )
        }
    )
    api.state = api.state.model_copy(
        update={"deployments": {instance.deployment_id: paused}}
    )

    with anyio.fail_after(1):
        with pytest.raises(HTTPException) as refused:
            await api._validate_model_has_instance(MODEL.model_id)

    assert refused.value.status_code == 503
    assert "paused (it lost a device 3 times in 10 minutes)" in refused.value.detail


async def test_a_request_for_a_model_nobody_launched_is_still_not_found() -> None:
    api, _commands = single_node_api()

    with pytest.raises(HTTPException) as missing:
        await api._validate_model_has_instance(MODEL.model_id)

    assert missing.value.status_code == 404
