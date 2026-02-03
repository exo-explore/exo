import socket
from typing import Literal

import anyio
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from hypercorn import Config
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from loguru import logger
from pydantic import BaseModel

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import CommandId
from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import ChunkGenerated, Event, RunnerStatusUpdated
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TextGeneration,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.utils.channels import channel, mp_channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.worker.runner.bootstrap import entrypoint


class Tests(BaseModel):
    # list[hostname, ip addr]
    devs: list[list[str]]
    ibv_devs: list[list[str | None]] | None
    model_id: ModelId
    kind: Literal["ring", "jaccl", "both"]


iid = InstanceId("im testing here")


async def main():
    logger.info("starting cool server majig")
    cfg = Config()
    cfg.bind = "0.0.0.0:52414"
    # nb: shared.logging needs updating if any of this changes
    cfg.accesslog = "-"
    cfg.errorlog = "-"
    ev = anyio.Event()
    app = FastAPI()
    app.post("/run_test")(run_test)
    app.post("/kill")(lambda: kill(ev))
    app.get("/tb_detection")(tb_detection)
    app.get("/models")(list_models)
    await serve(
        app,  # type: ignore
        cfg,
        shutdown_trigger=lambda: ev.wait(),
    )


def kill(ev: anyio.Event):
    ev.set()
    return Response(status_code=204)


async def tb_detection():
    send, recv = channel[GatheredInfo]()
    ig = InfoGatherer(send)
    with anyio.move_on_after(1):
        await ig._monitor_system_profiler_thunderbolt_data()  # pyright: ignore[reportPrivateUsage]
    with recv:
        return recv.collect()


def list_models():
    sent = set[str]()
    for path in EXO_MODELS_DIR.rglob("model-*.safetensors"):
        if "--" not in path.parent.name:
            continue
        name = path.parent.name.replace("--", "/")
        if name in sent:
            continue
        sent.add(name)
        yield ModelId(path.parent.name.replace("--", "/"))


async def run_test(test: Tests):
    weird_hn = socket.gethostname()
    for dev in test.devs:
        if weird_hn.startswith(dev[0]) or dev[0].startswith(weird_hn):
            hn = dev[0]
            break
    else:
        raise ValueError(f"{weird_hn} not in {test.devs}")

    async def run():
        logger.info(f"testing {test.model_id}")

        instances: list[Instance] = []
        if test.kind in ["ring", "both"]:
            i = await ring_instance(test, hn)
            if i is None:
                yield "no model found"
                return
            instances.append(i)
        if test.kind in ["jaccl", "both"]:
            i = await jaccl_instance(test)
            if i is None:
                yield "no model found"
                return
            instances.append(i)

        for instance in instances:
            recv = await execute_test(test, instance, hn)

            str_out = ""

            for item in recv:
                if isinstance(item, ChunkGenerated):
                    assert isinstance(item.chunk, TokenChunk)
                    str_out += item.chunk.text

                if isinstance(item, RunnerStatusUpdated) and isinstance(
                    item.runner_status, (RunnerFailed, RunnerShutdown)
                ):
                    yield str_out + "\n"
                    yield item.model_dump_json() + "\n"

    return StreamingResponse(run())


async def ring_instance(test: Tests, hn: str) -> Instance | None:
    hbn = [Host(ip="198.51.100.0", port=52417) for _ in test.devs]
    world_size = len(test.devs)
    for i in range(world_size):
        if test.devs[i][0] == hn:
            hn = test.devs[i][0]
        hbn[(i - 1) % world_size] = Host(ip=test.devs[i - 1][1], port=52417)
        hbn[(i + 1) % world_size] = Host(ip=test.devs[i + 1][1], port=52417)
        hbn[i] = Host(ip="0.0.0.0", port=52417)
        break
    else:
        raise ValueError(f"{hn} not in {test.devs}")

    card = await ModelCard.load(test.model_id)
    instance = MlxRingInstance(
        instance_id=iid,
        ephemeral_port=52417,
        hosts_by_node={NodeId(hn): hbn},
        shard_assignments=ShardAssignments(
            model_id=test.model_id,
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(test.devs[i][0]): PipelineShardMetadata(
                    model_card=card,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=(card.n_layers // world_size) * i,
                    end_layer=min(
                        card.n_layers, (card.n_layers // world_size) * (i + 1)
                    ),
                    n_layers=min(card.n_layers, (card.n_layers // world_size) * (i + 1))
                    - (card.n_layers // world_size) * i,
                )
                for i in range(world_size)
            },
        ),
    )

    return instance


async def execute_test(test: Tests, instance: Instance, hn: str) -> list[Event]:
    world_size = len(test.devs)
    commands: list[Task] = [
        (LoadModel(instance_id=iid)),
        (StartWarmup(instance_id=iid)),
        (
            TextGeneration(
                task_params=TextGenerationTaskParams(
                    model=test.model_id,
                    instructions="You are a helpful assistant",
                    input=[
                        InputMessage(
                            role="user", content="What is the capital of France?"
                        )
                    ],
                ),
                command_id=CommandId("yo"),
                instance_id=iid,
            )
        ),
        (Shutdown(runner_id=RunnerId(hn), instance_id=iid)),
    ]
    if world_size > 1:
        commands.insert(0, ConnectToGroup(instance_id=iid))
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RunnerId(hn), bound_node_id=NodeId(hn)
    )
    ev_send, _ev_recv = mp_channel[Event]()
    task_send, task_recv = mp_channel[Task]()

    for command in commands:
        task_send.send(command)

    entrypoint(
        bound_instance,
        ev_send,
        task_recv,
        logger,
    )

    # TODO(evan): return ev_recv.collect()
    return []


async def jaccl_instance(test: Tests) -> MlxJacclInstance | None:
    card = await ModelCard.load(test.model_id)
    world_size = len(test.devs)
    assert test.ibv_devs

    return MlxJacclInstance(
        instance_id=iid,
        jaccl_devices=test.ibv_devs,
        # rank 0 is always coordinator
        jaccl_coordinators={
            NodeId(host[0]): test.devs[0][1] + ":52417" for host in test.devs
        },
        shard_assignments=ShardAssignments(
            model_id=test.model_id,
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(host[0]): TensorShardMetadata(
                    model_card=card,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=0,
                    end_layer=card.n_layers,
                    n_layers=card.n_layers,
                )
                for i, host in enumerate(test.devs)
            },
        ),
    )


if __name__ == "__main__":
    anyio.run(main)
