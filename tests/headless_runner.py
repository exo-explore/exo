import multiprocessing as mp
import socket
import time
import typing

import anyio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from hypercorn import Config
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from loguru import logger
from pydantic import BaseModel

from exo.shared.logging import InterceptLogger, logger_setup
from exo.shared.models.model_cards import MODEL_CARDS, ModelId
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.commands import CommandId
from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import Event
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.utils.channels import MpReceiver, MpSender, mp_channel
from exo.worker.download.impl_shard_downloader import (
    build_full_shard,
    exo_shard_downloader,
)
from exo.worker.runner.bootstrap import entrypoint


class Tests(BaseModel):
    # list[hostname, ip addr]
    devs: list[list[str]]
    model_id: str
    kind: typing.Literal["init", "warmup", "inference"]


hn = socket.gethostname()
mp.set_start_method("spawn", force=True)
logger_setup(None)


async def main():
    logger.info("starting cool server majig")
    logger.info(hn)
    await assert_downloads()
    cfg = Config()
    cfg.bind = "0.0.0.0:52415"
    # nb: shared.logging needs updating if any of this changes
    cfg.accesslog = "-"
    cfg.errorlog = "-"
    cfg.logger_class = InterceptLogger
    app = FastAPI()
    app.post("/ring")(ring_backend)
    app.post("/jaccl")(jaccl_backend)
    shutdown = anyio.Event()
    await serve(
        app,  # type: ignore
        cfg,
        shutdown_trigger=lambda: shutdown.wait(),
    )
    await anyio.sleep_forever()
    # gracefully shutdown the api
    shutdown.set()


async def assert_downloads():
    sd = exo_shard_downloader()
    # await sd.ensure_shard(await build_full_shard(MODEL_CARDS["qwen3-0.6b"].model_id))
    await sd.ensure_shard(await build_full_shard(MODEL_CARDS["llama-3.2-1b"].model_id))


async def ring_backend(test: Tests):
    iid = InstanceId(str(hash(str(test.devs))))
    return await execute_test(test, ring_instance(test, iid))


def ring_instance(test: Tests, iid: InstanceId) -> Instance:
    global hn
    hbn = [Host(ip="i dont care", port=52416) for _ in test.devs]
    world_size = len(test.devs)
    for i in range(world_size):
        if hn.startswith(test.devs[i][0]):
            hn = test.devs[i][0]
            if i - 1 >= 0:
                hbn[i - 1] = Host(ip=test.devs[i - 1][1], port=52416)
            if i + 1 < len(test.devs):
                hbn[i + 1] = Host(ip=test.devs[i + 1][1], port=52416)
            hbn[i] = Host(ip="0.0.0.0", port=52416)
            break

    meta = MODEL_CARDS[test.model_id].metadata
    instance = MlxRingInstance(
        instance_id=iid,
        ephemeral_port=52416,
        hosts_by_node={NodeId(hn): hbn},
        shard_assignments=ShardAssignments(
            model_id=ModelId(test.model_id),
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(test.devs[i][0]): PipelineShardMetadata(
                    model_meta=meta,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=(meta.n_layers // world_size) * i,
                    end_layer=min(
                        meta.n_layers, (meta.n_layers // world_size) * (i + 1)
                    ),
                    n_layers=min(meta.n_layers, (meta.n_layers // world_size) * (i + 1))
                    - (meta.n_layers // world_size) * i,
                )
                for i in range(world_size)
            },
        ),
    )

    return instance


async def execute_test(test: Tests, instance: Instance):
    world_size = len(test.devs)
    iid = InstanceId(str(hash(str(test.devs))))
    _handle, recv, send = new_runner(instance)
    if world_size > 1:
        send.send(ConnectToGroup(instance_id=iid))
    send.send(LoadModel(instance_id=iid))

    match test.kind:
        case "init":
            pass
        case "warmup":
            send.send(StartWarmup(instance_id=iid))
        case "inference":
            send.send(StartWarmup(instance_id=iid))
            send.send(
                ChatCompletion(
                    task_params=ChatCompletionTaskParams(
                        model=test.model_id,
                        messages=[
                            ChatCompletionMessage(
                                role="system", content="You are a helpful assistant"
                            ),
                            ChatCompletionMessage(
                                role="user", content="What is the capital of France?"
                            ),
                        ],
                    ),
                    command_id=CommandId("yo"),
                    instance_id=iid,
                )
            )

    send.send(Shutdown(runner_id=RunnerId(hn), instance_id=iid))

    async def map_recv():
        with recv:
            try:
                async for item in recv:
                    yield item.model_dump_json() + "\n"
            except anyio.ClosedResourceError:
                pass

    ret = StreamingResponse(map_recv())
    ret._pls_dont_gc = _handle  # type: ignore
    return ret


async def jaccl_backend(test: Tests):
    iid = InstanceId(str(hash(str(test.devs))))
    return await execute_test(test, jaccl_instance(test, iid))


def jaccl_instance(test: Tests, iid: InstanceId):
    global hn
    meta = MODEL_CARDS[test.model_id].metadata
    world_size = len(test.devs)
    for name, _ in test.devs:
        if hn.startswith(name):
            hn = name
            break

    return MlxJacclInstance(
        instance_id=iid,
        ibv_devices=[[None, "rdma_en3"], ["rdma_en3", None]],
        # rank 0 is always coordinator
        jaccl_coordinators={
            NodeId(host[0]): test.devs[0][1] + ":52416" for host in test.devs
        },
        shard_assignments=ShardAssignments(
            model_id=ModelId(test.model_id),
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(test.devs[i][0]): TensorShardMetadata(
                    model_meta=meta,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=meta.n_layers,
                    end_layer=meta.n_layers,
                    n_layers=meta.n_layers,
                )
                for i in range(world_size)
            },
        ),
    )


def new_runner(
    instance: Instance,
) -> tuple[mp.Process, MpReceiver[Event], MpSender[Task]]:
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RunnerId(hn), bound_node_id=NodeId(hn)
    )
    ev_send, ev_recv = mp_channel[Event]()
    task_send, task_recv = mp_channel[Task]()

    runner_process = mp.Process(
        target=entrypoint,
        args=(
            bound_instance,
            ev_send,
            task_recv,
            logger,
        ),
    )
    runner_process._pls_dont_gc = (ev_send, task_recv)  # type: ignore
    runner_process.start()
    time.sleep(0.1)
    return (runner_process, ev_recv, task_send)


if __name__ == "__main__":
    anyio.run(main)
