import asyncio
import sys

import anyio
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import LoadModel, StartWarmup, TaskId, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxRingInstance,
    ShardAssignments,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import channel
from exo.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "darwin", reason="MLX only available on macOS/Apple Silicon"
)
async def test_runner_supervisor_with_mlx_engine():
    """Test MlxRingInstance with MLX engine - requires Apple Silicon."""
    # 1. Create a BoundInstance for an MLX model
    model_card = ModelCard(
        model_id=ModelId("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        n_layers=22,
        storage_size=Memory.from_gb(2.2),
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        start_layer=0,
        end_layer=11,
        device_rank=0,
        world_size=2,
        n_layers=22,
    )
    instance = MlxRingInstance(
        instance_id="inst_mlx_test",
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            node_to_runner={"node1": RunnerId("runner1")},
            runner_to_shard={RunnerId("runner1"): shard_metadata},
        ),
        hosts_by_node={},
        ephemeral_port=12345,
    )
    bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=RunnerId("runner1"),
        bound_node_id=NodeId("node1"),
    )

    # 2. Create a RunnerSupervisor
    event_sender, event_receiver = channel()
    supervisor = RunnerSupervisor.create(
        bound_instance=bound_instance,
        event_sender=event_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)

        # Background listener to capture events
        runner_ready = False
        response_received = False

        async def event_listener():
            nonlocal runner_ready, response_received
            async with event_receiver:
                async for event in event_receiver:
                    if hasattr(event, "runner_status"):
                        print(f"Received status update: {event.runner_status}")
                        if event.runner_status.__class__.__name__ == "RunnerReady":
                            runner_ready = True
                    if isinstance(event, ChunkGenerated):
                        print(f"Received chunk: {event.chunk}")
                        response_received = True
                        supervisor.shutdown()
                        return

        tg.start_soon(event_listener)

        # 3. Send a LoadModel task
        load_task = LoadModel(task_id=TaskId(), instance_id=instance.instance_id)
        await supervisor.start_task(load_task)

        # 3.5 Send StartWarmup task
        warmup_task = StartWarmup(task_id=TaskId(), instance_id=instance.instance_id)
        await supervisor.start_task(warmup_task)

        # Wait for ready
        # We give it some time to process the warmup and transition to Ready
        for _ in range(100):  # Wait up to 10 seconds
            if runner_ready:
                break
            await asyncio.sleep(0.1)

        assert runner_ready, "Runner failed to reach Ready state after Warmup"

        # 4. Send a TextGeneration task
        text_gen_task = TextGeneration(
            task_id=TaskId(),
            command_id="cmd1",
            instance_id=instance.instance_id,
            task_params=TextGenerationTaskParams(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                input=[InputMessage(role="user", content="Hello, world!")],
                max_tokens=10,
            ),
        )

        await supervisor.start_task(text_gen_task)

        # Wait for response
        for _ in range(600):  # Wait up to 60 seconds
            if response_received:
                break
            await asyncio.sleep(0.1)

        # Ensure we shutdown if we timed out (or if listener didn't shut down yet)
        # Note: event_listener calls shutdown() on success, but calling it again is safe/idempotent usually,
        # or we can check. simpler to just cancel scope if we are done.
        if not response_received:
            supervisor.shutdown()

    # After the TaskGroup block exits, supervisor.run() has finished and joined the process.
    assert response_received, "Did not receive any text chunks"
