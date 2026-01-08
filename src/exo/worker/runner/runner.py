import base64
import time

from exo.master.api import get_model_card
from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import ImageChunk, TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.models import ModelTask
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.image import (
    ImageGenerator,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.runner.bootstrap import logger


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    try:
        logger.info("hello from the runner")
        if getattr(shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        setup_start_time = time.time()

        model = None
        tokenizer = None
        sampler = None
        group = None

        model_card = get_model_card(shard_metadata.model_meta.model_id)
        assert model_card
        model_tasks = model_card.tasks

        current_status: RunnerStatus = RunnerIdle()
        logger.info("runner created")
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
        )
        with task_receiver as tasks:
            for task in tasks:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Running
                    )
                )
                event_sender.send(TaskAcknowledged(task_id=task.task_id))
                match task:
                    case ConnectToGroup() if isinstance(
                        current_status, (RunnerIdle, RunnerFailed)
                    ):
                        logger.info("runner connecting")
                        current_status = RunnerConnecting()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        group = initialize_mlx(bound_instance)

                        logger.info("runner connected")
                        current_status = RunnerConnected()

                    # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                    case LoadModel() if (
                        isinstance(current_status, RunnerConnected)
                        and group is not None
                    ) or (isinstance(current_status, RunnerIdle) and group is None):
                        current_status = RunnerLoading()
                        logger.info("runner loading")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        # TODO(ciaran): switch
                        if ModelTask.TextGeneration in model_tasks:
                            model, tokenizer, sampler = load_mlx_items(
                                bound_instance, group
                            )
                        elif (
                            ModelTask.TextToImage in model_tasks
                            or ModelTask.ImageToImage in model_tasks
                        ):
                            model = initialize_image_model(bound_instance)
                        else:
                            raise ValueError(
                                f"Unknown model task(s): {model_card.tasks}"
                            )

                        current_status = RunnerLoaded()
                        logger.info("runner loaded")
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        assert model

                        current_status = RunnerWarmingUp()
                        logger.info("runner warming up")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        logger.info(f"warming up inference for instance: {instance}")
                        if ModelTask.TextGeneration in model_tasks:
                            assert model and isinstance(model, Model)
                            assert tokenizer
                            assert sampler

                            toks = warmup_inference(
                                model=model,
                                tokenizer=tokenizer,
                                sampler=sampler,
                                # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                            )
                            logger.info(f"warmed up by generating {toks} tokens")
                            logger.info(
                                f"runner initialized in {time.time() - setup_start_time} seconds"
                            )
                        elif (
                            ModelTask.TextToImage in model_tasks
                            or ModelTask.ImageToImage in model_tasks
                        ):
                            assert isinstance(model, ImageGenerator)
                            image = warmup_image_generator(model=model)
                            if image is not None:
                                logger.info(
                                    f"warmed up by generating {image.size} image"
                                )
                            else:
                                logger.info("warmup completed (non-primary node)")

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case ChatCompletion(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert model and isinstance(model, Model)
                        assert tokenizer
                        assert sampler
                        logger.info(f"received chat request: {str(task)[:500]}")
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        assert task_params.messages[0].content is not None
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Generate responses using the actual MLX generation
                        for response in mlx_generate(
                            model=model,
                            tokenizer=tokenizer,
                            sampler=sampler,
                            task=task_params,
                        ):
                            match response:
                                case GenerationResponse():
                                    if shard_metadata.device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    idx=response.token,
                                                    model=shard_metadata.model_meta.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    finish_reason=response.finish_reason,
                                                ),
                                            )
                                        )
                                    # case TokenizedResponse():
                                    # TODO: something here ig

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case ImageGeneration(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert isinstance(model, ImageGenerator)
                        logger.info(
                            f"received image generation request: {str(task)[:500]}"
                        )
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        # Generate images using the image generation backend
                        for image_index, response in enumerate(
                            generate_image(
                                model=model,
                                task=task_params,
                            )
                        ):
                            match response:
                                case ImageGenerationResponse():
                                    if (
                                        shard_metadata.device_rank
                                        == shard_metadata.world_size - 1
                                    ):
                                        encoded_data = base64.b64encode(
                                            response.image_data
                                        ).decode("utf-8")
                                        # Split into chunks to stay under gossipsub 1MB limit
                                        data_chunks = [
                                            encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
                                            for i in range(
                                                0, len(encoded_data), EXO_MAX_CHUNK_SIZE
                                            )
                                        ]
                                        total_chunks = len(data_chunks)
                                        logger.info(
                                            f"sending ImageChunk: {len(encoded_data)} bytes in {total_chunks} chunks"
                                        )
                                        for chunk_index, chunk_data in enumerate(
                                            data_chunks
                                        ):
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ImageChunk(
                                                        idx=chunk_index,
                                                        model=shard_metadata.model_meta.model_id,
                                                        data=chunk_data,
                                                        chunk_index=chunk_index,
                                                        total_chunks=total_chunks,
                                                        image_index=image_index,
                                                    ),
                                                )
                                            )

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case ImageEdits(task_params=task_params, command_id=command_id) if (
                        isinstance(current_status, RunnerReady)
                    ):
                        assert isinstance(model, ImageGenerator)
                        logger.info(f"received image edits request: {str(task)[:500]}")
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        for image_index, response in enumerate(
                            generate_image(
                                model=model,
                                task=task_params,
                            )
                        ):
                            match response:
                                case ImageGenerationResponse():
                                    if shard_metadata.device_rank == shard_metadata.world_size - 1:
                                        encoded_data = base64.b64encode(
                                            response.image_data
                                        ).decode("utf-8")
                                        # Split into chunks to stay under gossipsub 1MB limit
                                        data_chunks = [
                                            encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
                                            for i in range(
                                                0, len(encoded_data), EXO_MAX_CHUNK_SIZE
                                            )
                                        ]
                                        total_chunks = len(data_chunks)
                                        logger.info(
                                            f"sending ImageChunk: {len(encoded_data)} bytes in {total_chunks} chunks"
                                        )
                                        for chunk_index, chunk_data in enumerate(
                                            data_chunks
                                        ):
                                            is_last_chunk = (
                                                chunk_index == total_chunks - 1
                                            )
                                            event_sender.send(
                                                ChunkGenerated(
                                                    command_id=command_id,
                                                    chunk=ImageChunk(
                                                        idx=chunk_index,
                                                        model=shard_metadata.model_meta.model_id,
                                                        data=chunk_data,
                                                        chunk_index=chunk_index,
                                                        total_chunks=total_chunks,
                                                        image_index=image_index,
                                                    ),
                                                )
                                            )

                        current_status = RunnerReady()
                        logger.info("runner ready")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=RunnerReady()
                            )
                        )
                    case Shutdown():
                        current_status = RunnerShuttingDown()
                        logger.info("runner shutting down")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        current_status = RunnerShutdown()
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                        )
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id, runner_status=current_status
                    )
                )
                if isinstance(current_status, RunnerShutdown):
                    break
    except ClosedResourceError:
        logger.warning("runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        event_sender.close()
        task_receiver.close()
        event_sender.join()
        task_receiver.join()
        logger.info("bye from the runner")


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
