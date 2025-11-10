import time

from exo.engines.mlx.utils_mlx import (
    mx_barrier,
    initialize_mlx,
    mlx_force_oom,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.commands_runner import (
    GenerationResponse,
    TokenizedResponse,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWaitingForModel,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.generate import mlx_generate, warmup_inference


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard(),
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

        current_status: RunnerStatus = RunnerWaitingForModel()
        logger.info("runner waiting for model")
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
                    case LoadModel() if isinstance(
                        current_status, (RunnerWaitingForModel, RunnerFailed)
                    ):
                        current_status = RunnerLoading()
                        logger.info("runner loading")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        model, tokenizer, sampler = initialize_mlx(bound_instance)

                        current_status = RunnerLoaded()
                        logger.info("runner loaded")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        assert model
                        assert tokenizer
                        assert sampler
                        current_status = RunnerWarmingUp()
                        logger.info("runner warming up")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )

                        logger.info(f"warming up inference for instance: {instance}")
                        toks = warmup_inference(
                            model=model,
                            tokenizer=tokenizer,
                            sampler=sampler,
                        )
                        logger.info(f"warmed up by generating {toks} tokens")
                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                        current_status = RunnerReady()
                        logger.info("runner ready")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=RunnerReady()
                            )
                        )
                    case ChatCompletion(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert model
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
                        # Ensure we have a chat-completion task subtype
                        # TODO: this is a hack, why are we only looking at the first message? should have a tokenizer
                        prompt = task_params.messages[0]
                        if (
                            prompt.content is not None
                            and "EXO RUNNER MUST FAIL" in prompt.content
                        ):
                            logger.info("raising exception")
                            raise Exception(
                                "Artificial runner exception - for testing purposes only."
                            )
                        if (
                            prompt.content is not None
                            and "EXO RUNNER MUST OOM" in prompt.content
                        ):
                            mlx_force_oom()
                        if (
                            prompt.content is not None
                            and "EXO RUNNER MUST TIMEOUT" in prompt.content
                        ):
                            time.sleep(100)

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
                                case TokenizedResponse():
                                    # TODO: something here ig
                                    logger.info("Finished tokenizing?")

                        current_status = RunnerReady()
                        logger.info("runner ready")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=RunnerReady()
                            )
                        )
                    case Shutdown():
                        break
                    case _:
                        raise ValueError("Received task outside of state machine")
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )

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
