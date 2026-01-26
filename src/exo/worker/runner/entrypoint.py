from exo.shared.types.worker.instances import CudaSingleInstance, LlamaRpcInstance
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.utils.channels import MpReceiver, MpSender


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    if isinstance(bound_instance.instance, CudaSingleInstance):
        from exo.worker.runner.cuda_runner import main as cuda_main

        return cuda_main(bound_instance, event_sender, task_receiver)

    if isinstance(bound_instance.instance, LlamaRpcInstance):
        from exo.worker.runner.llama_cpp_runner import main as llama_main

        return llama_main(bound_instance, event_sender, task_receiver)

    from exo.worker.runner.runner import main as mlx_main

    return mlx_main(bound_instance, event_sender, task_receiver)
