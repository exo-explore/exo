from typing import Final

from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import InstanceId, RunnerId

MASTER_NODE_ID = NodeId("ffffffff-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

NODE_A: Final[NodeId] = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_B: Final[NodeId] = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

RUNNER_1_ID: Final[RunnerId] = RunnerId("11111111-1111-4111-8111-111111111111")
RUNNER_2_ID: Final[RunnerId] = RunnerId("33333333-3333-4333-8333-333333333333")

INSTANCE_1_ID: Final[InstanceId] = InstanceId("22222222-2222-4222-8222-222222222222")
INSTANCE_2_ID: Final[InstanceId] = InstanceId("44444444-4444-4444-8444-444444444444")

MODEL_A_ID: Final[ModelId] = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")
MODEL_B_ID: Final[ModelId] = ModelId("mlx-community/TinyLlama-1.1B-Chat-v1.0")

TASK_1_ID: Final[TaskId] = TaskId("55555555-5555-4555-8555-555555555555")
TASK_2_ID: Final[TaskId] = TaskId("66666666-6666-4666-8666-666666666666")

COMMAND_1_ID: Final[CommandId] = CommandId("77777777-7777-4777-8777-777777777777")
COMMAND_2_ID: Final[CommandId] = CommandId("88888888-8888-4888-8888-888888888888")
