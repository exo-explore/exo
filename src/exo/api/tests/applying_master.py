"""A stand-in for the master in API tests: it applies the launches it is sent to the API's state,
so a request that waits for its change to show (POST /instance, say) returns."""

from collections.abc import Awaitable, Callable

import anyio

from exo.api.main import API
from exo.shared.types.commands import Command, CreateInstance, ForwarderCommand
from exo.utils.channels import Receiver


async def sent_while_applying[T](
    api: API, commands: Receiver[ForwarderCommand], request: Callable[[], Awaitable[T]]
) -> tuple[T, list[Command]]:
    """Make the request, applying each launch it sends, and return its answer and the commands
    it sent."""
    answers: list[T] = []
    sent: list[Command] = []

    async def make_request() -> None:
        answers.append(await request())

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(make_request)
            while not answers:
                with anyio.move_on_after(0.05):
                    command = (await commands.receive()).command
                    sent.append(command)
                    if isinstance(command, CreateInstance):
                        api.state = api.state.model_copy(
                            update={
                                "instances": {
                                    **api.state.instances,
                                    command.instance.instance_id: command.instance,
                                }
                            }
                        )
                        api._note_applied_cluster_commands()  # pyright: ignore[reportPrivateUsage]
    return answers[0], sent
