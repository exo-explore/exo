"""What nodes send each other to serve a request, without the master.

The API node that took the request sends a ServeRequest to the instance's output
node: the node whose runner emits the request's output. That node hands the task to every rank
of the instance (a Delivery to each other node) and streams back ServeFrames, one JSON object
per line: Accepted once every rank holds the task (or Refused, and nothing more), then the
request's chunks. A blank line is a heartbeat.
"""

from typing import Annotated, final

from pydantic import TypeAdapter

from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import NodeId
from exo.shared.types.tasks import GenerationTask
from exo.utils.pydantic_ext import ByTag, FrozenModel, TaggedModel


@final
class ServeRequest(FrozenModel):
    # The output node. A stale address must not reach another node and run the request there.
    target: NodeId
    # With its inputs (images) inline
    task: GenerationTask


@final
class Delivery(FrozenModel):
    """A request's task, for one rank. Cancelled means: you must hold this task, and it is
    cancelled. A rank that never received it starts it and then cancels it, because the ranks
    drop a task only when every rank holds it and its cancellation."""

    task: GenerationTask
    cancelled: bool = False


@final
class Accepted(TaggedModel):
    """Every rank of the instance holds the task, so the ranks will agree to run it."""


@final
class Refused(TaggedModel):
    """The request can't run; nothing of it runs anywhere. One sentence for the user."""

    message: str


# Every chunk a request's output can be (exo.shared.types.chunks.Chunk), validated by tag: one
# line per token of every reply
ServeFrame = Annotated[
    Accepted
    | Refused
    | TokenChunk
    | ImageChunk
    | ToolCallChunk
    | ErrorChunk
    | PrefillProgressChunk,
    ByTag(),
]

serve_frames: TypeAdapter[ServeFrame] = TypeAdapter(ServeFrame)
