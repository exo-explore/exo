"""A request's stream between nodes (exo.shared.types.serving) carries every kind of chunk intact,
including a failure's diagnostics: a frame a node can't read would end the request."""

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    PrefillProgressChunk,
    TokenChunk,
)
from exo.shared.types.serving import Accepted, Refused, ServeFrame, serve_frames
from exo.worker.runner.diagnostics import (
    RunnerMetalGpuTimeout,
    RunnerMetalOutOfMemory,
    RunnerRingSocketReceivingError,
    RunnerRingTransportError,
)

MODEL = ModelId("test-model")


@pytest.mark.parametrize(
    "frame",
    [
        Accepted(),
        Refused(message="A device serving this model didn't take the request."),
        TokenChunk(model=MODEL, text="héllo\n", token_id=7, usage=None),
        PrefillProgressChunk(model=MODEL, processed_tokens=1, total_tokens=2),
        ImageChunk(
            model=MODEL, data="aGk=", chunk_index=0, total_chunks=1, image_index=0
        ),
        *(
            ErrorChunk(
                model=MODEL,
                error_message="The model stopped unexpectedly before the reply finished.",
                diagnostics=[diagnostic],
            )
            for diagnostic in (
                RunnerRingTransportError(message="aborted", evidence=("a", "b")),
                RunnerRingSocketReceivingError(
                    message="failed",
                    evidence=("c",),
                    error_number=54,
                    error_name="ECONNRESET",
                    error_description="Connection reset by peer",
                ),
                RunnerMetalGpuTimeout(message="timeout"),
                RunnerMetalOutOfMemory(message="out of memory", evidence=("d",)),
            )
        ),
    ],
)
def test_a_frame_reads_back_as_it_was_sent(frame: ServeFrame) -> None:
    line = serve_frames.dump_json(frame)
    assert b"\n" not in line, "one frame is one line"
    assert serve_frames.validate_json(line) == frame
