from collections.abc import AsyncIterable
from http import HTTPStatus

from starlette.responses import StreamingResponse
from starlette.types import Send

from exo.api.types import ErrorInfo, ErrorResponse
from exo.shared.request_errors import GenerationErrorCode, generation_error_code


class GenerationFailedError(ValueError):
    """A generation that ended with an error, and the stable code saying why"""

    def __init__(self, message: str):
        super().__init__(message)
        self.code: GenerationErrorCode | None = generation_error_code(message)


# The HTTP status of a whole response whose generation failed, for the errors that aren't
# the server's fault (500 otherwise)
_STATUS_OF_CODE: dict[GenerationErrorCode, HTTPStatus] = {
    # No ready instance by the time the request reached the master
    "model_not_ready": HTTPStatus.SERVICE_UNAVAILABLE,
}


class CollectedResponse(StreamingResponse):
    """A non-streaming response that sends nothing until its body is complete.

    The body comes from a collector that gathers a whole generation, raising ValueError if it
    fails. A StreamingResponse would already have sent "200 OK" by then, so the client got an
    empty 200; this sends a 500 with the error instead. Like a StreamingResponse, it stops
    collecting, which cancels the generation, if the client disconnects first.
    """

    async def stream_response(self, send: Send) -> None:
        try:
            body = "".join([chunk async for chunk in self._collect()])
        except ValueError as error:
            code = error.code if isinstance(error, GenerationFailedError) else None
            status = (
                _STATUS_OF_CODE.get(code, HTTPStatus.INTERNAL_SERVER_ERROR)
                if code is not None
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            self.status_code = status
            body = ErrorResponse(
                error=ErrorInfo(
                    message=str(error), type=status.phrase, code=code or status
                )
            ).model_dump_json()
        content = body.encode(self.charset)
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [
                    *self.raw_headers,
                    (b"content-length", str(len(content)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": content})

    async def _collect(self) -> AsyncIterable[str]:
        async for chunk in self.body_iterator:
            yield chunk if isinstance(chunk, str) else bytes(chunk).decode(self.charset)
