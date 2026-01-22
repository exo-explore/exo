# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any, get_args

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.shared.types.api import ErrorInfo, ErrorResponse, FinishReason
from exo.shared.types.chunks import ImageChunk, TokenChunk
from exo.worker.tests.constants import MODEL_A_ID


def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""
    from exo.master.api import API

    app = FastAPI()

    # Setup exception handler
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    # Add test routes that raise HTTPException
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # Test 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # Test 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_finish_reason_includes_error() -> None:
    valid_reasons = get_args(FinishReason)
    assert "error" in valid_reasons


def test_token_chunk_with_error_fields() -> None:
    chunk = TokenChunk(
        idx=0,
        model=MODEL_A_ID,
        text="",
        token_id=0,
        finish_reason="error",
        error_message="Something went wrong",
    )

    assert chunk.finish_reason == "error"
    assert chunk.error_message == "Something went wrong"


def test_token_chunk_without_error() -> None:
    chunk = TokenChunk(
        idx=1,
        model=MODEL_A_ID,
        text="Hello",
        token_id=42,
        finish_reason=None,
    )

    assert chunk.finish_reason is None
    assert chunk.error_message is None


def test_error_response_construction() -> None:
    error_response = ErrorResponse(
        error=ErrorInfo(
            message="Generation failed",
            type="InternalServerError",
            code=500,
        )
    )

    assert error_response.error.message == "Generation failed"
    assert error_response.error.code == 500


def test_normal_finish_reasons_still_work() -> None:
    for reason in ["stop", "length", "tool_calls", "content_filter", "function_call"]:
        chunk = TokenChunk(
            idx=0,
            model=MODEL_A_ID,
            text="done",
            token_id=100,
            finish_reason=reason,  # type: ignore[arg-type]
        )
        assert chunk.finish_reason == reason


def test_image_chunk_with_error_fields() -> None:
    chunk = ImageChunk(
        idx=0,
        model=MODEL_A_ID,
        data="",
        chunk_index=0,
        total_chunks=1,
        image_index=0,
        finish_reason="error",
        error_message="Image generation failed",
    )

    assert chunk.finish_reason == "error"
    assert chunk.error_message == "Image generation failed"
    assert chunk.data == ""
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 1
    assert chunk.image_index == 0


def test_image_chunk_without_error() -> None:
    chunk = ImageChunk(
        idx=0,
        model=MODEL_A_ID,
        data="base64encodeddata",
        chunk_index=0,
        total_chunks=1,
        image_index=0,
    )

    assert chunk.finish_reason is None
    assert chunk.error_message is None
    assert chunk.data == "base64encodeddata"
