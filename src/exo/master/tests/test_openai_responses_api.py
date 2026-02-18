"""Tests for OpenAI Responses API wire types.

ResponsesRequest is the API wire type for the Responses endpoint.
The responses adapter converts it to TextGenerationTaskParams for the pipeline.
"""

import pydantic
import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.openai_responses import (
    ResponseInputMessage,
    ResponsesRequest,
)


class TestResponsesRequestValidation:
    """Tests for OpenAI Responses API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ResponsesRequest.model_validate(
                {
                    "input": "Hello",
                }
            )

    def test_request_requires_input(self):
        with pytest.raises(pydantic.ValidationError):
            ResponsesRequest.model_validate(
                {
                    "model": "gpt-4o",
                }
            )

    def test_request_accepts_string_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello",
        )
        assert request.input == "Hello"

    def test_request_accepts_message_array_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input=[ResponseInputMessage(role="user", content="Hello")],
        )
        assert len(request.input) == 1
