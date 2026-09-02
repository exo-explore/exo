from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    Base64Image,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.generator.batch_generate import (
    UnsupportedRequestError,
    check_vision_support,
)


def _params(images: list[Base64Image]) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        images=images,
    )


def test_images_without_vision_processor_are_rejected() -> None:
    try:
        check_vision_support(_params([Base64Image("data:image/png;base64,AAAA")]), None)
    except UnsupportedRequestError as e:
        assert "vision processor" in str(e)
    else:
        raise AssertionError("expected UnsupportedRequestError")


def test_text_only_request_passes_without_vision_processor() -> None:
    check_vision_support(_params([]), None)


def test_unsupported_request_error_is_a_value_error() -> None:
    # Callers that already handle ValueError keep working.
    assert issubclass(UnsupportedRequestError, ValueError)
