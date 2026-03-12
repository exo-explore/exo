from _typeshed import Incomplete
from fastapi import Request as Request
from fastapi.responses import Response as Response
from vllm.entrypoints.openai.utils import validate_json_request as validate_json_request
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationRequest as ClassificationRequest,
)
from vllm.entrypoints.pooling.classify.serving import (
    ServingClassification as ServingClassification,
)
from vllm.entrypoints.utils import (
    load_aware_call as load_aware_call,
    with_cancellation as with_cancellation,
)

router: Incomplete

def classify(request: Request) -> ServingClassification | None: ...
@with_cancellation
@load_aware_call
async def create_classify(
    request: ClassificationRequest, raw_request: Request
) -> Response: ...
