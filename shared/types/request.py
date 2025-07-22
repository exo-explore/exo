from pydantic import BaseModel

from shared.types.api import ChatCompletionTaskParams
from shared.types.common import NewUUID


class RequestId(NewUUID):
    pass

class APIRequest(BaseModel):
    request_id: RequestId
    request_params: ChatCompletionTaskParams