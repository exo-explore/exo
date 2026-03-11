class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    logprobs: object | None
    finish_reason: str | None
    stop_reason: int | str | None

    def finished(self) -> bool: ...

class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    outputs: list[CompletionOutput]
    finished: bool
